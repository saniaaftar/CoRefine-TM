"""
Complete Hadith Topic Modeling Pipeline
- Tests multiple K values
- Computes CV, NPMI, TD
- Runs GSM (grouping + scoring)
- Saves all results for professor
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, '.')

from octis.dataset.dataset import Dataset
from octis.models.CTMN2 import CTMN2
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================
# CONFIG
# ============================================
DATASET_PATH = 'preprocessed_datasets/sahih_bukhari'
K_VALUES = [20, 30, 40, 50]
BERT_MODEL = 'aubmindlab/bert-base-arabertv02'
NUM_EPOCHS = 50
OUTPUT_DIR = Path('results_bukhari')
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================
# LOAD DATASET
# ============================================
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)
dataset = Dataset()
dataset.load_custom_dataset_from_folder(DATASET_PATH)
corpus = dataset.get_corpus()
vocab = dataset.get_vocabulary()
print(f"Docs: {len(corpus)}, Vocab: {len(vocab)}")

# Metrics objects
cv_metric = Coherence(texts=corpus, topk=10, measure='c_v')
npmi_metric = Coherence(texts=corpus, topk=10, measure='c_npmi')
td_metric = TopicDiversity(topk=10)

# ============================================
# K SELECTION - Test multiple K values
# ============================================
all_k_results = []
best_cv = -1
best_k = 20
best_results = None
best_model = None

print("\n" + "=" * 60)
print("K SELECTION")
print(f"Testing K = {K_VALUES}")
print("=" * 60)

for k in K_VALUES:
    print(f"\n{'─' * 50}")
    print(f"Training K = {k}")
    print(f"{'─' * 50}")
    
    model = CTMN2(
        num_topics=k,
        num_epochs=NUM_EPOCHS,
        bert_model=BERT_MODEL,
        inference_type='combined',
        topic_perturb=1,
        tloss_weight=1.0,
    )
    
    results = model.train_model(dataset)
    
    cv_score = cv_metric.score(results)
    npmi_score = npmi_metric.score(results)
    td_score = td_metric.score(results)
    
    print(f"\n  K={k}: CV={cv_score:.4f}  NPMI={npmi_score:.4f}  TD={td_score:.4f}")
    
    k_result = {
        'k': k,
        'cv': cv_score,
        'npmi': npmi_score,
        'td': td_score,
        'topics': results['topics'],
    }
    all_k_results.append(k_result)
    
    if cv_score > best_cv:
        best_cv = cv_score
        best_k = k
        best_results = results
        best_model = model

print(f"\n{'=' * 60}")
print(f"OPTIMAL K = {best_k} (CV = {best_cv:.4f})")
print(f"{'=' * 60}")

# Save K selection results
k_selection = {
    'k_values': [r['k'] for r in all_k_results],
    'cv_scores': [r['cv'] for r in all_k_results],
    'npmi_scores': [r['npmi'] for r in all_k_results],
    'td_scores': [r['td'] for r in all_k_results],
    'best_k': best_k,
    'best_cv': best_cv,
}
with open(OUTPUT_DIR / 'k_selection.json', 'w') as f:
    json.dump(k_selection, f, indent=2)

# ============================================
# K SELECTION PLOT
# ============================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ks = k_selection['k_values']
    
    axes[0].plot(ks, k_selection['cv_scores'], 'b-o', linewidth=2, markersize=8)
    axes[0].axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    axes[0].set_xlabel('K'); axes[0].set_ylabel('CV Coherence')
    axes[0].set_title('CV Coherence vs K'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(ks, k_selection['npmi_scores'], 'g-s', linewidth=2, markersize=8)
    axes[1].axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    axes[1].set_xlabel('K'); axes[1].set_ylabel('NPMI')
    axes[1].set_title('NPMI vs K'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(ks, k_selection['td_scores'], 'r-^', linewidth=2, markersize=8)
    axes[2].axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    axes[2].set_xlabel('K'); axes[2].set_ylabel('Topic Diversity')
    axes[2].set_title('Topic Diversity vs K'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / 'k_selection_plot.png'), dpi=300)
    plt.close()
    print(f"Plot saved: {OUTPUT_DIR / 'k_selection_plot.png'}")
except Exception as e:
    print(f"Plot error (non-critical): {e}")

# ============================================
# GSM - G COMPONENT: Topic Clustering
# ============================================
print(f"\n{'=' * 60}")
print(f"GSM PIPELINE (K={best_k})")
print(f"{'=' * 60}")

# Get topic-word distributions from best model
topics = best_results['topics']  # list of lists of words
print(f"Total topics: {len(topics)}")

# Try to get topic vectors using sentence transformer
try:
    from sentence_transformers import SentenceTransformer
    
    print("Loading CamelBERT for word embeddings...")
    word_model = SentenceTransformer('CAMeL-Lab/bert-base-arabic-camelbert-da')
    
    # Compute topic vectors (weighted by position - top words get more weight)
    topic_vectors = []
    for i, topic_words in enumerate(topics):
        top10 = topic_words[:10]
        # Weight by rank: top word gets weight 10, second 9, etc.
        weights = np.array([10 - j for j in range(len(top10))], dtype=float)
        weights = weights / weights.sum()
        
        word_embeddings = word_model.encode(top10)
        weighted_vector = np.average(word_embeddings, axis=0, weights=weights)
        topic_vectors.append(weighted_vector)
    
    topic_vectors = np.array(topic_vectors)
    print(f"Topic vectors shape: {topic_vectors.shape}")
    
    # Find optimal number of clusters
    best_sil = -1
    best_n_clusters = 5
    sil_results = {}
    
    for nc in range(5, min(16, len(topics))):
        km = KMeans(n_clusters=nc, random_state=42, n_init=10)
        labels = km.fit_predict(topic_vectors)
        sil = silhouette_score(topic_vectors, labels)
        sil_results[nc] = sil
        if sil > best_sil:
            best_sil = sil
            best_n_clusters = nc
    
    print(f"Optimal clusters: {best_n_clusters} (silhouette: {best_sil:.4f})")
    
    # Cluster topics
    km = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(topic_vectors)
    
    # Build cluster summary
    clusters = {}
    for topic_idx, cluster_id in enumerate(cluster_labels):
        cluster_id = int(cluster_id)
        if cluster_id not in clusters:
            clusters[cluster_id] = {'topic_indices': [], 'all_words': []}
        clusters[cluster_id]['topic_indices'].append(topic_idx)
        clusters[cluster_id]['all_words'].extend(topics[topic_idx][:10])
    
    # Compute cluster coherence
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    
    dictionary = Dictionary(corpus)
    
    print(f"\n{'─' * 50}")
    print(f"CLUSTER RESULTS")
    print(f"{'─' * 50}")
    
    cluster_details = []
    for cid in sorted(clusters.keys()):
        c = clusters[cid]
        # Get unique top words for this cluster
        word_counts = Counter(c['all_words'])
        top_words = [w for w, _ in word_counts.most_common(10)]
        
        # Compute coherence
        filtered_words = [w for w in top_words if w in dictionary.token2id]
        if len(filtered_words) >= 2:
            try:
                cm = CoherenceModel(topics=[filtered_words], texts=corpus,
                                   dictionary=dictionary, coherence='c_v')
                cluster_cv = cm.get_coherence()
            except:
                cluster_cv = 0.0
        else:
            cluster_cv = 0.0
        
        detail = {
            'cluster_id': cid,
            'num_topics': len(c['topic_indices']),
            'topic_indices': c['topic_indices'],
            'top_words': top_words,
            'cv': cluster_cv,
        }
        cluster_details.append(detail)
        
        print(f"\n  Cluster {cid} ({len(c['topic_indices'])} topics, CV={cluster_cv:.4f}):")
        print(f"    Topics: {c['topic_indices']}")
        print(f"    Words: {' | '.join(top_words)}")
    
    # Overall metrics after GSM
    valid_clusters = [c for c in cluster_details if c['cv'] >= 0.4]
    avg_cv = np.mean([c['cv'] for c in cluster_details]) if cluster_details else 0
    avg_cv_filtered = np.mean([c['cv'] for c in valid_clusters]) if valid_clusters else 0
    
    gsm_results = {
        'num_clusters': best_n_clusters,
        'silhouette': best_sil,
        'clusters': cluster_details,
        'avg_cv_all': avg_cv,
        'avg_cv_filtered': avg_cv_filtered,
        'num_valid_clusters': len(valid_clusters),
    }

except Exception as e:
    print(f"GSM error: {e}")
    import traceback
    traceback.print_exc()
    gsm_results = {'error': str(e)}

# ============================================
# SAVE ALL RESULTS
# ============================================
print(f"\n{'=' * 60}")
print("SAVING RESULTS")
print("=" * 60)

# Save topics
topics_file = []
for i, topic in enumerate(topics):
    topics_file.append({
        'topic_id': i,
        'words': topic[:15],
        'cluster': int(cluster_labels[i]) if 'cluster_labels' in dir() else -1,
    })

with open(OUTPUT_DIR / 'topics.json', 'w', encoding='utf-8') as f:
    json.dump(topics_file, f, ensure_ascii=False, indent=2)

# Save GSM results
with open(OUTPUT_DIR / 'gsm_results.json', 'w', encoding='utf-8') as f:
    json.dump(gsm_results, f, ensure_ascii=False, indent=2, default=str)

# Save summary
summary = {
    'dataset': 'Sahih Bukhari',
    'total_docs': len(corpus),
    'vocab_size': len(vocab),
    'model': 'CTMNeg (CombinedTM + Negative Sampling)',
    'bert_model': BERT_MODEL,
    'optimal_k': best_k,
    'metrics_raw': {
        'cv': best_cv,
        'npmi': k_selection['npmi_scores'][k_selection['k_values'].index(best_k)],
        'td': k_selection['td_scores'][k_selection['k_values'].index(best_k)],
    },
    'gsm': {
        'num_clusters': gsm_results.get('num_clusters', 0),
        'silhouette': gsm_results.get('silhouette', 0),
        'avg_cv_all': gsm_results.get('avg_cv_all', 0),
        'avg_cv_filtered': gsm_results.get('avg_cv_filtered', 0),
        'num_valid_clusters': gsm_results.get('num_valid_clusters', 0),
    },
    'k_selection': k_selection,
}

with open(OUTPUT_DIR / 'summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

# Save readable text summary
with open(OUTPUT_DIR / 'RESULTS_SUMMARY.txt', 'w', encoding='utf-8') as f:
    f.write("HADITH TOPIC MODELING RESULTS - SAHIH BUKHARI\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset: Sahih Bukhari ({len(corpus)} documents)\n")
    f.write(f"Model: CTMNeg (CombinedTM + Negative Sampling)\n")
    f.write(f"BERT: {BERT_MODEL}\n\n")
    
    f.write("K SELECTION RESULTS:\n")
    f.write("-" * 40 + "\n")
    for r in all_k_results:
        f.write(f"  K={r['k']:3d}  CV={r['cv']:.4f}  NPMI={r['npmi']:.4f}  TD={r['td']:.4f}\n")
    f.write(f"\n  Optimal K = {best_k} (CV = {best_cv:.4f})\n\n")
    
    f.write(f"RAW TOPIC METRICS (K={best_k}):\n")
    f.write("-" * 40 + "\n")
    f.write(f"  CV Coherence:    {summary['metrics_raw']['cv']:.4f}\n")
    f.write(f"  NPMI:            {summary['metrics_raw']['npmi']:.4f}\n")
    f.write(f"  Topic Diversity: {summary['metrics_raw']['td']:.4f}\n\n")
    
    f.write(f"TOPICS (K={best_k}):\n")
    f.write("-" * 40 + "\n")
    for i, topic in enumerate(topics):
        f.write(f"  Topic {i:2d}: {' | '.join(topic[:8])}\n")
    
    if 'clusters' in gsm_results:
        f.write(f"\nGSM CLUSTERING:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Clusters: {gsm_results['num_clusters']}\n")
        f.write(f"  Silhouette: {gsm_results['silhouette']:.4f}\n")
        f.write(f"  Avg CV (all): {gsm_results['avg_cv_all']:.4f}\n")
        f.write(f"  Avg CV (filtered): {gsm_results['avg_cv_filtered']:.4f}\n\n")
        
        for c in gsm_results.get('clusters', []):
            f.write(f"  Cluster {c['cluster_id']} ({c['num_topics']} topics, CV={c['cv']:.4f}):\n")
            f.write(f"    {' | '.join(c['top_words'])}\n\n")

print(f"\nAll results saved in: {OUTPUT_DIR}/")
print(f"  - k_selection.json")
print(f"  - k_selection_plot.png")
print(f"  - topics.json")
print(f"  - gsm_results.json")
print(f"  - summary.json")
print(f"  - RESULTS_SUMMARY.txt")

print(f"\n{'=' * 60}")
print("ALL DONE!")
print(f"{'=' * 60}")
