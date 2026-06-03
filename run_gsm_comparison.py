"""
GSM Pipeline with Weighted vs Mean Pooling Comparison
- Extracts topic-word probabilities from CTMNeg
- Runs both mean and weighted pooling
- Compares clustering quality
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
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path('results_bukhari')
OUTPUT_DIR.mkdir(exist_ok=True)


print("=" * 60)
print("STEP 1: Training CTMNeg (K=20)")
print("=" * 60)

dataset = Dataset()
dataset.load_custom_dataset_from_folder('preprocessed_datasets/sahih_bukhari')
corpus = dataset.get_corpus()
vocab = dataset.get_vocabulary()
print(f"Docs: {len(corpus)}, Vocab: {len(vocab)}")

model = CTMN2(
    num_topics=20,
    num_epochs=50,
    bert_model='aubmindlab/bert-base-arabertv02',
    inference_type='combined',
    topic_perturb=1,
    tloss_weight=1.0,
)

results = model.train_model(dataset)
print("Training done!")

# Extract topic-word probability matrix (beta)
# The internal model stores this in the decoder weights
inner_model = model.model
beta = inner_model.model.get_beta().detach().cpu().numpy()
print(f"Beta matrix shape: {beta.shape}")  # Should be [num_topics x vocab_size]

# Get vocabulary mapping
train_vocab = inner_model.train_data.cv.get_feature_names_out()
print(f"Train vocab size: {len(train_vocab)}")

# Build topic-word with probabilities
topics_with_probs = []
for topic_idx in range(beta.shape[0]):
    topic_probs = beta[topic_idx]
    # Softmax to get probabilities
    topic_probs = np.exp(topic_probs) / np.exp(topic_probs).sum()
    
    # Sort by probability
    sorted_indices = np.argsort(topic_probs)[::-1]
    
    topic_words = []
    for word_idx in sorted_indices[:15]:
        word = train_vocab[word_idx]
        prob = float(topic_probs[word_idx])
        topic_words.append((word, prob))
    
    topics_with_probs.append(topic_words)

print("\nSample Topic with Probabilities:")
for word, prob in topics_with_probs[0][:5]:
    print(f"  {word}: {prob:.4f}")

# ============================================
# STEP 2: Load CamelBERT for word embeddings
# ============================================
print("\n" + "=" * 60)
print("STEP 2: Loading CamelBERT")
print("=" * 60)

word_model = SentenceTransformer('CAMeL-Lab/bert-base-arabic-camelbert-da')
print("CamelBERT loaded!")

# ============================================
# STEP 3: Compute topic vectors - BOTH methods
# ============================================
print("\n" + "=" * 60)
print("STEP 3: Computing Topic Vectors")
print("=" * 60)

def compute_vectors_mean(topics_wp, word_model, top_k=10):
    """BASELINE: Simple mean pooling (all words equal)"""
    vectors = []
    for topic in topics_wp:
        words = [w for w, p in topic[:top_k]]
        embeddings = word_model.encode(words)
        vector = np.mean(embeddings, axis=0)
        vectors.append(vector)
    return np.array(vectors)

def compute_vectors_weighted(topics_wp, word_model, top_k=10):
    """PROPOSED: Probability-weighted mean pooling"""
    vectors = []
    for topic in topics_wp:
        words = [w for w, p in topic[:top_k]]
        probs = np.array([p for w, p in topic[:top_k]])
        probs = probs / probs.sum()  # normalize
        
        embeddings = word_model.encode(words)
        vector = np.average(embeddings, axis=0, weights=probs)
        vectors.append(vector)
    return np.array(vectors)

print("Computing MEAN pooling vectors...")
vectors_mean = compute_vectors_mean(topics_with_probs, word_model)
print(f"  Shape: {vectors_mean.shape}")

print("Computing WEIGHTED pooling vectors...")
vectors_weighted = compute_vectors_weighted(topics_with_probs, word_model)
print(f"  Shape: {vectors_weighted.shape}")

# ============================================
# STEP 4: Cluster with both methods and compare
# ============================================
print("\n" + "=" * 60)
print("STEP 4: Clustering Comparison")
print("=" * 60)

def run_clustering(vectors, method_name, corpus, topics_wp):
    """Run K-Means clustering and compute metrics."""
    # Find optimal K
    best_sil = -1
    best_k = 5
    sil_scores = {}
    
    for k in range(4, min(16, len(vectors))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(vectors)
        sil = silhouette_score(vectors, labels)
        sil_scores[k] = sil
        if sil > best_sil:
            best_sil = sil
            best_k = k
    
    # Cluster with best K
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(vectors)
    
    # Intra-cluster similarity (higher = tighter clusters)
    intra_sims = []
    for c in range(best_k):
        cluster_vectors = vectors[labels == c]
        if len(cluster_vectors) > 1:
            sim_matrix = cosine_similarity(cluster_vectors)
            # Average of upper triangle
            n = len(cluster_vectors)
            total = 0
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    total += sim_matrix[i][j]
                    count += 1
            if count > 0:
                intra_sims.append(total / count)
    avg_intra = np.mean(intra_sims) if intra_sims else 0
    
    # Inter-cluster distance (higher = more separated)
    centroids = km.cluster_centers_
    inter_dists = []
    for i in range(best_k):
        for j in range(i+1, best_k):
            dist = 1 - cosine_similarity([centroids[i]], [centroids[j]])[0][0]
            inter_dists.append(dist)
    avg_inter = np.mean(inter_dists) if inter_dists else 0
    
    # Build clusters
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    dictionary = Dictionary(corpus)
    
    clusters = []
    for c in range(best_k):
        topic_indices = [i for i, l in enumerate(labels) if l == c]
        all_words = []
        for ti in topic_indices:
            all_words.extend([w for w, p in topics_wp[ti][:10]])
        word_counts = Counter(all_words)
        top_words = [w for w, _ in word_counts.most_common(10)]
        
        # CV for cluster
        filtered = [w for w in top_words if w in dictionary.token2id]
        try:
            if len(filtered) >= 2:
                cm = CoherenceModel(topics=[filtered], texts=corpus,
                                   dictionary=dictionary, coherence='c_v')
                cv = cm.get_coherence()
            else:
                cv = 0.0
        except:
            cv = 0.0
        
        clusters.append({
            'cluster_id': c,
            'num_topics': len(topic_indices),
            'topic_indices': topic_indices,
            'top_words': top_words,
            'cv': cv,
        })
    
    # Overall metrics
    all_cvs = [c['cv'] for c in clusters]
    valid_clusters = [c for c in clusters if c['cv'] >= 0.35]
    
    result = {
        'method': method_name,
        'num_clusters': best_k,
        'silhouette': best_sil,
        'silhouette_all': sil_scores,
        'avg_intra_similarity': avg_intra,
        'avg_inter_distance': avg_inter,
        'avg_cv_all': np.mean(all_cvs),
        'avg_cv_filtered': np.mean([c['cv'] for c in valid_clusters]) if valid_clusters else 0,
        'num_valid_clusters': len(valid_clusters),
        'clusters': clusters,
    }
    
    return result

print("\nRunning MEAN pooling clustering...")
result_mean = run_clustering(vectors_mean, "mean_pooling", corpus, topics_with_probs)

print("Running WEIGHTED pooling clustering...")
result_weighted = run_clustering(vectors_weighted, "weighted_pooling", corpus, topics_with_probs)


print("\n" + "=" * 60)
print("COMPARISON: MEAN vs WEIGHTED POOLING")
print("=" * 60)

print(f"\n{'Metric':<30} {'Mean (Baseline)':<20} {'Weighted (Proposed)':<20}")
print("-" * 70)
print(f"{'Clusters':<30} {result_mean['num_clusters']:<20} {result_weighted['num_clusters']:<20}")
print(f"{'Silhouette':<30} {result_mean['silhouette']:<20.4f} {result_weighted['silhouette']:<20.4f}")
print(f"{'Intra-cluster Similarity':<30} {result_mean['avg_intra_similarity']:<20.4f} {result_weighted['avg_intra_similarity']:<20.4f}")
print(f"{'Inter-cluster Distance':<30} {result_mean['avg_inter_distance']:<20.4f} {result_weighted['avg_inter_distance']:<20.4f}")
print(f"{'Avg CV (all clusters)':<30} {result_mean['avg_cv_all']:<20.4f} {result_weighted['avg_cv_all']:<20.4f}")
print(f"{'Avg CV (filtered)':<30} {result_mean['avg_cv_filtered']:<20.4f} {result_weighted['avg_cv_filtered']:<20.4f}")
print(f"{'Valid Clusters':<30} {result_mean['num_valid_clusters']:<20} {result_weighted['num_valid_clusters']:<20}")

# Improvements
sil_imp = ((result_weighted['silhouette'] - result_mean['silhouette']) / abs(result_mean['silhouette'])) * 100 if result_mean['silhouette'] != 0 else 0
intra_imp = ((result_weighted['avg_intra_similarity'] - result_mean['avg_intra_similarity']) / abs(result_mean['avg_intra_similarity'])) * 100 if result_mean['avg_intra_similarity'] != 0 else 0
cv_imp = ((result_weighted['avg_cv_filtered'] - result_mean['avg_cv_filtered']) / abs(result_mean['avg_cv_filtered'])) * 100 if result_mean['avg_cv_filtered'] != 0 else 0

print(f"\n{'IMPROVEMENTS:'}")
print(f"  Silhouette:    {sil_imp:+.1f}%")
print(f"  Intra-cluster: {intra_imp:+.1f}%")
print(f"  CV filtered:   {cv_imp:+.1f}%")


print("\n" + "=" * 60)
print("WEIGHTED POOLING CLUSTERS (Proposed Method)")
print("=" * 60)

for c in sorted(result_weighted['clusters'], key=lambda x: x['cv'], reverse=True):
    status = "KEPT" if c['cv'] >= 0.35 else "REMOVED"
    print(f"\n  Cluster {c['cluster_id']} ({c['num_topics']} topics, CV={c['cv']:.4f}) [{status}]")
    print(f"    Words: {' | '.join(c['top_words'][:8])}")
    print(f"    Topics: {c['topic_indices']}")


print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Topics with probabilities
topics_save = []
for i, topic in enumerate(topics_with_probs):
    topics_save.append({
        'topic_id': i,
        'words': [{'word': w, 'prob': float(p)} for w, p in topic[:15]]
    })

with open(OUTPUT_DIR / 'topics_with_probs.json', 'w', encoding='utf-8') as f:
    json.dump(topics_save, f, ensure_ascii=False, indent=2)

# Comparison results
comparison = {
    'mean_pooling': {
        'silhouette': result_mean['silhouette'],
        'intra_similarity': result_mean['avg_intra_similarity'],
        'inter_distance': result_mean['avg_inter_distance'],
        'avg_cv_all': result_mean['avg_cv_all'],
        'avg_cv_filtered': result_mean['avg_cv_filtered'],
        'num_clusters': result_mean['num_clusters'],
        'num_valid': result_mean['num_valid_clusters'],
    },
    'weighted_pooling': {
        'silhouette': result_weighted['silhouette'],
        'intra_similarity': result_weighted['avg_intra_similarity'],
        'inter_distance': result_weighted['avg_inter_distance'],
        'avg_cv_all': result_weighted['avg_cv_all'],
        'avg_cv_filtered': result_weighted['avg_cv_filtered'],
        'num_clusters': result_weighted['num_clusters'],
        'num_valid': result_weighted['num_valid_clusters'],
    },
    'improvement': {
        'silhouette_pct': sil_imp,
        'intra_pct': intra_imp,
        'cv_filtered_pct': cv_imp,
    }
}

with open(OUTPUT_DIR / 'pooling_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

# Save weighted clustering details
with open(OUTPUT_DIR / 'gsm_weighted.json', 'w', encoding='utf-8') as f:
    weighted_save = {
        'method': 'probability_weighted_pooling',
        'num_clusters': result_weighted['num_clusters'],
        'silhouette': result_weighted['silhouette'],
        'clusters': result_weighted['clusters'],
    }
    json.dump(weighted_save, f, ensure_ascii=False, indent=2, default=str)

# Save readable comparison
with open(OUTPUT_DIR / 'COMPARISON_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write("MEAN vs WEIGHTED POOLING COMPARISON\n")
    f.write("Sahih Bukhari - CTMNeg (K=20)\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"{'Metric':<30} {'Mean':<15} {'Weighted':<15} {'Change':<10}\n")
    f.write("-" * 70 + "\n")
    f.write(f"{'Clusters':<30} {result_mean['num_clusters']:<15} {result_weighted['num_clusters']:<15}\n")
    f.write(f"{'Silhouette':<30} {result_mean['silhouette']:<15.4f} {result_weighted['silhouette']:<15.4f} {sil_imp:+.1f}%\n")
    f.write(f"{'Intra-cluster Sim':<30} {result_mean['avg_intra_similarity']:<15.4f} {result_weighted['avg_intra_similarity']:<15.4f} {intra_imp:+.1f}%\n")
    f.write(f"{'Inter-cluster Dist':<30} {result_mean['avg_inter_distance']:<15.4f} {result_weighted['avg_inter_distance']:<15.4f}\n")
    f.write(f"{'Avg CV (all)':<30} {result_mean['avg_cv_all']:<15.4f} {result_weighted['avg_cv_all']:<15.4f}\n")
    f.write(f"{'Avg CV (filtered)':<30} {result_mean['avg_cv_filtered']:<15.4f} {result_weighted['avg_cv_filtered']:<15.4f} {cv_imp:+.1f}%\n")
    f.write(f"{'Valid Clusters':<30} {result_mean['num_valid_clusters']:<15} {result_weighted['num_valid_clusters']:<15}\n")
    
    f.write(f"\nWEIGHTED POOLING CLUSTERS:\n")
    f.write("-" * 40 + "\n")
    for c in sorted(result_weighted['clusters'], key=lambda x: x['cv'], reverse=True):
        status = "KEPT" if c['cv'] >= 0.35 else "REMOVED"
        f.write(f"\n  Cluster {c['cluster_id']} (CV={c['cv']:.4f}, {c['num_topics']} topics) [{status}]\n")
        f.write(f"    {' | '.join(c['top_words'][:8])}\n")

print(f"\nFiles saved in {OUTPUT_DIR}/:")
print(f"  - topics_with_probs.json")
print(f"  - pooling_comparison.json")
print(f"  - gsm_weighted.json")
print(f"  - COMPARISON_REPORT.txt")

print(f"\n{'=' * 60}")
print("ALL DONE!")
print(f"{'=' * 60}")
