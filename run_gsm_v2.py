"""
GSM Pipeline with Weighted vs Mean Pooling Comparison
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
    num_topics=40,
    num_epochs=50,
    bert_model='aubmindlab/bert-base-arabertv02',
    inference_type='combined',
    topic_perturb=1,
    tloss_weight=1.0,
)

results = model.train_model(dataset)
print("Training done!")


print("\n" + "=" * 60)
print("STEP 2: Extracting Probabilities")
print("=" * 60)

inner = model.model

# Use get_topic_word_mat
beta = inner.get_topic_word_mat()
print(f"Topic-word matrix shape: {beta.shape}")

# Get vocabulary from training data
train_vocab = [inner.train_data.idx2token[i] for i in range(len(inner.train_data.idx2token))]
print(f"Train vocab size: {len(train_vocab)}")

# Build topics with probabilities
topics_with_probs = []
for topic_idx in range(beta.shape[0]):
    topic_probs = beta[topic_idx]
    sorted_indices = np.argsort(topic_probs)[::-1]
    
    topic_words = []
    for word_idx in sorted_indices[:15]:
        word = train_vocab[word_idx]
        prob = float(topic_probs[word_idx])
        topic_words.append((word, prob))
    topics_with_probs.append(topic_words)

print("\nSample Topics with Probabilities:")
for i in range(3):
    print(f"\nTopic {i}:")
    for word, prob in topics_with_probs[i][:5]:
        print(f"  {word}: {prob:.4f}")

probs_topic0 = [p for _, p in topics_with_probs[0][:10]]
print(f"\nTopic 0 prob range: {min(probs_topic0):.4f} - {max(probs_topic0):.4f}")


print("\n" + "=" * 60)
print("STEP 3: Loading CamelBERT")
print("=" * 60)

word_model = SentenceTransformer('CAMeL-Lab/bert-base-arabic-camelbert-da')
print("CamelBERT loaded!")


print("\n" + "=" * 60)
print("STEP 4: Computing Topic Vectors")
print("=" * 60)

def compute_vectors_mean(topics_wp, wmodel, top_k=10):
    vectors = []
    for topic in topics_wp:
        words = [w for w, p in topic[:top_k]]
        embeddings = wmodel.encode(words)
        vector = np.mean(embeddings, axis=0)
        vectors.append(vector)
    return np.array(vectors)

def compute_vectors_weighted(topics_wp, wmodel, top_k=10):
    vectors = []
    for topic in topics_wp:
        words = [w for w, p in topic[:top_k]]
        probs = np.array([p for w, p in topic[:top_k]])
        probs = probs / probs.sum()
        embeddings = wmodel.encode(words)
        vector = np.average(embeddings, axis=0, weights=probs)
        vectors.append(vector)
    return np.array(vectors)

print("Computing MEAN pooling vectors...")
vectors_mean = compute_vectors_mean(topics_with_probs, word_model)

print("Computing WEIGHTED pooling vectors...")
vectors_weighted = compute_vectors_weighted(topics_with_probs, word_model)


print("\n" + "=" * 60)
print("STEP 5: Clustering")
print("=" * 60)

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
dictionary = Dictionary(corpus)

def run_clustering(vectors, name, corpus, topics_wp):
    best_sil = -1
    best_k = 5
    
    for k in range(4, min(16, len(vectors))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(vectors)
        sil = silhouette_score(vectors, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
    
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(vectors)
    
    intra_sims = []
    for c in range(best_k):
        cv_idx = np.where(labels == c)[0]
        if len(cv_idx) > 1:
            cluster_vecs = vectors[cv_idx]
            sim = cosine_similarity(cluster_vecs)
            n = len(cv_idx)
            pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    pairs.append(sim[i][j])
            if pairs:
                intra_sims.append(np.mean(pairs))
    avg_intra = np.mean(intra_sims) if intra_sims else 0
    
    centroids = km.cluster_centers_
    inter = []
    for i in range(best_k):
        for j in range(i+1, best_k):
            d = 1 - cosine_similarity([centroids[i]], [centroids[j]])[0][0]
            inter.append(d)
    avg_inter = np.mean(inter) if inter else 0
    
    clusters = []
    for c in range(best_k):
        tidx = [i for i, l in enumerate(labels) if l == c]
        all_w = []
        for ti in tidx:
            all_w.extend([w for w, p in topics_wp[ti][:10]])
        wc = Counter(all_w)
        top = [w for w, _ in wc.most_common(10)]
        
        filt = [w for w in top if w in dictionary.token2id]
        try:
            cm = CoherenceModel(topics=[filt], texts=corpus, dictionary=dictionary, coherence='c_v')
            cv = cm.get_coherence()
        except:
            cv = 0.0
        
        clusters.append({
            'cluster_id': c,
            'num_topics': len(tidx),
            'topic_indices': tidx,
            'top_words': top,
            'cv': round(cv, 4),
        })
    
    valid = [c for c in clusters if c['cv'] >= 0.35]
    
    return {
        'method': name,
        'num_clusters': best_k,
        'silhouette': round(best_sil, 4),
        'intra_sim': round(avg_intra, 4),
        'inter_dist': round(avg_inter, 4),
        'avg_cv_all': round(np.mean([c['cv'] for c in clusters]), 4),
        'avg_cv_filtered': round(np.mean([c['cv'] for c in valid]), 4) if valid else 0,
        'num_valid': len(valid),
        'clusters': clusters,
    }

print("MEAN pooling clustering...")
r_mean = run_clustering(vectors_mean, "mean", corpus, topics_with_probs)
print(f"  Clusters: {r_mean['num_clusters']}, Silhouette: {r_mean['silhouette']}")

print("WEIGHTED pooling clustering...")
r_wt = run_clustering(vectors_weighted, "weighted", corpus, topics_with_probs)
print(f"  Clusters: {r_wt['num_clusters']}, Silhouette: {r_wt['silhouette']}")


print("\n" + "=" * 60)
print("COMPARISON: MEAN vs WEIGHTED POOLING")
print("=" * 60)

metrics = [
    ("Clusters", r_mean['num_clusters'], r_wt['num_clusters']),
    ("Silhouette", r_mean['silhouette'], r_wt['silhouette']),
    ("Intra-cluster Similarity", r_mean['intra_sim'], r_wt['intra_sim']),
    ("Inter-cluster Distance", r_mean['inter_dist'], r_wt['inter_dist']),
    ("Avg CV (all)", r_mean['avg_cv_all'], r_wt['avg_cv_all']),
    ("Avg CV (filtered)", r_mean['avg_cv_filtered'], r_wt['avg_cv_filtered']),
    ("Valid Clusters", r_mean['num_valid'], r_wt['num_valid']),
]

print(f"\n  {'Metric':<28} {'Mean':<15} {'Weighted':<15} {'Better?':<10}")
print("  " + "-" * 68)
for name, m, w in metrics:
    if isinstance(m, float):
        better = "Weighted" if w > m else "Mean" if m > w else "Same"
        print(f"  {name:<28} {m:<15.4f} {w:<15.4f} {better}")
    else:
        print(f"  {name:<28} {m:<15} {w:<15}")

# Weighted clusters
print("\n" + "=" * 60)
print("WEIGHTED POOLING CLUSTERS")
print("=" * 60)
for c in sorted(r_wt['clusters'], key=lambda x: x['cv'], reverse=True):
    tag = "KEPT" if c['cv'] >= 0.35 else "REMOVED"
    print(f"\n  Cluster {c['cluster_id']} (CV={c['cv']:.4f}, {c['num_topics']} topics) [{tag}]")
    print(f"    {' | '.join(c['top_words'][:8])}")


with open(OUTPUT_DIR / 'topics_with_probs.json', 'w', encoding='utf-8') as f:
    save = [{'id': i, 'words': [{'w': w, 'p': round(float(p),4)} for w,p in t[:15]]} for i,t in enumerate(topics_with_probs)]
    json.dump(save, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / 'pooling_comparison.json', 'w') as f:
    json.dump({'mean': r_mean, 'weighted': r_wt}, f, indent=2, default=str)

with open(OUTPUT_DIR / 'COMPARISON_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write("MEAN vs WEIGHTED POOLING COMPARISON\n")
    f.write("Sahih Bukhari - CTMNeg (K=20)\n")
    f.write("=" * 60 + "\n\n")
    for name, m, w in metrics:
        if isinstance(m, float):
            f.write(f"  {name:<28} {m:<15.4f} {w:<15.4f}\n")
        else:
            f.write(f"  {name:<28} {m:<15} {w:<15}\n")
    f.write("\nWEIGHTED CLUSTERS:\n")
    for c in sorted(r_wt['clusters'], key=lambda x: x['cv'], reverse=True):
        f.write(f"  Cluster {c['cluster_id']} (CV={c['cv']:.4f}): {' | '.join(c['top_words'][:8])}\n")

print(f"\nSaved to {OUTPUT_DIR}/")
print("DONE!")
