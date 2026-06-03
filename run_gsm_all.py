"""
GSM Pipeline - All Books
"""
import sys, json
import numpy as np
from pathlib import Path
from collections import Counter
sys.path.insert(0, '.')

from octis.dataset.dataset import Dataset
from octis.models.CTMN2 import CTMN2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

# =====================
# BOOKS CONFIG
# =====================
BOOKS = [
    {'name': 'Sahih Muslim',    'folder': 'sahih_muslim',    'k': 20},
    {'name': 'Sunan Abi Daud',  'folder': 'sunan_abi_daud',  'k': 50},
    {'name': 'Sahih Bukhari',   'folder': 'sahih_bukhari',  'k': 30},
    {'name': "Jami' al-Tirmidhi", 'folder': 'jami_al-tirmidhi', 'k': 20},
    {'name': "Sunan an-Nasa'i",   'folder': 'sunan_an-nasai',   'k': 60},
    {'name': 'Sunan Ibn Majah',  'folder': 'sunan_ibn_majah',  'k': 20},
]

word_model = SentenceTransformer('CAMeL-Lab/bert-base-arabic-camelbert-da')

def compute_vectors_mean(topics_wp, wmodel, top_k=10):
    vectors = []
    for topic in topics_wp:
        words = [w for w, p in topic[:top_k]]
        embeddings = wmodel.encode(words)
        vectors.append(np.mean(embeddings, axis=0))
    return np.array(vectors)

def compute_vectors_weighted(topics_wp, wmodel, top_k=10):
    vectors = []
    for topic in topics_wp:
        words = [w for w, p in topic[:top_k]]
        probs = np.array([p for w, p in topic[:top_k]])
        probs = probs / probs.sum()
        embeddings = wmodel.encode(words)
        vectors.append(np.average(embeddings, axis=0, weights=probs))
    return np.array(vectors)

def run_clustering(vectors, name, corpus, topics_wp):
    best_sil, best_k = -1, 5
    for k in range(4, min(16, len(vectors))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(vectors)
        sil = silhouette_score(vectors, labels)
        if sil > best_sil:
            best_sil, best_k = sil, k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(vectors)
    dictionary = Dictionary(corpus)

    clusters = []
    for c in range(best_k):
        tidx = [i for i, l in enumerate(labels) if l == c]
        all_w = []
        for ti in tidx:
            all_w.extend([w for w, p in topics_wp[ti][:10]])
        top = [w for w, _ in Counter(all_w).most_common(10)]
        filt = [w for w in top if w in dictionary.token2id]
        try:
            cv = CoherenceModel(topics=[filt], texts=corpus, dictionary=dictionary, coherence='c_v').get_coherence()
        except:
            cv = 0.0
        clusters.append({'cluster_id': c, 'num_topics': len(tidx), 'top_words': top, 'cv': round(cv, 4)})

    valid = [c for c in clusters if c['cv'] >= 0.35]
    return {
        'method': name,
        'num_clusters': best_k,
        'silhouette': round(best_sil, 4),
        'avg_cv_all': round(np.mean([c['cv'] for c in clusters]), 4),
        'avg_cv_filtered': round(np.mean([c['cv'] for c in valid]), 4) if valid else 0,
        'num_valid': len(valid),
        'clusters': clusters,
    }

# =====================
# MAIN LOOP
# =====================
all_summary = []

for book in BOOKS:
    print("\n" + "=" * 60)
    print(f"PROCESSING: {book['name']} (K={book['k']})")
    print("=" * 60)

    OUTPUT_DIR = Path(f"results_{book['folder']}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(f"preprocessed_datasets/{book['folder']}")
    corpus = dataset.get_corpus()
    print(f"Docs: {len(corpus)}, Vocab: {len(dataset.get_vocabulary())}")

    model = CTMN2(
        num_topics=book['k'],
        num_epochs=50,
        bert_model='aubmindlab/bert-base-arabertv02',
        bert_path=book['folder'],
        inference_type='combined',
        topic_perturb=1,
        tloss_weight=1.0,
    )
    results = model.train_model(dataset)
    print("Training done!")

    inner = model.model
    beta = inner.get_topic_word_mat()
    train_vocab = [inner.train_data.idx2token[i] for i in range(len(inner.train_data.idx2token))]

    topics_with_probs = []
    for topic_idx in range(beta.shape[0]):
        topic_probs = beta[topic_idx]
        sorted_indices = np.argsort(topic_probs)[::-1]
        topics_with_probs.append([(train_vocab[i], float(topic_probs[i])) for i in sorted_indices[:15]])

    print("Computing vectors...")
    vectors_mean = compute_vectors_mean(topics_with_probs, word_model)
    vectors_weighted = compute_vectors_weighted(topics_with_probs, word_model)

    print("Clustering...")
    r_mean = run_clustering(vectors_mean, "mean", corpus, topics_with_probs)
    r_wt   = run_clustering(vectors_weighted, "weighted", corpus, topics_with_probs)

    print(f"\n  {'Metric':<28} {'Mean':<12} {'Weighted':<12} {'Better'}")
    print("  " + "-" * 60)
    for metric in ['num_clusters','silhouette','avg_cv_all','avg_cv_filtered','num_valid']:
        m, w = r_mean[metric], r_wt[metric]
        better = "Weighted" if w > m else "Mean" if m > w else "Same"
        if isinstance(m, float):
            print(f"  {metric:<28} {m:<12.4f} {w:<12.4f} {better}")
        else:
            print(f"  {metric:<28} {m:<12} {w:<12} {better}")

    with open(OUTPUT_DIR / 'gsm_comparison.json', 'w') as f:
        json.dump({'mean': r_mean, 'weighted': r_wt}, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/")
    all_summary.append({'book': book['name'], 'k': book['k'],
                        'mean_cv': r_mean['avg_cv_filtered'],
                        'weighted_cv': r_wt['avg_cv_filtered'],
                        'mean_clusters': r_mean['num_clusters'],
                        'weighted_clusters': r_wt['num_clusters']})

print("\n" + "=" * 60)
print("ALL BOOKS GSM SUMMARY")
print("=" * 60)
print(f"  {'Book':<25} {'K':<6} {'Mean CV':<12} {'Weighted CV':<12}")
print("  " + "-" * 55)
for s in all_summary:
    print(f"  {s['book']:<25} {s['k']:<6} {s['mean_cv']:<12.4f} {s['weighted_cv']:<12.4f}")
print("\nDONE!")
