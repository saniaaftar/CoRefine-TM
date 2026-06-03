"""
Run complete pipeline on all 3 books
- Convert data
- Train CTMNeg
- Compute metrics
- Save results
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

BOOKS = {
    'sahih_muslim': 'Sahih Muslim',
    'sunan_abi_daud': "Sunan Abi Da'ud",
}

for folder, book_name in BOOKS.items():
    print("\n" + "=" * 70)
    print(f"PROCESSING: {book_name}")
    print("=" * 70)
    
    # Step 1: Convert data to OCTIS format
    tokens_path = f"../data/processed/{folder}/tokens.jsonl"
    out_path = Path(f"preprocessed_datasets/{folder}")
    out_path.mkdir(parents=True, exist_ok=True)
    
    docs = []
    with open(tokens_path) as f:
        for line in f:
            rec = json.loads(line)
            if len(rec["tokens"]) > 0:
                docs.append(rec["tokens"])
    
    print(f"Docs: {len(docs)}")
    
    word_freq = Counter()
    for d in docs:
        word_freq.update(d)
    vocab = [w for w, c in word_freq.most_common() if c >= 5]
    vocab_set = set(vocab)
    print(f"Vocab: {len(vocab)}")
    
    total = len(docs)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    
    with open(out_path / "corpus.tsv", "w", encoding="utf-8") as f:
        for i, tokens in enumerate(docs):
            filtered = [t for t in tokens if t in vocab_set]
            if len(filtered) == 0:
                filtered = ["_empty_"]
            if i < train_end:
                part = "train"
            elif i < val_end:
                part = "val"
            else:
                part = "test"
            f.write(" ".join(filtered) + "\t" + part + "\n")
    
    with open(out_path / "vocabulary.txt", "w", encoding="utf-8") as f:
        for w in vocab:
            f.write(w + "\n")
    
    metadata = {
        "total_documents": total,
        "vocabulary_length": len(vocab),
        "preprocessing-info": [],
        "labels": [],
        "total_labels": 0,
        "last-training-doc": train_end,
        "last-validation-doc": val_end
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    print(f"Data converted: {out_path}")
    
    # Step 2: Load dataset
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(str(out_path))
    corpus = dataset.get_corpus()
    
    # Step 3: Train with K=40
    print(f"\nTraining CTMNeg (K=40)...")
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
    
    # Step 4: Compute metrics
    cv = Coherence(texts=corpus, topk=10, measure='c_v')
    npmi = Coherence(texts=corpus, topk=10, measure='c_npmi')
    td = TopicDiversity(topk=10)
    
    cv_score = cv.score(results)
    npmi_score = npmi.score(results)
    td_score = td.score(results)
    
    print(f"\nMetrics:")
    print(f"  CV:   {cv_score:.4f}")
    print(f"  NPMI: {npmi_score:.4f}")
    print(f"  TD:   {td_score:.4f}")
    
    # Step 5: Save results
    result_dir = Path(f"results_{folder}")
    result_dir.mkdir(exist_ok=True)
    
    output = {
        'book': book_name,
        'docs': len(docs),
        'vocab': len(vocab),
        'num_topics': 40,
        'cv': cv_score,
        'npmi': npmi_score,
        'td': td_score,
        'topics': results['topics'],
    }
    
    with open(result_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Save readable summary
    with open(result_dir / "SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write(f"RESULTS: {book_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Documents: {len(docs)}\n")
        f.write(f"Vocabulary: {len(vocab)}\n")
        f.write(f"Topics: 40\n")
        f.write(f"CV:   {cv_score:.4f}\n")
        f.write(f"NPMI: {npmi_score:.4f}\n")
        f.write(f"TD:   {td_score:.4f}\n\n")
        f.write("Topics:\n")
        for i, t in enumerate(results['topics']):
            f.write(f"  Topic {i:2d}: {' | '.join(t[:8])}\n")
    
    print(f"Saved to {result_dir}/")

# Final combined summary
print("\n" + "=" * 70)
print("ALL BOOKS SUMMARY")
print("=" * 70)


all_results = []

buk_path = Path("results_bukhari/summary.json")
if buk_path.exists():
    with open(buk_path) as f:
        buk = json.load(f)
    all_results.append({
        'book': 'Sahih Bukhari',
        'docs': buk.get('total_docs', 7295),
        'cv': buk['metrics_raw']['cv'],
        'npmi': buk['metrics_raw']['npmi'],
        'td': buk['metrics_raw']['td'],
    })

for folder in BOOKS:
    rpath = Path(f"results_{folder}/results.json")
    if rpath.exists():
        with open(rpath) as f:
            r = json.load(f)
        all_results.append({
            'book': r['book'],
            'docs': r['docs'],
            'cv': r['cv'],
            'npmi': r['npmi'],
            'td': r['td'],
        })

print(f"\n  {'Book':<25} {'Docs':<10} {'CV':<10} {'NPMI':<10} {'TD':<10}")
print("  " + "-" * 65)
for r in all_results:
    print(f"  {r['book']:<25} {r['docs']:<10} {r['cv']:<10.4f} {r['npmi']:<10.4f} {r['td']:<10.4f}")

# Save combined
with open("ALL_BOOKS_RESULTS.txt", "w", encoding="utf-8") as f:
    f.write("COMPLETE RESULTS - ALL BOOKS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"  {'Book':<25} {'Docs':<10} {'CV':<10} {'NPMI':<10} {'TD':<10}\n")
    f.write("  " + "-" * 65 + "\n")
    for r in all_results:
        f.write(f"  {r['book']:<25} {r['docs']:<10} {r['cv']:<10.4f} {r['npmi']:<10.4f} {r['td']:<10.4f}\n")

print("\nALL DONE!")
