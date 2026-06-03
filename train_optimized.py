"""
Train CTMNeg on Muslim and Abi Dawud
With optimal K selection for each book
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from octis.dataset.dataset import Dataset
from octis.models.CTMN2 import CTMN2
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

books = [
    ('sahih_muslim', 'Sahih Muslim'),
    ('sunan_abi_daud', "Sunan Abi Da'ud"),
    ('sahih_bukhari', 'Sahih Bukhari'),
    ('jami_al-tirmidhi', "Jami' al-Tirmidhi"),
    ('sunan_an-nasai', "Sunan an-Nasa'i"),
    ('sunan_ibn_majah', 'Sunan Ibn Majah'),
]

K_VALUES = [20, 30, 40, 50, 60, 70, 80]
all_results = []

for folder, name in books:
    print('\n' + '=' * 70)
    print(f'PROCESSING: {name}')
    print('=' * 70)
    
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(f'preprocessed_datasets/{folder}')
    corpus = dataset.get_corpus()
    vocab = dataset.get_vocabulary()
    print(f'Docs: {len(corpus)}, Vocab: {len(vocab)}')
    
    # K SELECTION
    print(f'\nK SELECTION: Testing {K_VALUES}')
    print('-' * 50)
    
    best_cv = -1
    best_k = 20
    best_results = None
    k_results = []
    
    for k in K_VALUES:
        print(f'\n  Training K={k}...')
        model = CTMN2(
            num_topics=k,
            num_epochs=50,
            batch_size=32,
            bert_model='aubmindlab/bert-base-arabertv02',
            bert_path=folder,
            inference_type='combined',
            topic_perturb=1,
            tloss_weight=1.0,
        )
        results = model.train_model(dataset)
        
        cv = Coherence(texts=corpus, topk=10, measure='c_v').score(results)
        npmi = Coherence(texts=corpus, topk=10, measure='c_npmi').score(results)
        td = TopicDiversity(topk=10).score(results)
        
        print(f'  K={k}: CV={cv:.4f}  NPMI={npmi:.4f}  TD={td:.4f}')
        
        k_results.append({'k': k, 'cv': cv, 'npmi': npmi, 'td': td})
        
        if cv > best_cv:
            best_cv = cv
            best_k = k
            best_results = results
    
    print(f'\n  OPTIMAL K = {best_k} (CV = {best_cv:.4f})')
    
    # Get best metrics
    best_metrics = next(r for r in k_results if r['k'] == best_k)
    
    # SAVE RESULTS
    rd = Path(f'results_{folder}')
    rd.mkdir(exist_ok=True)
    
    with open(rd / 'SUMMARY.txt', 'w', encoding='utf-8') as f:
        f.write(f'RESULTS: {name}\n{"=" * 50}\n\n')
        f.write(f'Documents: {len(corpus)}\n')
        f.write(f'Vocabulary: {len(vocab)}\n\n')
        
        f.write('K SELECTION:\n')
        f.write('-' * 40 + '\n')
        for r in k_results:
            marker = ' ← OPTIMAL' if r['k'] == best_k else ''
            f.write(f"  K={r['k']:3d}  CV={r['cv']:.4f}  NPMI={r['npmi']:.4f}  TD={r['td']:.4f}{marker}\n")
        
        f.write(f'\nOptimal K = {best_k}\n')
        f.write(f'CV:   {best_metrics["cv"]:.4f}\n')
        f.write(f'NPMI: {best_metrics["npmi"]:.4f}\n')
        f.write(f'TD:   {best_metrics["td"]:.4f}\n\n')
        
        f.write(f'TOPICS (K={best_k}):\n')
        f.write('-' * 40 + '\n')
        for i, t in enumerate(best_results['topics']):
            f.write(f'  Topic {i:2d}: {" | ".join(t[:8])}\n')
    
    with open(rd / 'results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'book': name, 'docs': len(corpus), 'vocab': len(vocab),
            'optimal_k': best_k,
            'cv': best_metrics['cv'], 'npmi': best_metrics['npmi'], 'td': best_metrics['td'],
            'k_selection': k_results,
            'topics': best_results['topics'],
        }, f, ensure_ascii=False, indent=2)
    
    all_results.append({
        'book': name, 'docs': len(corpus),
        'optimal_k': best_k,
        'cv': best_metrics['cv'], 'npmi': best_metrics['npmi'], 'td': best_metrics['td'],
    })
    print(f'Saved to {rd}/')

# COMBINED SUMMARY
print('\n' + '=' * 70)
print('ALL BOOKS SUMMARY')
print('=' * 70)
print(f"\n  {'Book':<25} {'Docs':<8} {'K':<5} {'CV':<10} {'NPMI':<10} {'TD':<10}")
print('  ' + '-' * 68)
for r in all_results:
    print(f"  {r['book']:<25} {r['docs']:<8} {r['optimal_k']:<5} {r['cv']:<10.4f} {r['npmi']:<10.4f} {r['td']:<10.4f}")

with open('ALL_BOOKS_RESULTS.txt', 'w', encoding='utf-8') as f:
    f.write('COMPLETE RESULTS - ALL BOOKS\n')
    f.write('=' * 70 + '\n\n')
    f.write(f"  {'Book':<25} {'Docs':<8} {'K':<5} {'CV':<10} {'NPMI':<10} {'TD':<10}\n")
    f.write('  ' + '-' * 68 + '\n')
    for r in all_results:
        f.write(f"  {r['book']:<25} {r['docs']:<8} {r['optimal_k']:<5} {r['cv']:<10.4f} {r['npmi']:<10.4f} {r['td']:<10.4f}\n")

print('\nDONE!')
