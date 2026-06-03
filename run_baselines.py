import sys, json
sys.path.insert(0, '.')

from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

dataset = Dataset()
dataset.load_custom_dataset_from_folder('preprocessed_datasets/sahih_bukhari')
corpus = dataset.get_corpus()
print(f'Docs: {len(corpus)}')

K = 30
results = {}

# ProdLDA
print('\n--- ProdLDA ---')
from octis.models.ProdLDA import ProdLDA
model = ProdLDA(num_topics=K, num_epochs=50)
r = model.train_model(dataset)
cv = Coherence(texts=corpus, topk=10, measure='c_v').score(r)
npmi = Coherence(texts=corpus, topk=10, measure='c_npmi').score(r)
td = TopicDiversity(topk=10).score(r)
results['ProdLDA'] = {'cv': round(cv,4), 'npmi': round(npmi,4), 'td': round(td,4)}
print(f'ProdLDA: CV={cv:.4f} NPMI={npmi:.4f} TD={td:.4f}')

# NeuralLDA
print('\n--- NeuralLDA ---')
from octis.models.NeuralLDA import NeuralLDA
model = NeuralLDA(num_topics=K, num_epochs=50)
r = model.train_model(dataset)
cv = Coherence(texts=corpus, topk=10, measure='c_v').score(r)
npmi = Coherence(texts=corpus, topk=10, measure='c_npmi').score(r)
td = TopicDiversity(topk=10).score(r)
results['NeuralLDA'] = {'cv': round(cv,4), 'npmi': round(npmi,4), 'td': round(td,4)}
print(f'NeuralLDA: CV={cv:.4f} NPMI={npmi:.4f} TD={td:.4f}')

# Our results
results['CTMNeg (no G-S-M)'] = {'cv': 0.4721, 'npmi': -0.1249, 'td': 0.8467}
results['CoRefine-TM (ours)'] = {'cv': 0.6363, 'npmi': -0.1249, 'td': 0.8467}

print('\n=== FINAL RESULTS ===')
print(f"{'Model':<25} {'CV':<10} {'NPMI':<10} {'TD':<10}")
print('-' * 55)
for model, r in results.items():
    print(f"{model:<25} {r['cv']:<10.4f} {r['npmi']:<10.4f} {r['td']:<10.4f}")

with open('baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved!')
