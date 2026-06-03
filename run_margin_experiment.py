import sys, json
import numpy as np
sys.path.insert(0, '.')

from octis.dataset.dataset import Dataset
from octis.models.CTMN2 import CTMN2
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

dataset = Dataset()
dataset.load_custom_dataset_from_folder('preprocessed_datasets/sahih_bukhari')
corpus = dataset.get_corpus()

margins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
results = []

for margin in margins:
    print(f'\n--- Margin = {margin} ---')
    model = CTMN2(
        num_topics=30,
        num_epochs=50,
        batch_size=32,
        bert_model='aubmindlab/bert-base-arabertv02',
        bert_path='sahih_bukhari',
        inference_type='combined',
        topic_perturb=1,
        tloss_weight=1.0,
    )
    model.margin = margin
    r = model.train_model(dataset)
    cv   = Coherence(texts=corpus, topk=10, measure='c_v').score(r)
    npmi = Coherence(texts=corpus, topk=10, measure='c_npmi').score(r)
    td   = TopicDiversity(topk=10).score(r)
    results.append({'margin': margin, 'cv': round(cv,4),
                    'npmi': round(npmi,4), 'td': round(td,4)})
    print(f'CV={cv:.4f} NPMI={npmi:.4f} TD={td:.4f}')

print('\n=== RESULTS ===')
for r in results:
    print(f"Margin={r['margin']}: CV={r['cv']} NPMI={r['npmi']} TD={r['td']}")

with open('margin_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved!')
