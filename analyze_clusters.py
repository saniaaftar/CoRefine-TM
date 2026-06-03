import json
import numpy as np

with open('results_bukhari/gsm_results.json') as f:
    gsm = json.load(f)

clusters = gsm['clusters']

print("ALL CLUSTERS:")
print("Cluster    Topics    CV")
print("-" * 30)
for c in clusters:
    print(f"  {c['cluster_id']:<8} {c['num_topics']:<8} {c['cv']:.4f}")

print()
print("THRESHOLD ANALYSIS:")
print("Threshold   Surviving   Avg CV     Coverage   Score")
print("-" * 55)

best_score = 0
best_tau = 0.40

for tau in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    surviving = [c for c in clusters if c['cv'] >= tau]
    if len(surviving) == 0:
        continue
    avg_cv = np.mean([c['cv'] for c in surviving])
    coverage = len(surviving) / len(clusters)
    score = avg_cv * coverage
    marker = ""
    if score > best_score:
        best_score = score
        best_tau = tau
        marker = " <-- BEST"
    print(f"  {tau:.2f}        {len(surviving):<10} {avg_cv:.4f}     {coverage:.2f}       {score:.4f}{marker}")

print()
print(f"BEST THRESHOLD: {best_tau}")
print()

# filtered clusters
surviving = [c for c in clusters if c['cv'] >= best_tau]
print(f"FILTERED CLUSTERS (CV >= {best_tau}):")
print("=" * 50)
for c in surviving:
    words = " | ".join(c['top_words'][:6])
    print(f"  Cluster {c['cluster_id']} (CV={c['cv']:.4f}, {c['num_topics']} topics):")
    print(f"    {words}")
    print()

removed = [c for c in clusters if c['cv'] < best_tau]
print(f"REMOVED CLUSTERS (CV < {best_tau}):")
for c in removed:
    words = " | ".join(c['top_words'][:6])
    print(f"  Cluster {c['cluster_id']} (CV={c['cv']:.4f}): {words}")

print(f"\nSurvived: {len(surviving)}/{len(clusters)} clusters")
print(f"Avg CV after filtering: {np.mean([c['cv'] for c in surviving]):.4f}")
