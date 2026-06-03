"""
M Component: LLM-guided Taxonomy Labeling
- Baseline: Cosine similarity labeling
- Proposed: LLM  labeling with explanations
"""

import sys
import json
import numpy as np
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, '.')

import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = Path('results_bukhari')

# ============================================
# CONFIG
# ============================================
ANTHROPIC_API_KEY = "PUT  K-E-Y HERE"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_HEADERS = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}

# ============================================
# LOAD DATA
# ============================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

# Load weighted clusters
with open(OUTPUT_DIR / 'pooling_comparison.json') as f:
    comparison = json.load(f)

weighted_clusters = comparison['weighted']['clusters']
valid_clusters = [c for c in weighted_clusters if c['cv'] >= 0.35]
print(f"Valid clusters: {len(valid_clusters)}")

# Load taxonomy
with open('../data/taxonomy/taxonomy_converted.json', 'r', encoding='utf-8') as f:
    taxonomy = json.load(f)

print(f"Taxonomy categories: {len(taxonomy['categories'])}")
for cat in taxonomy['categories']:
    print(f"  {cat['name_en']} ({cat['name_ar']}): {len(cat['subcategories'])} subcategories")

# ============================================
# METHOD 1: COSINE SIMILARITY LABELING (Baseline)
# ============================================
print("\n" + "=" * 60)
print("METHOD 1: COSINE SIMILARITY LABELING (Baseline)")
print("=" * 60)

word_model = SentenceTransformer('CAMeL-Lab/bert-base-arabic-camelbert-da')

# Build taxonomy embeddings
tax_embeddings = {}
for cat in taxonomy['categories']:
    for sub in cat['subcategories']:
        key = f"{cat['name_en']} > {sub['name_en']}"
        descriptors = sub['descriptors']
        if descriptors:
            emb = word_model.encode(descriptors)
            tax_embeddings[key] = {
                'embedding': np.mean(emb, axis=0),
                'name_ar': f"{cat['name_ar']} > {sub['name_ar']}",
                'descriptors': descriptors,
            }

print(f"Taxonomy embeddings: {len(tax_embeddings)} subcategories")

# Label each cluster with cosine
cosine_results = []
for cluster in valid_clusters:
    words = cluster['top_words'][:10]
    cluster_emb = np.mean(word_model.encode(words), axis=0)
    
    best_label = "Unlabeled"
    best_sim = 0
    best_ar = ""
    
    for label, tax_info in tax_embeddings.items():
        sim = cosine_similarity([cluster_emb], [tax_info['embedding']])[0][0]
        if sim > best_sim:
            best_sim = sim
            best_label = label
            best_ar = tax_info['name_ar']
    
    cosine_results.append({
        'cluster_id': cluster['cluster_id'],
        'cv': cluster['cv'],
        'top_words': words,
        'label': best_label,
        'label_ar': best_ar,
        'similarity': round(float(best_sim), 4),
    })
    
    print(f"\n  Cluster {cluster['cluster_id']} (CV={cluster['cv']:.4f}):")
    print(f"    Words: {' | '.join(words[:6])}")
    print(f"    Label: {best_label}")
    print(f"    Arabic: {best_ar}")
    print(f"    Similarity: {best_sim:.4f}")

# ============================================
# METHOD 2: LLM LABELING (Proposed - )
# ============================================
print("\n" + "=" * 60)
print("METHOD 2: LLM LABELING ")
print("=" * 60)

# Build taxonomy string for prompt
tax_string = ""
for cat in taxonomy['categories']:
    tax_string += f"\n{cat['id']}. {cat['name_ar']} ({cat['name_en']})\n"
    for sub in cat['subcategories']:
        descs = "، ".join(sub['descriptors'][:5])
        tax_string += f"   {sub['id']} {sub['name_ar']} ({sub['name_en']}) - {descs}\n"

def call_gemini(prompt, max_retries=3):
    """Call Gemini API with retry."""
    for attempt in range(max_retries):
        try:
            payload = {
                "model": "claude-sonnet-4-5",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            }
            response = requests.post(ANTHROPIC_URL, headers=ANTHROPIC_HEADERS, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                text = data['content'][0]['text']
                return text
            elif response.status_code == 429:
                print(f"    Rate limited, waiting 60s...")
                time.sleep(60)
            else:
                print(f"    API error {response.status_code}: {response.text[:200]}")
                time.sleep(5)
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(5)
    return None

llm_results = []
NUM_RUNS = 1  # 3 runs per cluster for majority voting

for cluster in valid_clusters:
    words = cluster['top_words'][:10]
    words_str = "، ".join(words)
    
    print(f"\n  Cluster {cluster['cluster_id']} (CV={cluster['cv']:.4f}):")
    print(f"    Words: {' | '.join(words[:6])}")
    
    votes = []
    explanations = []
    confidences = []
    
    for run in range(NUM_RUNS):
        prompt = f"""You are an expert Islamic scholar specializing in Hadith classification and Classical Arabic terminology.

TASK: Classify the following topic cluster into the most appropriate Hadith taxonomy category.

TOPIC CLUSTER WORDS:
{words_str}

HADITH TAXONOMY:
{tax_string}

Respond ONLY in this exact JSON format (no markdown, no backticks):
{{"main_category": "...", "main_category_ar": "...", "subcategory": "...", "subcategory_ar": "...", "confidence": 0.0, "key_evidence_words": ["...", "..."], "explanation": "..."}}"""

        response = call_gemini(prompt)
        
        if response:
            # Parse JSON
            try:
                # Clean response
                response = response.strip()
                if response.startswith("```"):
                    response = response.split("\n", 1)[1]
                if response.endswith("```"):
                    response = response.rsplit("```", 1)[0]
                response = response.strip()
                
                result = json.loads(response)
                label = f"{result.get('main_category', '?')} > {result.get('subcategory', '?')}"
                votes.append(label)
                confidences.append(result.get('confidence', 0))
                explanations.append(result.get('explanation', ''))
                print(f"    Run {run+1}: {label} (conf={result.get('confidence', 0)})")
            except json.JSONDecodeError:
                print(f"    Run {run+1}: JSON parse error")
                print(f"    Response: {response[:200]}")
        
        time.sleep(5)  # Rate limiting
    
    # Majority voting
    if votes:
        vote_counts = Counter(votes)
        majority_label = vote_counts.most_common(1)[0][0]
        consistency = vote_counts.most_common(1)[0][1] / len(votes)
        
        # Get best confidence and explanation for majority label
        best_conf = 0
        best_expl = ""
        for v, c, e in zip(votes, confidences, explanations):
            if v == majority_label and c > best_conf:
                best_conf = c
                best_expl = e
    else:
        majority_label = "Unlabeled"
        consistency = 0
        best_conf = 0
        best_expl = "LLM failed to classify"
    
    llm_result = {
        'cluster_id': cluster['cluster_id'],
        'cv': cluster['cv'],
        'top_words': words,
        'label': majority_label,
        'confidence': best_conf,
        'consistency': consistency,
        'explanation': best_expl,
        'all_votes': votes,
    }
    llm_results.append(llm_result)
    
    print(f"    FINAL: {majority_label} (conf={best_conf}, consistency={consistency:.0%})")

# ============================================
# COMPARISON: Cosine vs LLM
# ============================================
print("\n" + "=" * 60)
print("COMPARISON: COSINE vs LLM LABELING")
print("=" * 60)

print(f"\n  {'Cluster':<10} {'CV':<8} {'Cosine Label':<40} {'LLM Label':<40}")
print("  " + "-" * 98)

match_count = 0
for cos, llm in zip(cosine_results, llm_results):
    match = "SAME" if cos['label'] == llm['label'] else "DIFF"
    if cos['label'] == llm['label']:
        match_count += 1
    print(f"  {cos['cluster_id']:<10} {cos['cv']:<8.4f} {cos['label']:<40} {llm['label']:<40} [{match}]")

agreement = match_count / len(cosine_results) * 100 if cosine_results else 0
print(f"\n  Agreement: {match_count}/{len(cosine_results)} ({agreement:.0f}%)")

# ============================================
# LLM DETAILED RESULTS
# ============================================
print("\n" + "=" * 60)
print("LLM LABELING DETAILS")
print("=" * 60)

for llm in llm_results:
    print(f"\n  Cluster {llm['cluster_id']} (CV={llm['cv']:.4f}):")
    print(f"    Words: {' | '.join(llm['top_words'][:6])}")
    print(f"    Label: {llm['label']}")
    print(f"    Confidence: {llm['confidence']}")
    print(f"    Consistency: {llm['consistency']:.0%}")
    print(f"    Explanation: {llm['explanation'][:150]}")

# ============================================
# SAVE RESULTS
# ============================================
print("\n" + "=" * 60)
print("SAVING")
print("=" * 60)

with open(OUTPUT_DIR / 'cosine_labeling.json', 'w', encoding='utf-8') as f:
    json.dump(cosine_results, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / 'llm_labeling.json', 'w', encoding='utf-8') as f:
    json.dump(llm_results, f, ensure_ascii=False, indent=2)

# Save readable report
with open(OUTPUT_DIR / 'LABELING_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write("M COMPONENT: COSINE vs LLM LABELING\n")
    f.write("Sahih Bukhari - CTMNeg (K=40)\n")
    f.write("=" * 70 + "\n\n")
    
    f.write(f"{'Cluster':<10} {'CV':<8} {'Cosine Label':<40} {'LLM Label':<40}\n")
    f.write("-" * 98 + "\n")
    for cos, llm in zip(cosine_results, llm_results):
        f.write(f"{cos['cluster_id']:<10} {cos['cv']:<8.4f} {cos['label']:<40} {llm['label']:<40}\n")
    
    f.write(f"\nAgreement: {agreement:.0f}%\n")
    
    f.write(f"\nCOSINE DETAILS:\n")
    f.write("-" * 40 + "\n")
    for cos in cosine_results:
        f.write(f"  Cluster {cos['cluster_id']}: {cos['label']} (sim={cos['similarity']})\n")
        f.write(f"    Words: {' | '.join(cos['top_words'][:6])}\n\n")
    
    f.write(f"\nLLM DETAILS:\n")
    f.write("-" * 40 + "\n")
    for llm in llm_results:
        f.write(f"  Cluster {llm['cluster_id']}: {llm['label']}\n")
        f.write(f"    Confidence: {llm['confidence']}, Consistency: {llm['consistency']:.0%}\n")
        f.write(f"    Words: {' | '.join(llm['top_words'][:6])}\n")
        f.write(f"    Explanation: {llm['explanation'][:200]}\n\n")

print("Saved:")
print("  - cosine_labeling.json")
print("  - llm_labeling.json")
print("  - LABELING_REPORT.txt")
print("\nDONE!")
