"""
M Component: LLM Labeling - All Books
- 3 runs majority voting
- Complex chain-of-thought prompt
- Cosine vs LLM comparison
"""
import sys, json, time, requests
import numpy as np
from pathlib import Path
from collections import Counter
sys.path.insert(0, '.')
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================
# CONFIG
# =====================
BOOKS = [
    {'name': 'Sahih Muslim',   'folder': 'sahih_muslim',   'k': 20},
    {'name': 'Sunan Abi Daud', 'folder': 'sunan_abi_daud', 'k': 50},
    {'name': 'Sahih Bukhari',  'folder': 'sahih_bukhari',  'k': 30},
    {'name': "Jami' al-Tirmidhi", 'folder': 'jami_al-tirmidhi', 'k': 20},
    {'name': "Sunan an-Nasa'i",   'folder': 'sunan_an-nasai',   'k': 60},
    {'name': 'Sunan Ibn Majah',  'folder': 'sunan_ibn_majah',  'k': 20},
    {'name': "Jami' al-Tirmidhi", 'folder': 'jami_al-tirmidhi', 'k': 20},
    {'name': "Sunan an-Nasa'i",   'folder': 'sunan_an-nasai',   'k': 60},
    {'name': 'Sunan Ibn Majah',  'folder': 'sunan_ibn_majah',  'k': 20},
]

ANTHROPIC_API_KEY = "API KEY"
ANTHROPIC_URL = "https://api.anthr......"
HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}
NUM_RUNS = 3
CV_THRESHOLD = 0.35

# =====================
# LOAD TAXONOMY
# =====================
with open('../data/taxonomy/taxonomy_converted.json', 'r', encoding='utf-8') as f:
    taxonomy = json.load(f)

# Build taxonomy string
tax_string = ""
for cat in taxonomy['categories']:
    tax_string += f"\n{cat['id']}. {cat['name_ar']} ({cat['name_en']})\n"
    for sub in cat['subcategories']:
        descs = "، ".join(sub['descriptors'][:6])
        tax_string += f"   {sub['id']} {sub['name_ar']} ({sub['name_en']})\n"
        tax_string += f"        Key terms: {descs}\n"

# Load word model
word_model = SentenceTransformer('CAMeL-Lab/bert-base-arabic-camelbert-da')

# Build taxonomy embeddings for cosine
tax_embeddings = {}
for cat in taxonomy['categories']:
    for sub in cat['subcategories']:
        key = f"{cat['name_en']} > {sub['name_en']}"
        if sub['descriptors']:
            emb = word_model.encode(sub['descriptors'])
            tax_embeddings[key] = {
                'embedding': np.mean(emb, axis=0),
                'name_ar': f"{cat['name_ar']} > {sub['name_ar']}",
            }

def call_llm(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            payload = {
                "model": "claude-sonnet-4-5",
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}]
            }
            r = requests.post(ANTHROPIC_URL, headers=HEADERS, json=payload, timeout=45)
            if r.status_code == 200:
                return r.json()['content'][0]['text']
            elif r.status_code == 429:
                print(f"    Rate limited, waiting 60s...")
                time.sleep(60)
            else:
                print(f"    API error {r.status_code}")
                time.sleep(5)
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(5)
    return None

def complex_prompt(words_str, tax_string, run_num):
    """Complex chain-of-thought prompt with different angles per run."""
    
    angles = [
        "Focus on the PRIMARY religious practice or ritual these words indicate.",
        "Focus on the SOCIAL or LEGAL context these words suggest in Islamic jurisprudence.",
        "Focus on the THEMATIC and CONCEPTUAL category that best unifies all these words.",
    ]
    angle = angles[run_num % 3]
    
    return f"""You are a senior Islamic scholar and computational linguist specializing in Hadith sciences and Classical Arabic NLP.

TASK: Perform expert taxonomy classification of a topic cluster extracted from Hadith corpus analysis.

TOPIC CLUSTER WORDS (Arabic):
{words_str}

CLASSIFICATION INSTRUCTIONS:
1. STEP 1 - Word Analysis: Examine each Arabic word carefully for its root meaning and Islamic context.
2. STEP 2 - Pattern Recognition: Identify the dominant thematic pattern across ALL words.
3. STEP 3 - {angle}
4. STEP 4 - Taxonomy Matching: Match to the MOST SPECIFIC subcategory possible.
5. STEP 5 - Confidence Assessment: Rate your confidence based on evidence strength.

HADITH TAXONOMY (use EXACT names from this list):
{tax_string}

CLASSIFICATION CRITERIA:
- Choose the subcategory where AT LEAST 60% of the words relate to it
- If words span multiple categories, choose the DOMINANT one
- High confidence (>0.8): 7+ words clearly match
- Medium confidence (0.5-0.8): 4-6 words match
- Low confidence (<0.5): fewer than 4 words match

Respond ONLY in this exact JSON format (no markdown, no backticks, no extra text):
{{"step1_word_analysis": "brief analysis of key words", "step2_pattern": "dominant pattern identified", "step3_focus": "specific focus finding", "main_category": "exact English name from taxonomy", "main_category_ar": "exact Arabic name", "subcategory": "exact English subcategory name", "subcategory_ar": "exact Arabic subcategory name", "confidence": 0.0, "key_evidence_words": ["word1", "word2", "word3"], "explanation": "2-3 sentence justification"}}"""

def cosine_label(cluster):
    words = cluster['top_words'][:10]
    emb = np.mean(word_model.encode(words), axis=0)
    best_label, best_sim, best_ar = "Unlabeled", 0, ""
    for label, info in tax_embeddings.items():
        sim = cosine_similarity([emb], [info['embedding']])[0][0]
        if sim > best_sim:
            best_sim, best_label, best_ar = sim, label, info['name_ar']
    return best_label, best_ar, round(float(best_sim), 4)

def llm_label(cluster):
    words = cluster['top_words'][:10]
    words_str = "، ".join(words)
    votes, confidences, explanations, analyses = [], [], [], []

    for run in range(NUM_RUNS):
        prompt = complex_prompt(words_str, tax_string, run)
        response = call_llm(prompt)

        if response:
            try:
                resp = response.strip()
                if resp.startswith("```"):
                    resp = resp.split("\n", 1)[1]
                if resp.endswith("```"):
                    resp = resp.rsplit("```", 1)[0]
                result = json.loads(resp.strip())
                label = f"{result.get('main_category','?')} > {result.get('subcategory','?')}"
                votes.append(label)
                confidences.append(result.get('confidence', 0))
                explanations.append(result.get('explanation', ''))
                analyses.append(result.get('step2_pattern', ''))
                print(f"      Run {run+1}: {label} (conf={result.get('confidence',0):.2f})")
            except json.JSONDecodeError as e:
                print(f"      Run {run+1}: JSON error - {str(e)[:50]}")
                print(f"      Response preview: {response[:150]}")
        time.sleep(8)

    if votes:
        vote_counts = Counter(votes)
        majority = vote_counts.most_common(1)[0][0]
        consistency = vote_counts.most_common(1)[0][1] / len(votes)
        best_conf, best_expl = 0, ""
        for v, c, e in zip(votes, confidences, explanations):
            if v == majority and c > best_conf:
                best_conf, best_expl = c, e
    else:
        majority, consistency, best_conf, best_expl = "Unlabeled", 0, 0, "LLM failed"

    return majority, consistency, best_conf, best_expl, votes

# =====================
# MAIN LOOP
# =====================
all_summary = []

for book in BOOKS:
    print("\n" + "=" * 65)
    print(f"LABELING: {book['name']}")
    print("=" * 65)

    OUT = Path(f"results_{book['folder']}")
    gsm_file = OUT / 'gsm_comparison.json'

    if not gsm_file.exists():
        print(f"  GSM file not found: {gsm_file} — skipping")
        continue

    with open(gsm_file) as f:
        comparison = json.load(f)

    weighted_clusters = comparison['weighted']['clusters']
    valid_clusters = [c for c in weighted_clusters if c['cv'] >= CV_THRESHOLD]
    print(f"  Valid clusters (CV>={CV_THRESHOLD}): {len(valid_clusters)}")

    # COSINE LABELING
    print(f"\n  --- COSINE LABELING ---")
    cosine_results = []
    for cluster in valid_clusters:
        label, label_ar, sim = cosine_label(cluster)
        cosine_results.append({
            'cluster_id': cluster['cluster_id'],
            'cv': cluster['cv'],
            'top_words': cluster['top_words'],
            'label': label,
            'label_ar': label_ar,
            'similarity': sim,
        })
        print(f"    Cluster {cluster['cluster_id']}: {label} (sim={sim:.4f})")

    # LLM LABELING
    print(f"\n  --- LLM LABELING (3-run majority voting) ---")
    llm_results = []
    for cluster in valid_clusters:
        print(f"\n    Cluster {cluster['cluster_id']} (CV={cluster['cv']:.4f})")
        print(f"    Words: {' | '.join(cluster['top_words'][:6])}")
        label, consistency, conf, expl, votes = llm_label(cluster)
        llm_results.append({
            'cluster_id': cluster['cluster_id'],
            'cv': cluster['cv'],
            'top_words': cluster['top_words'],
            'label': label,
            'confidence': conf,
            'consistency': consistency,
            'explanation': expl,
            'all_votes': votes,
        })
        print(f"    FINAL: {label} | conf={conf:.2f} | consistency={consistency:.0%}")

    # COMPARISON
    print(f"\n  --- COMPARISON ---")
    print(f"  {'Cluster':<10} {'CV':<8} {'Cosine':<38} {'LLM':<38} Match")
    print("  " + "-" * 100)
    match_count = 0
    for cos, llm in zip(cosine_results, llm_results):
        match = "✓" if cos['label'] == llm['label'] else "✗"
        if cos['label'] == llm['label']:
            match_count += 1
        c_short = cos['label'][:36]
        l_short = llm['label'][:36]
        print(f"  {cos['cluster_id']:<10} {cos['cv']:<8.4f} {c_short:<38} {l_short:<38} {match}")

    agreement = match_count / len(cosine_results) * 100 if cosine_results else 0
    print(f"\n  Agreement: {match_count}/{len(cosine_results)} ({agreement:.0f}%)")
    print(f"  Avg LLM confidence: {np.mean([r['confidence'] for r in llm_results]):.2f}")
    print(f"  Avg LLM consistency: {np.mean([r['consistency'] for r in llm_results]):.0%}")

    # SAVE
    with open(OUT / 'cosine_labeling.json', 'w', encoding='utf-8') as f:
        json.dump(cosine_results, f, ensure_ascii=False, indent=2)
    with open(OUT / 'llm_labeling.json', 'w', encoding='utf-8') as f:
        json.dump(llm_results, f, ensure_ascii=False, indent=2)

    report_path = OUT / 'LABELING_REPORT.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"LABELING REPORT: {book['name']}\n")
        f.write(f"K={book['k']}, Valid Clusters={len(valid_clusters)}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Cluster':<10} {'CV':<8} {'Cosine Label':<38} {'LLM Label':<38}\n")
        f.write("-" * 94 + "\n")
        for cos, llm in zip(cosine_results, llm_results):
            f.write(f"{cos['cluster_id']:<10} {cos['cv']:<8.4f} {cos['label']:<38} {llm['label']:<38}\n")
        f.write(f"\nAgreement: {agreement:.0f}%\n")
        f.write(f"Avg LLM Confidence: {np.mean([r['confidence'] for r in llm_results]):.2f}\n")
        f.write(f"Avg Consistency: {np.mean([r['consistency'] for r in llm_results]):.0%}\n\n")
        f.write("LLM DETAILS:\n" + "-" * 40 + "\n")
        for llm in llm_results:
            f.write(f"\nCluster {llm['cluster_id']} (CV={llm['cv']:.4f}): {llm['label']}\n")
            f.write(f"  Words: {' | '.join(llm['top_words'][:6])}\n")
            f.write(f"  Confidence: {llm['confidence']:.2f} | Consistency: {llm['consistency']:.0%}\n")
            f.write(f"  All votes: {llm['all_votes']}\n")
            f.write(f"  Explanation: {llm['explanation'][:250]}\n")

    print(f"\n  Saved to {OUT}/")
    all_summary.append({
        'book': book['name'], 'k': book['k'],
        'valid_clusters': len(valid_clusters),
        'agreement': round(agreement, 1),
        'avg_conf': round(np.mean([r['confidence'] for r in llm_results]), 2),
        'avg_consistency': round(np.mean([r['consistency'] for r in llm_results]), 2),
    })

print("\n" + "=" * 65)
print("ALL BOOKS LABELING SUMMARY")
print("=" * 65)
print(f"  {'Book':<25} {'K':<6} {'Clusters':<10} {'Agreement':<12} {'Avg Conf':<12} {'Consistency'}")
print("  " + "-" * 75)
for s in all_summary:
    print(f"  {s['book']:<25} {s['k']:<6} {s['valid_clusters']:<10} {s['agreement']:<12.1f}% {s['avg_conf']:<12.2f} {s['avg_consistency']:.0%}")
print("\nDONE!")
