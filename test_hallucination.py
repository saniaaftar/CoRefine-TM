"""
Hallucination Test for LLM Labeling
Tests if LLM gives correct labels or hallucinates.
"""

import json
import time
import requests
from pathlib import Path

OUTPUT_DIR = Path('results_bukhari')

ANTHROPIC_API_KEY = "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"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

# Load taxonomy
with open('../data/taxonomy/taxonomy_converted.json', 'r', encoding='utf-8') as f:
    taxonomy = json.load(f)

tax_string = ""
for cat in taxonomy['categories']:
    tax_string += f"\n{cat['id']}. {cat['name_ar']} ({cat['name_en']})\n"
    for sub in cat['subcategories']:
        descs = ", ".join(sub['descriptors'][:5])
        tax_string += f"   {sub['id']} {sub['name_ar']} ({sub['name_en']}) - {descs}\n"

def call_llm(prompt):
    try:
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        response = requests.post(ANTHROPIC_URL, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data["content"][0]["text"]
        else:
            print(f"  API error: {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def classify_cluster(words_str, cluster_name):
    prompt = f"""You are an expert Islamic scholar specializing in Hadith classification.

TASK: Classify the following topic cluster into the most appropriate Hadith taxonomy category.
If the words do NOT clearly belong to any category, set main_category to "Unlabeled".

TOPIC CLUSTER WORDS:
{words_str}

HADITH TAXONOMY:
{tax_string}

Respond ONLY in this exact JSON format (no markdown, no backticks):
{{"main_category": "...", "main_category_ar": "...", "subcategory": "...", "subcategory_ar": "...", "confidence": 0.0, "key_evidence_words": ["...", "..."], "explanation": "..."}}"""

    response = call_llm(prompt)
    if response:
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("\n", 1)[1]
            if response.endswith("```"):
                response = response.rsplit("```", 1)[0]
            return json.loads(response.strip())
        except:
            return {"error": "parse_failed", "raw": response[:200]}
    return {"error": "api_failed"}

# ============================================
# TEST CASES
# ============================================

test_cases = [
    # TEST 1: OBVIOUS TOPICS (should get correct label, high confidence)
    {
        "name": "TEST 1a: Clear Prayer Topic",
        "words": "صلاه، ركوع، سجود، وضوء، قبله، اذان، جماعه، امام",
        "expected_category": "Worship",
        "expected_sub": "Prayer",
        "test_type": "obvious",
    },
    {
        "name": "TEST 1b: Clear Fasting Topic",
        "words": "صيام، رمضان، سحور، افطار، نيه، اعتكاف، امساك",
        "expected_category": "Worship",
        "expected_sub": "Fasting",
        "test_type": "obvious",
    },
    {
        "name": "TEST 1c: Clear Pilgrimage Topic",
        "words": "حج، عمره، طواف، كعبه، عرفه، منى، احرام، تلبيه",
        "expected_category": "Worship",
        "expected_sub": "Pilgrimage",
        "test_type": "obvious",
    },
    {
        "name": "TEST 1d: Clear Charity Topic",
        "words": "زكاه، صدقه، مال، فقراء، مساكين، انفاق، زكاه الفطر",
        "expected_category": "Charity and Social Justice",
        "expected_sub": "Obligatory Charity (Zakat)",
        "test_type": "obvious",
    },
    {
        "name": "TEST 1e: Clear Day of Judgment Topic",
        "words": "القيامه، البعث، الحساب، الميزان، الصراط، الشفاعه، الساعه",
        "expected_category": "Belief and Theology",
        "expected_sub": "The Day of Judgment",
        "test_type": "obvious",
    },

    # TEST 2: RANDOM/GARBAGE (should get Unlabeled or very low confidence)
    {
        "name": "TEST 2a: Random Arabic Words",
        "words": "سياره، كمبيوتر، هاتف، طائره، قطار، شاشه، لوحه",
        "expected_category": "Unlabeled",
        "expected_sub": "None",
        "test_type": "garbage",
    },
    {
        "name": "TEST 2b: Numbers and Common Words",
        "words": "واحد، اثنين، ثلاثه، كبير، صغير، طويل، قصير",
        "expected_category": "Unlabeled",
        "expected_sub": "None",
        "test_type": "garbage",
    },

    # TEST 3: MIXED CATEGORIES (should get low confidence or partial match)
    {
        "name": "TEST 3a: Prayer + Fasting Mixed",
        "words": "صلاه، صيام، ركوع، رمضان، سجود، سحور، وضوء، افطار",
        "expected_category": "Mixed",
        "expected_sub": "Mixed",
        "test_type": "mixed",
    },
    {
        "name": "TEST 3b: All Worship Mixed",
        "words": "صلاه، صيام، حج، زكاه، اذان، رمضان، طواف، صدقه",
        "expected_category": "Worship",
        "expected_sub": "Mixed",
        "test_type": "mixed",
    },

    # TEST 4: TRICKY/ADVERSARIAL (words that could mislead)
    {
        "name": "TEST 4a: Narrator Names Only",
        "words": "محمد، ابراهيم، عمر، عثمان، علي، خالد، زيد، انس",
        "expected_category": "Unlabeled",
        "expected_sub": "None",
        "test_type": "adversarial",
    },
    {
        "name": "TEST 4b: Ambiguous - Could be Ethics or Law",
        "words": "ظلم، عدل، حكم، قضاء، شهاده، حق، باطل",
        "expected_category": "Law and Prohibitions or Ethics",
        "expected_sub": "Ambiguous",
        "test_type": "adversarial",
    },
]

# ============================================
# RUN TESTS
# ============================================
print("=" * 70)
print("HALLUCINATION TEST FOR LLM LABELING")
print("=" * 70)

results = []
pass_count = 0
fail_count = 0

for test in test_cases:
    print(f"\n{'─' * 60}")
    print(f"{test['name']}")
    print(f"  Words: {test['words']}")
    print(f"  Expected: {test['expected_category']} > {test['expected_sub']}")
    
    result = classify_cluster(test['words'], test['name'])
    time.sleep(4)
    
    if 'error' in result:
        print(f"  ERROR: {result['error']}")
        verdict = "ERROR"
    else:
        label = f"{result.get('main_category', '?')} > {result.get('subcategory', '?')}"
        conf = result.get('confidence', 0)
        expl = result.get('explanation', '')[:150]
        
        print(f"  Got: {label}")
        print(f"  Confidence: {conf}")
        print(f"  Explanation: {expl}")
        
        # Evaluate
        if test['test_type'] == 'obvious':
            if test['expected_category'].lower() in result.get('main_category', '').lower():
                verdict = "PASS"
                pass_count += 1
                print(f"  VERDICT: PASS ✓ (correct category, conf={conf})")
            else:
                verdict = "FAIL - HALLUCINATION"
                fail_count += 1
                print(f"  VERDICT: FAIL ✗ (wrong category!)")
        
        elif test['test_type'] == 'garbage':
            if result.get('main_category', '') == 'Unlabeled' or conf < 0.5:
                verdict = "PASS"
                pass_count += 1
                print(f"  VERDICT: PASS ✓ (correctly uncertain, conf={conf})")
            else:
                verdict = "FAIL - HALLUCINATION"
                fail_count += 1
                print(f"  VERDICT: FAIL ✗ (confident on garbage! conf={conf})")
        
        elif test['test_type'] == 'mixed':
            if conf < 0.8:
                verdict = "PASS"
                pass_count += 1
                print(f"  VERDICT: PASS ✓ (lower confidence for mixed, conf={conf})")
            else:
                verdict = "PARTIAL"
                pass_count += 0.5
                print(f"  VERDICT: PARTIAL (high confidence on mixed, conf={conf})")
        
        elif test['test_type'] == 'adversarial':
            if result.get('main_category', '') == 'Unlabeled' or conf < 0.7:
                verdict = "PASS"
                pass_count += 1
                print(f"  VERDICT: PASS ✓ (handled adversarial case)")
            else:
                verdict = "PARTIAL"
                pass_count += 0.5
                print(f"  VERDICT: PARTIAL (conf={conf})")
        
        result['verdict'] = verdict
        result['test_name'] = test['name']
        result['test_type'] = test['test_type']
        result['expected'] = f"{test['expected_category']} > {test['expected_sub']}"
    
    results.append(result)

# ============================================
# SUMMARY
# ============================================
total = len(test_cases)
print(f"\n{'=' * 70}")
print(f"HALLUCINATION TEST SUMMARY")
print(f"{'=' * 70}")
print(f"\n  Total tests:  {total}")
print(f"  Passed:       {pass_count}/{total}")
print(f"  Failed:       {fail_count}/{total}")
print(f"  Pass rate:    {pass_count/total*100:.0f}%")

print(f"\n  By test type:")
for ttype in ['obvious', 'garbage', 'mixed', 'adversarial']:
    type_tests = [r for r in results if r.get('test_type') == ttype]
    type_pass = sum(1 for r in type_tests if r.get('verdict', '').startswith('PASS'))
    print(f"    {ttype:<15}: {type_pass}/{len(type_tests)} passed")

# Save
with open(OUTPUT_DIR / 'hallucination_test.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

with open(OUTPUT_DIR / 'HALLUCINATION_REPORT.txt', 'w', encoding='utf-8') as f:
    f.write("LLM HALLUCINATION TEST RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Pass rate: {pass_count/total*100:.0f}%\n\n")
    for r in results:
        f.write(f"{r.get('test_name', '?')}\n")
        f.write(f"  Expected: {r.get('expected', '?')}\n")
        got = f"{r.get('main_category', '?')} > {r.get('subcategory', '?')}"
        f.write(f"  Got: {got}\n")
        f.write(f"  Confidence: {r.get('confidence', '?')}\n")
        f.write(f"  Verdict: {r.get('verdict', '?')}\n\n")

print(f"\nSaved: hallucination_test.json, HALLUCINATION_REPORT.txt")
print("DONE!")
