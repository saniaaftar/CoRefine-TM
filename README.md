# CoRefine-TM

**CoRefine-TM: LLM-Guided Semantic Topic 
Refinement for Classical Arabic Hadith Corpora**

* Elsevier (Under Review)*

---

## Overview
CoRefine-TM is a semantic topic refinement 
framework for Classical Arabic Hadith texts. 
It integrates decoder-level negative sampling 
with a Grouping–Scoring–Modeling (G-S-M) 
pipeline to produce coherent, interpretable, 
and taxonomy-aligned topics without manual 
intervention.

## Novel Contributions
- **Probability-Weighted Pooling** — leverages 
  the learned topic-word distribution β to 
  construct discriminative cluster representations
- **LLM-Guided Majority Voting** — chain-of-thought 
  reasoning across three independent classification 
  runs for automated taxonomy alignment

---

## Requirements
```bash
conda create -n hadith python=3.11
conda activate hadith
pip install -r requirements.txt
```

---

## Usage

### 1. Training
```bash
python train_optimized.py
```

### 2. G-S-M Refinement
```bash
python run_gsm_all.py
```

### 3. LLM Labeling
```bash
python run_labeling_all.py
```

### 4. Baseline Comparison
```bash
python run_baselines.py
```

### 5. Margin Sensitivity Analysis
```bash
python run_margin_experiment.py
```

---

## Dataset
[Sanadset corpus](https://data.mendeley.com/datasets/5xth87zwb5)
— 6 major Hadith collections from the 
Sanadset corpus.

---

## Citation
```bibtex
@article{aftar2026corefine,
  title={CoRefine-TM: LLM-Guided Semantic 
         Topic Refinement for Classical Arabic 
         Hadith Corpora},
  author={Aftar, Sania and Beneventano, Domenico 
          and Riaz, Adnan and Janjua, Naeem 
          and Bergamaschi, Sonia},
  journal={ Elsevier},
  year={2026}
}
```

---

## License
MIT License
