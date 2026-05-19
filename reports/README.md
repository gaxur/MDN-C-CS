# PDF Report Template (CVCS 2026)

This directory contains the LaTeX template for the final project report.

## Expected Report Structure

The suggested limit is 8 pages plus references.

```text
reports/
├── report_template.tex  # LaTeX template
├── report.pdf           # Final compiled PDF
└── images/              # Figures and plots
```

## Recommended Sections

### 1. Abstract

- Problem summary
- Study objective
- Brief methodology
- Key findings

### 2. Introduction

- What hallucination means in VLMs
- Why it matters for trustworthy multimodal systems
- Short related-work overview
- Project objectives

### 3. Evaluated Models

- LLaVA-v1.5
- Qwen3-VL
- Optional: InternVL3.5
- Relevant architectural and practical differences

### 4. Methodology

- Benchmark design
- Trap-question categories:
  - Object absent
  - Attribute
  - Relation
  - Count
- Evaluation criteria
- Experimental setup
- POPE-style metrics for negative yes/no probes

### 5. Experiments and Results

- Overall accuracy and hallucination rate
- Accuracy by question type
- Condition-level analysis
- Baseline vs strict-prompt mitigation

### 6. Analysis

- Which question type fails most often?
- Which visual or linguistic conditions increase hallucination?
- Why might those failures happen?
- Study limitations

### 7. Mitigation

- Prompt-engineering strategy
- Comparison against the baseline prompt
- Practical implications

### 8. Conclusions

- Main findings
- Safety implications
- Future work

### 9. References

References do not count toward the suggested 8-page limit.

## LaTeX Installation

To compile locally:

```bash
sudo apt install texlive-latex-recommended texlive-fonts-recommended
sudo apt install texlive-latex-extra
pdflatex report_template.tex
```

Overleaf is also a good option if you do not want to install LaTeX locally.
