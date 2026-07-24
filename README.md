# MOSAIC : A Dataset for Cultural Dimension Evaluation in Arabic LLMs

#### Nadhem Benhadjali*, Seifeddine Hamdi*, Istabrak Abbes, Safa Messaoud, Ines Arous

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![Dataset](https://img.shields.io/badge/dataset-1%2C483%20Arabic%20scenarios-green)](#dataset)
[![Dimensions](https://img.shields.io/badge/Hofstede-5%20dimensions-purple)](#cultural-dimensions)
[![Status](https://img.shields.io/badge/status-research%20repo-lightgrey)](#citation)

MOSAIC is a dataset and evaluation pipeline for studying how Arabic and
multilingual large language models respond to social dilemmas across cultural
value dimensions.

The dataset contains Arabic scenarios framed as two-choice dilemmas. Each
choice reflects one side of a cultural dimension, allowing models to be compared
through their option preferences rather than free-form explanations.

## Overview

This repository contains the cleaned code and data release for:

**MOSAIC: A Dataset for Cultural Dimension Evaluation in Arabic LLMs**

The project evaluates cultural tendencies in LLM outputs using Arabic social
scenarios grounded in Hofstede's cultural dimensions. It supports three main
workflows:

1. Building and validating the MOSAIC dataset.
2. Running standard model evaluation on Arabic dilemmas.
3. Running country-conditioned evaluation across Arab country contexts.

## Dataset

The main dataset is available at:

```text
data/final/mosaic_all.csv
```

It contains **1,483 Arabic social dilemmas** across five cultural dimensions.
Each row includes a scenario, two answer options, and labels indicating the
cultural side represented by each option.

| Dimension code | Dimension | Samples |
|---|---:|---:|
| `idv` | Individualism vs. Collectivism | 299 |
| `lto` | Long-Term vs. Short-Term Orientation | 300 |
| `mas` | Masculinity vs. Femininity | 292 |
| `pdi` | Power Distance | 294 |
| `uai` | Uncertainty Avoidance | 298 |
| **Total** |  | **1,483** |

The same data is also split by dimension:

```text
data/final/by_dimension/idv.csv
data/final/by_dimension/lto.csv
data/final/by_dimension/mas.csv
data/final/by_dimension/pdi.csv
data/final/by_dimension/uai.csv
```

### Dataset columns

| Column | Description |
|---|---|
| `id` | Stable example identifier. |
| `dimension` | Short dimension code: `idv`, `lto`, `mas`, `pdi`, or `uai`. |
| `dimension_name` | Full cultural dimension name. |
| `prompt_ar` | Arabic scenario/question. |
| `option_1_ar` | First Arabic answer option. |
| `option_1_label` | Cultural side represented by option 1. |
| `option_2_ar` | Second Arabic answer option. |
| `option_2_label` | Cultural side represented by option 2. |
| `source_file` | Original source CSV used to build the normalized dataset. |

## Cultural dimensions

MOSAIC follows five Hofstede-style dimensions:

| Code | Dimension | What it measures |
|---|---|---|
| `pdi` | Power Distance | Acceptance of hierarchy and unequal authority. |
| `idv` | Individualism vs. Collectivism | Preference for autonomy or group responsibility. |
| `mas` | Masculinity vs. Femininity | Emphasis on competitiveness or care-oriented values. |
| `uai` | Uncertainty Avoidance | Preference for structure under ambiguity. |
| `lto` | Long-Term vs. Short-Term Orientation | Planning for the future or prioritizing tradition. |

## Repository layout

```text
mosaic/
├── configs/                 # Model, country, and dimension metadata
├── data/
│   ├── source/dimensions/   # Source dimension CSVs
│   ├── final/               # Public normalized dataset
│   └── audits/sentiment/    # Sentiment and neutrality audit files
├── scripts/                 # Reproducible command-line entry points
├── src/mosaic/              # Reusable Python package
├── results/                 # Generated prompts, model outputs, summaries
├── tests/                   # Lightweight sanity tests
├── .env.example             # Local environment template
├── pyproject.toml           # Package metadata
└── requirements.txt         # Minimal dependencies
```

The Python package is grouped by workflow:

```text
src/mosaic/
├── core/        # Shared config, file paths, schemas
├── data/        # Dataset loading, building, validation
├── generation/  # Translation and augmentation utilities
├── evaluation/  # Prompt builders, model runner, parsers
├── analysis/    # Score aggregation and summaries
└── audit/       # Sentiment and quality audit utilities
```

## Installation

Clone the repository and install it in editable mode:

```bash
git clone <repo-url>
cd mosaic
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For gated Hugging Face models, create a local `.env` file:

```bash
cp .env.example .env
```

Then add your token:

```bash
HF_TOKEN=your_huggingface_token_here
```

Do not commit `.env`.

## Quick start

Load the dataset in Python:

```python
import pandas as pd

mosaic = pd.read_csv("data/final/mosaic_all.csv")
print(mosaic.shape)
print(mosaic["dimension"].value_counts())
```

Build the normalized dataset from the source dimension files:

```bash
python scripts/00_build_dataset.py
```

Run the test suite:

```bash
pytest
```

## Reproducing the evaluation

### 1. Build standard evaluation prompts

```bash
python scripts/02_build_prompts.py
```

This writes prompt files to:

```text
results/prompts/standard/
```

### 2. Run a model on one dimension

```bash
python scripts/03_run_evaluation.py \
  --model Fanar \
  --dimension idv
```

Raw model generations are written to:

```text
results/raw/standard/
```

### 3. Analyze standard evaluation outputs

```bash
python scripts/04_analyze_results.py
```

Summaries are written to:

```text
results/summaries/
```

## Country-conditioned evaluation

MOSAIC also supports prompts that ask models to answer from a specific Arab
country context. The country list is stored in:

```text
configs/countries.yaml
```

Build country-conditioned prompts:

```bash
python scripts/05_build_country_prompts.py
```

Run country-conditioned evaluation:

```bash
python scripts/06_run_country_evaluation.py \
  --model Fanar \
  --dimension idv
```

Analyze country-conditioned results:

```bash
python scripts/07_analyze_country_results.py
```

## Dataset construction utilities

The repository also keeps the translation and augmentation utilities used for
reproducibility.

Translate scenarios:

```bash
python scripts/08_translate.py \
  --input input.csv \
  --output translated.csv
```

Augment a dimension:

```bash
python scripts/09_augment.py \
  --input data.csv \
  --output augmented.csv \
  --dimension idv
```

Run a sentiment audit:

```bash
python scripts/01_audit_sentiment.py --dimension idv
```

## Configuration

Model, country, and dimension metadata live in `configs/`:

```text
configs/models.yaml
configs/countries.yaml
configs/dimensions.yaml
```

Edit these files to add new models, change country contexts, or extend the
cultural dimension metadata.

## Outputs

Generated files are intentionally separated from the dataset.

```text
results/prompts/      # Prompt files sent to models
results/raw/          # Raw model generations
results/summaries/    # Parsed scores and aggregate tables
results/figures/      # Optional generated plots
```

For a clean public repository, raw model outputs can be regenerated and do not
need to be committed unless required for a release artifact.

## Responsible use

MOSAIC is an evaluation resource. It should not be used to make claims about
individual people or to assign fixed cultural traits to Arab societies.

The dataset uses binary choices to make model behavior measurable, but real
social situations are more nuanced. Hofstede-style dimensions are useful for
structured comparison, yet they can simplify culture and may amplify stereotypes
when used without care.

Use MOSAIC to analyze model behavior, not to define what any culture is or
should be.

## Citation

Please cite the paper if you use this dataset or code.

```bibtex
@inproceedings{benhadjali2025mosaic,
  title     = {MOSAIC: A Dataset for Cultural Dimension Evaluation in Arabic LLMs},
  author    = {Benhadjali, Nadhem and Hamdi, Seifeddine and Abbes, Istabrak
               and Messaoud, Safa and Arous, Ines},
  booktitle = {NeurIPS 2025 Workshop: Muslims in ML},
  year      = {2025}
}
```

## Contact

For questions about the dataset or evaluation pipeline, please open an issue or
contact the authors listed in the paper.
