# Cleanup notes

This repository was rebuilt from the original folder with the goal of making it
clear enough for paper release.

## Removed

- `.git/`
- `__pycache__/` and `.pyc` files
- hardcoded Hugging Face tokens
- stale README content from another project
- duplicate files named `old version`, `copy`, `run_data_1`, `run_data_2`, etc.
- `Old_Situation_Data/`
- `case_study/`
- `steerability/`
- generated plot dumps from `analysis/`
- bulky raw model outputs from the old run
- separate `archive/` folder
- separate `paper/figures/` and `paper/tables/` folders

## Kept

- final MOSAIC dataset files
- sentiment audit files
- model, country, and dimension metadata
- clean reusable source package
- small runnable scripts for the full pipeline
- tests for dataset counts, prompt formatting, and response parsing

## Code changes

The old code mixed data formatting, model loading, prompting, parsing, and
analysis in long scripts. The cleaned repo separates them:

- `mosaic.dataset` builds the normalized dataset.
- `mosaic.prompts` builds standard and country prompts.
- `mosaic.model_runner` loads Hugging Face models without hardcoded secrets.
- `mosaic.evaluation` runs prompts and stores parsed responses.
- `mosaic.parsing` extracts selected options from model outputs.
- `mosaic.analysis` summarizes model scores.
- `mosaic.sentiment` runs the option sentiment audit.
- `mosaic.translation` and `mosaic.augmentation` keep construction utilities.
