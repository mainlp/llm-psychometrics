# Do LLMs give psychometrically plausible responses in educational assessments?

This repository contains code, data, and results for the paper [*Do LLMs give psychometrically plausible responses in educational assessments?*](https://aclanthology.org/2025.bea-1.21/)

```bibtex
@inproceedings{sauberli-etal-2025-llms,
  title = "Do {LLM}s Give Psychometrically Plausible Responses in Educational Assessments?",
  author = {S{\"a}uberli, Andreas and Frassinelli, Diego and Plank, Barbara},
  editor = {Kochmar, Ekaterina and Alhafni, Bashar and Bexte, Marie and Burstein, Jill and Horbach, Andrea and Laarmann-Quante, Ronja and Tack, Ana{\"i}s and Yaneva, Victoria and Yuan, Zheng},
  booktitle = "Proceedings of the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2025)",
  month = jul,
  year = "2025",
  address = "Vienna, Austria",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2025.bea-1.21/",
  pages = "266--278",
  ISBN = "979-8-89176-270-1",
}
```

## Repository structure

- `data/`: Scripts for scraping item data
  - `data/output/`: Scraped items, response distributions, IRT parameters, and metadata (excluding copyrighted material)
- `llm-responses/`: Scripts for generating LLM responses
  - `llm-responses/models/`: Configuration files for LLM models
  - `llm-responses/prompts/`: Configuration files for prompt templates
  - `llm-responses/output/`: Generated LLM responses
- `analysis/`: Scripts for results analysis and visualization
  - `analysis/output/`: Rendered Quarto report including analysis output
- `paper/`: Source code and figures for the paper

## Reproducing results

### Environment setup

Required software: [Python](https://www.python.org/) (tested with version 3.12)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Scraping and preprocessing data

#### NAEP dataset

Items, response distributions, and IRT parameters are included in this repository (`data/output/naep-*.jsonl`), but copyrighted reading passages have been excluded.

To scrape full item content and response distributions:

```bash
cd data
python scrape_items.py reading
python scrape_items.py history
python scrape_items.py economics
```

To re-scrape IRT parameters:

```bash
python scrape_parameters.py
```

#### CMCQRD dataset

Download `Cambridge Multiple-Choice Questions Reading Dataset.zip` from [the dataset webpage](https://englishlanguageitutoring.com/datasets/cambridge-multiple-choice-questions-reading-dataset) and place it in the `data/` directory. Then convert the items, response distributions, and IRT parameters:

```bash
cd data
python convert_cmcqrd.py
```

### Generating LLM responses

LLM responses are included in this repository (`llm-responses/output/*.jsonl`). They can be re-generated with `generate_responses.py`:

```bash
python generate_responses.py \
  --items path/to/items.jsonl \
  --model-config path/to/model-config.yaml \
  --prompt-config path/to/prompt-config.yaml \
  --output path/to/output.jsonl
```

Note that a different prompt config is used for reading items (`prompts/with-passage.yaml`) than for history and economics (`prompts/without-passage.yaml`).

To generate responses for all models and subjects:

```bash
cd llm-responses

for model in $(ls models); do
  model=${model%.yaml}

  # NAEP reading
  python generate_responses.py \
    --items ../data/output/naep-reading-items.jsonl \
    --model-config models/$model.yaml \
    --prompt-config prompts/with-passage.yaml \
    --output output/naep-reading-responses-$model.jsonl

  # NAEP history
  python generate_responses.py \
    --items ../data/output/naep-history-items.jsonl \
    --model-config models/$model.yaml \
    --prompt-config prompts/without-passage.yaml \
    --output output/naep-history-responses-$model.jsonl

  # NAEP economics
  python generate_responses.py \
    --items ../data/output/naep-economics-items.jsonl \
    --model-config models/$model.yaml \
    --prompt-config prompts/without-passage.yaml \
    --output output/naep-economics-responses-$model.jsonl

  # CMCQRD reading
  python generate_responses.py \
    --items ../data/output/cmcqrd-reading-items.jsonl \
    --model-config models/$model.yaml \
    --prompt-config prompts/with-passage.yaml \
    --output output/cmcqrd-reading-responses-$model.jsonl
done
```

### Analyzing results

```bash
cd analysis
python analysis.py
```

Generated plots will be saved in `paper/figures/`.

To reproduce the PDF report (`analysis/output/analysis.pdf`), install [Quarto](https://quarto.org/) and run `quarto render`.

## License

### Item data

Existing files under `data/output/` have been scraped from the [Nation's Report Card website](https://www.nationsreportcard.gov/).

> The following citation should be used when referencing all NCES products, including the National Assessment of Educational Progress (NAEP). The year and the name of the assessment you are using (e.g., 2024 Reading Assessment) should appear at the end of the statement.
>
> **U.S. Department of Education. Institute of Education Sciences, National Center for Education Statistics, National Assessment of Educational Progress (NAEP), 2024 Reading Assessment.**
>
> *(https://www.nationsreportcard.gov/faq.aspx#q21)*

Reading passages which are not included in this repository may be copyrighted material and should not be redistributed.

### Code

All code in this repository is licensed under the MIT License.
