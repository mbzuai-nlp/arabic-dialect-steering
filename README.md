# Can Dialects Be Steered Like Languages? Sparse Neurons and Distributed Directions in Arabic LLMs

**Kareem Elozeiri\*, Mervat Abassy\*, Omar Kallas, Fahim Dalvi, Preslav Nakov, Kentaro Inui, Nadir Durrani**

Mohamed bin Zayed University of Artificial Intelligence · Qatar Computing Research Institute · Tohoku University · RIKEN

---

This repository contains the code for our paper, which investigates two complementary inference-time approaches for Arabic dialect steering in large language models: **neuron steering**, which identifies and rescales sparse dialect-associated MLP neurons, and **vector steering**, which extracts mean activation-difference directions from parallel dialect–MSA text and injects them during generation. Neither method requires fine-tuning.

## Contents

- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Data](#data)
  - [MADAR Parallel Corpus](#1-madar-parallel-corpus-required)
  - [AL-QASIDA Evaluation Benchmark](#2-al-qasida-evaluation-benchmark-required-for-eval_data)
- [Method 1: Neuron Steering](#method-1-neuron-steering)
- [Method 2: Vector Steering](#method-2-vector-steering)
- [Baselines](#baselines)
- [Evaluation](#evaluation)
- [Supported Models](#supported-models)
- [Citation](#citation)
- [License](#license)

## Repository Layout

```text
SteeringArabicDialects/
├── neuron_steering/          # Method 1: LAPE-based sparse neuron steering
│   ├── lape_extract.py           extract dialect-associated MLP neurons
│   ├── lape_analyze.py           analyze LAPE extraction outputs
│   ├── steer_dialect_generation.py  generate with LAPE-steered neurons
│   ├── measure_lape_residual_projection.py  residual-space coverage
│   ├── measure_lape_vector_coverage.py      vector-space coverage
│   ├── plot_lape_neuron_figures.py          paper figures (neuron counts)
│   ├── plot_lape_residual_projection_figures.py  paper figures (coverage)
│   ├── scripts/                  shell runners for all experiments
│   ├── experiment.jsonl          3-dialect MADAR manifest (MSA, Cairo, Rabat)
│   └── experiment_7dialects.jsonl  7-dialect MADAR manifest
│
├── arabic_steering_vector/   # Method 2: Activation-vector steering
│   ├── extract_dialect_vectors_fast.py   extract vectors from parallel corpus
│   ├── extract_dialect_vectors_prompt.py extract from prompt-position activations
│   ├── steer_dialect_and_compare.py      quick interactive steer & compare
│   ├── generate_steered_responses.py     batch layer sweep (ALLaM / Qwen)
│   ├── generate_steered_responses2.py    batch layer sweep (Fanar variant)
│   ├── generate_steered_responses_coeff_sweep.py  sweep coefficients & layers
│   ├── generate_steered_responses_token_ablation.py  ablate steered-token budget
│   ├── ablate_steer_tokens.py            interactive token-budget ablation
│   ├── analyze_sample_size_vector_sensitivity.py  sensitivity to corpus size
│   ├── sample_tsv_splits.py              create random TSV sample splits
│   ├── visualize_vector_similarity.py    cross-model cosine similarity plots
│   ├── lm_eval_steered_api.py            OpenAI-compatible server for lm-eval
│   ├── create_human_eval_sheet.py        build annotator Excel sheets
│   ├── activation_steer.py               shared steering hook (imported by others)
│   ├── dialect_vectors/                  precomputed vectors (ALLaM, Fanar, Jais)
│   └── eval_data/                        per-dialect evaluation prompts (JSONL)
│
├── baselines/                # Explicit-prompt baseline
│   ├── generate_explicit_prompt_eval.py
│   └── run_prompt_baseline.sh
│
└── llm_as_a_judge/           # LLM-as-judge dialect authenticity scoring
    └── openrouter_judge.py
```

## Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # GPU
pip install transformers accelerate sentencepiece safetensors
pip install numpy pandas scikit-learn matplotlib seaborn tqdm openpyxl
```

Or install all at once from the provided requirements file:

```bash
pip install -r requirements.txt
```

## Data

Two external, license-gated datasets are needed and are **not** bundled in this repository. Both must be downloaded before running the scripts below.

### 1. MADAR Parallel Corpus (required)

Both steering methods rely on the **MADAR Parallel Corpus** (Bouamor et al., 2018). The corpus is freely available for research use after completing a short registration form at:

> https://camel.abudhabi.nyu.edu/madar-parallel-corpus/

After downloading, place the TSV files under `arabic_steering_vector/data/`:

```text
arabic_steering_vector/data/
├── MADAR.corpus.MSA.tsv
├── MADAR.corpus.Cairo.tsv
├── MADAR.corpus.Beirut.tsv
├── MADAR.corpus.Rabat.tsv
├── MADAR.corpus.Aleppo.tsv
├── MADAR.corpus.Doha.tsv
├── MADAR.corpus.Riyadh.tsv
├── MADAR.corpus.Damascus.tsv
├── MADAR.corpus.Tunis.tsv
├── MADAR.corpus.Khartoum.tsv
├── MADAR.corpus.Jerusalem.tsv
├── MADAR.corpus.Jeddah.tsv
├── MADAR.corpus.Algiers.tsv
└── MADAR.corpus.Fes.tsv
```

The MADAR manifest files (`neuron_steering/experiment.jsonl`, `neuron_steering/experiment_7dialects.jsonl`) reference these paths and are used by `lape_extract.py`.

### 2. AL-QASIDA Evaluation Benchmark (required for `eval_data`)

The generation-eval scripts throughout this repo (`generate_steered_responses*.py`, `steer_dialect_generation.py`, `generate_explicit_prompt_eval.py`, and the shell runners under `neuron_steering/scripts/generation/`) read per-dialect prompt sets from `arabic_steering_vector/eval_data/`. These prompts come from the **AL-QASIDA** benchmark (Robinson et al., ACL 2025 Findings) and are not redistributed here.

Clone the **original** AL-QASIDA repository (not a fork):

```bash
git clone https://github.com/jhu-clsp/al-qasida
```

Then, inside `al-qasida/data_processing`, follow its `README.md`:

1. Place your downloaded MADAR-26 corpus (same corpus as above) under `./bitexts` as described there.
2. Run `python create_dataset.py` to build the monolingual prompt sets. The Egyptian/Moroccan/Saudi/Syrian/Sudanese/Algerian/Palestinian/Kuwaiti prompts used by this repo come from the `madar26` / `btec` source and are written to `data/mono/btec/madar26/<country>.csv` (country codes: `egy`, `mar`, `sau`, `syr`, `sdn`, `dza`, `pse`, `kwt`).

Convert each per-country CSV into a JSONL file with one prompt per line, matching the schema this repo's scripts expect:

```json
{"prompt": "...", "language": "Egyptian Arabic", "source": "MADAR-26", "genre": "btec", "dialect": "egy"}
```

and place the resulting files under `arabic_steering_vector/eval_data/`:

```text
arabic_steering_vector/eval_data/
├── egy.jsonl
├── mar.jsonl
├── sau.jsonl
├── syr.jsonl
├── sdn.jsonl
├── dza.jsonl
├── pse.jsonl
├── kwt.jsonl
├── msa_samples_300.jsonl   # MSA baseline samples
└── msa_samples_300.json    # same samples, JSON-array form (used by some scripts)
```

---

## Method 1: Neuron Steering

All scripts run from the repository root.

### Step 1 — Extract Dialect-Associated Neurons

`lape_extract.py` implements the LAPE (Language-Associating Probability Entropy) neuron identification pipeline. It reads parallel MADAR sentences for each dialect, runs them through the model, records per-neuron activation probabilities, and selects low-entropy (dialect-selective) neurons.

```bash
python neuron_steering/lape_extract.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --data_manifest neuron_steering/experiment_7dialects.jsonl \
  --out_dir extraction_output/allam_7dialects \
  --top_rate 0.01 \
  --filter_rate 0.95
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | required | HuggingFace model ID or local path |
| `--data_manifest` | required | JSONL mapping dialects to MADAR TSV paths |
| `--out_dir` | required | Directory for extraction outputs |
| `--top_rate` | `0.01` | Fraction of lowest-entropy neurons to keep (LAPE 1%) |
| `--filter_rate` | `0.95` | Global activation-probability quantile pre-filter |
| `--activation_bar_ratio` | `0.95` | Threshold for assigning neurons to dialects |
| `--batch_size` | `2` | Sentences per forward pass |
| `--max_parallel_sentences` | `None` | Optional sentence limit after ID intersection |
| `--dtype` | `auto` | Model dtype (`auto`, `float16`, `bfloat16`) |

The manifest JSONL format is:

```jsonl
{"dialect": "MSA", "path": "arabic_steering_vector/data/MADAR.corpus.MSA.tsv", "lang": "MSA"}
{"dialect": "CAI", "path": "arabic_steering_vector/data/MADAR.corpus.Cairo.tsv", "lang": "CAI"}
```

### Step 2 — Analyze Neuron Outputs

```bash
python neuron_steering/lape_analyze.py \
  --input_dir extraction_output/allam_7dialects
```

Writes tables and histograms to `extraction_output/allam_7dialects/analysis/`.

### Step 3 — Steer Generation

`steer_dialect_generation.py` loads the extracted neuron sets and steers generation by rescaling target-dialect neurons (factor `alpha`) and suppressing MSA neurons (factor `gamma`).

```bash
python neuron_steering/steer_dialect_generation.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --neurons_dir extraction_output/allam_7dialects \
  --target_dialect CAI \
  --out_file results/allam_cairo_steered.jsonl \
  --prompt_file arabic_steering_vector/eval_data/egy.jsonl \
  --alpha 1.3 \
  --gamma 0.7 \
  --generate_baseline
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | required | HuggingFace model ID or local path |
| `--neurons_dir` | required | LAPE extraction output directory |
| `--target_dialect` | required | Dialect code to amplify, e.g. `CAI`, `BEI`, `DOH`, `RAB` |
| `--out_file` | required | Output JSONL for generations |
| `--prompt_file` | — | JSONL or plain-text file of prompts |
| `--alpha` | `1.3` | Target neuron rescaling factor (>1 amplifies) |
| `--gamma` | `0.7` | MSA neuron rescaling factor (<1 suppresses) |
| `--msa_dialect` | `MSA` | Dialect label to suppress |
| `--generate_baseline` | — | Also save baseline (no-intervention) generations |
| `--suppress_competitor_dialects` | `""` | Comma-separated competitor dialects to suppress |
| `--max_new_tokens` | `128` | Max tokens to generate |

### Step 4 — Measure Neuron Coverage

**Residual-space projection** (how much of the residual dialect direction is explained by LAPE neurons):

```bash
python neuron_steering/measure_lape_residual_projection.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --neurons_dir extraction_output/allam_7dialects \
  --vector_path arabic_steering_vector/dialect_vectors/ALLaM-7B-Instruct-preview/Cairo_response_avg_diff.pt \
  --target_dialect CAI \
  --out_dir coverage_output/allam_cairo_residual
```

**Vector-space coverage** (overlap between LAPE neuron output directions and steering vectors):

```bash
python neuron_steering/measure_lape_vector_coverage.py \
  --neurons_dir extraction_output/allam_7dialects \
  --vectors_dir arabic_steering_vector/dialect_vectors/ALLaM-7B-Instruct-preview \
  --out_dir coverage_output/allam_vector_coverage
```

### Step 5 — Generate Paper Figures

```bash
python neuron_steering/plot_lape_neuron_figures.py \
  --out_dir paper_figures/neuron_counts

python neuron_steering/plot_lape_residual_projection_figures.py \
  --out_dir paper_figures/coverage
```

### Shell Runners

Pre-configured shell scripts for all experiments are under `neuron_steering/scripts/`:

```bash
# extraction
bash neuron_steering/scripts/extraction/run_lape_extract_analyze_7dialects_all_models.sh

# steered generation
bash neuron_steering/scripts/generation/run_all_dialects_explicit_prompt_inference.sh
bash neuron_steering/scripts/generation/run_coeff_abl_allam.sh

# coverage
bash neuron_steering/scripts/coverage/run_all_dialects_lape_residual_projection_allam_fanar_instruct.sh
```

See `neuron_steering/scripts/README.md` for a full list.

---

## Method 2: Vector Steering

All scripts run from the repository root.

### Step 1 — Extract Dialect Vectors

`extract_dialect_vectors_fast.py` reads parallel MADAR TSV files, computes mean hidden-state differences between each dialect and MSA, and saves a per-layer direction vector per dialect.

```bash
# extract vectors for specific dialects
python arabic_steering_vector/extract_dialect_vectors_fast.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --dialects Cairo Beirut Rabat \
  --data-dir arabic_steering_vector/data

# extract all available dialects
python arabic_steering_vector/extract_dialect_vectors_fast.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --all-dialects \
  --data-dir arabic_steering_vector/data
```

Vectors are saved to `arabic_steering_vector/dialect_vectors/<model_name>/<Dialect>_response_avg_diff.pt`.

**Precomputed vectors** for `humain-ai/ALLaM-7B-Instruct-preview`, `QCRI/Fanar-1-9B-Instruct`, and `inceptionai/Jais-2-8B-Chat` are already included under `arabic_steering_vector/dialect_vectors/`.

To extract from the prompt position instead of the response position:

```bash
python arabic_steering_vector/extract_dialect_vectors_prompt.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --all-dialects \
  --data-dir arabic_steering_vector/data
```

### Step 2 — Steer and Compare (Interactive)

`steer_dialect_and_compare.py` generates a baseline and a steered response for a single prompt and prints a side-by-side comparison.

```bash
python arabic_steering_vector/steer_dialect_and_compare.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --vector-path arabic_steering_vector/dialect_vectors/ALLaM-7B-Instruct-preview/Cairo_response_avg_diff.pt \
  --steer-dialect Egyptian \
  --layer 21 \
  --coef 3.0 \
  --prompt "كيف حالك اليوم؟"

# using a precomputed Qwen vector
python arabic_steering_vector/steer_dialect_and_compare.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --vector-path arabic_steering_vector/dialect_vectors/Qwen2.5-7B-Instruct/Levantine_response_avg_diff.pt \
  --steer-dialect Levantine \
  --layer 16 --coef 3.0 \
  --prompt "كيف حالك اليوم؟"
```

### Step 3 — Batch Evaluation (Layer Sweep)

`generate_steered_responses.py` runs baseline and steered generation over all eval prompts for a given dialect, sweeping across layers.

```bash
python arabic_steering_vector/generate_steered_responses.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --eval-file arabic_steering_vector/eval_data/egy.jsonl \
  --vector-path arabic_steering_vector/dialect_vectors/ALLaM-7B-Instruct-preview/Cairo_response_avg_diff.pt \
  --layers 18 19 20 21 22 \
  --coef 3.0 \
  --output-dir results_layers
```

Pre-configured batch runners:

```bash
bash arabic_steering_vector/generate_all_allam.sh
bash arabic_steering_vector/generate_all_fanar.sh
```

### Ablation Studies

**Coefficient and layer sweep:**

```bash
python arabic_steering_vector/generate_steered_responses_coeff_sweep.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --eval-file arabic_steering_vector/eval_data/egy.jsonl \
  --vector-path arabic_steering_vector/dialect_vectors/ALLaM-7B-Instruct-preview/Cairo_response_avg_diff.pt \
  --output-dir results_coeff

bash arabic_steering_vector/run_allam_coeff_sweep.sh
bash arabic_steering_vector/run_fanar_coeff_sweep.sh
```

**Steered-token budget ablation** (how many response tokens receive steering):

```bash
python arabic_steering_vector/generate_steered_responses_token_ablation.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --eval-file arabic_steering_vector/eval_data/egy.jsonl \
  --vector-path arabic_steering_vector/dialect_vectors/ALLaM-7B-Instruct-preview/Cairo_response_avg_diff.pt \
  --layers 21 \
  --n-steer-tokens 10 20 30 40 50 60 0 \
  --output-dir results_token_ablation

bash arabic_steering_vector/run_token_ablation.sh
```

**Corpus sample-size sensitivity:**

```bash
python arabic_steering_vector/sample_tsv_splits.py \
  --primary arabic_steering_vector/data/MADAR.corpus.Cairo.tsv \
  --paired arabic_steering_vector/data/MADAR.corpus.MSA.tsv \
  --sizes 1k 2k 4k 6k 12k \
  --out-dir arabic_steering_vector/data/sample_splits_size_sensitivity

python arabic_steering_vector/analyze_sample_size_vector_sensitivity.py \
  --runs-dir arabic_steering_vector/dialect_vectors/size_sensitivity \
  --dialect Cairo --layer 21
```

**Cross-model vector similarity:**

```bash
python arabic_steering_vector/visualize_vector_similarity.py \
  --layer 16 \
  --output-dir analysis_results_dialect/similarity
```

### lm-eval Integration

`lm_eval_steered_api.py` starts a local OpenAI-compatible completions server that applies activation steering to every request. Point any `lm-eval` run at this server to benchmark a steered model.

```bash
python arabic_steering_vector/lm_eval_steered_api.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --vector-path arabic_steering_vector/dialect_vectors/ALLaM-7B-Instruct-preview/Cairo_response_avg_diff.pt \
  --layer 21 --coef 3.0 \
  --port 8000
```

---

## Baselines

The explicit-prompt baseline generates responses using a dialect-instruction system prompt, without any activation intervention.

```bash
python baselines/generate_explicit_prompt_eval.py \
  --model humain-ai/ALLaM-7B-Instruct-preview \
  --eval-data arabic_steering_vector/eval_data \
  --dialects egy syr sau \
  --output-dir baselines/explicit_prompt_outputs

bash baselines/run_prompt_baseline.sh
```

---

## Evaluation

### LLM-as-Judge

`llm_as_a_judge/openrouter_judge.py` calls an OpenRouter-hosted model (default: `openai/gpt-4o-mini`) to score dialect authenticity, coherence, and Arabic fluency for all JSONL files in a directory.

```bash
export OPENROUTER_API_KEY=<your-key>

python llm_as_a_judge/openrouter_judge.py \
  results/allam_cairo_steered/ \
  --target-dialect Egyptian \
  --text-column steered_response \
  --prompt-column prompt \
  --output-jsonl results/allam_cairo_judge_scores.jsonl
```

The judge model and API key environment variable are configurable:

```bash
python llm_as_a_judge/openrouter_judge.py \
  results/ \
  --target-dialect Moroccan \
  --text-column steered_response \
  --judge-model anthropic/claude-3-5-sonnet \
  --api-key-env MY_OPENROUTER_KEY \
  --recursive
```

---

## Supported Models

The experiments in the paper use these Arabic-capable models:

| Model | HuggingFace ID |
|---|---|
| ALLaM-7B | `humain-ai/ALLaM-7B-Instruct-preview` |
| Fanar-1-9B | `QCRI/Fanar-1-9B-Instruct` |
| Jais-2-8B | `inceptionai/Jais-2-8B-Chat` |

Precomputed steering vectors are available for all three under `arabic_steering_vector/dialect_vectors/`.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{elozeiri-etal-2026-dialects,
  title     = {Can Dialects Be Steered Like Languages? {S}parse Neurons and Distributed Directions in {A}rabic {LLM}s},
  author    = {Elozeiri, Kareem and Abassy, Mervat and Kallas, Omar and Dalvi, Fahim and Nakov, Preslav and Inui, Kentaro and Durrani, Nadir},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
}
```

---

## License

This code is released under the MIT License. See [LICENSE](LICENSE).

The precomputed dialect vectors under `arabic_steering_vector/dialect_vectors/` are derived from the MADAR Parallel Corpus and are provided for research use only, consistent with the MADAR corpus license terms.
