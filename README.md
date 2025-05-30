# MMMLU Question Answering Experiments

## Overview

This project explores the performance of various Language Models (LLMs) on the Multilingual Massive Multitask Language Understanding (MMMLU) benchmark, focusing on German (DE_DE) and French (FR_FR) multiple-choice questions. The primary objective is to evaluate and compare different models and techniques, including Exploratory Data Analysis (EDA), finetuning of smaller models, and zero-shot evaluation of a larger model, in predicting the single correct answer letter.

## Project Components

The project is organized into several Jupyter notebooks:

1.  **`notebooks/eda.ipynb`**:
    * Performs an Exploratory Data Analysis on all 14 language 'test' splits of the `openai/mmmlu` dataset (196,588 examples).
    * Analyzes data schema, null values, language/answer/subject distributions, and token lengths (using `roberta-base` tokenizer) to inform preprocessing for finetuning.
    * Key finding: ~61.75% of 'Question + Choices' prompts are ≤ 256 tokens.

2.  **`notebooks/flan_t5_finetune.ipynb`**:
    * Finetunes the `google/flan-t5-base` model on the DE_DE and FR_FR 'test' splits of MMMLU.
    * Data is filtered for prompts (Question + Choices) < 256 tokens and split into 80-10-10 (train-val-test) sets, stratified by subject.
    * Trains for 2 epochs as a sequence-to-sequence task to predict the answer letter.
    * Baseline test accuracy: 0.2833; Post-finetuning test accuracy: 0.3332.
    * Uses Weights & Biases for logging. Model outputs saved in `outputs/flan_t5_mmmlu_fr_de`.

3.  **`notebooks/gemma_2_2b_finetune.ipynb`**:
    * Finetunes the `google/gemma-2-2b` model using 4-bit QLoRA on the DE_DE and FR_FR 'test' splits of MMMLU.
    * Dataset (28,084 examples initially) is filtered for prompts < 256 tokens (Gemma tokenizer), resulting in 23,696 examples, then split 80-10-10 (stratified).
    * Trains for 2 epochs with a specific prompt template.
    * Baseline test accuracy: 0.3751; Post-finetuning test accuracy: 0.5392.
    * Uses Weights & Biases for logging. Model outputs saved in `./outputs`.

4.  **`notebooks/gemma_3_27b_baseline.ipynb`**:
    * Evaluates the `gemma-3-27b-it` model in a zero-shot setting using LM Studio.
    * Uses the same DE_DE and FR_FR MMMLU 'test' splits, filtered for Q+Choices token length < 256 (using `google/gemma-2-2b-it` tokenizer), resulting in 2422 test samples.
    * Achieves an overall zero-shot accuracy of 0.6759.
    * Includes per-subject performance analysis.

## Dataset

The primary dataset is [openai/MMMLU](https://huggingface.co/datasets/openai/mmmlu).
* EDA uses all 14 available language 'test' splits.
* Finetuning and baseline evaluations focus on the DE_DE (German) and FR_FR (French) 'test' splits.
* A consistent preprocessing step across experiments is filtering examples where the combined "Question + Choices" (or prompt part for Gemma) is less than 256 tokens, determined by the respective model's tokenizer.

## Key Techniques

* **Exploratory Data Analysis (EDA)**: Understanding dataset characteristics.
* **Sequence-to-Sequence Finetuning**: For Flan-T5-Base.
* **QLoRA (4-bit Quantization with LoRA)**: For efficient finetuning of Gemma-2-2B.
* **Zero-Shot Evaluation**: For Gemma-3-27B via LM Studio.
* **Prompt Engineering**: Specific templates are used for instructing the models.
* **Stratified Splitting**: Ensuring balanced subject distribution across train/validation/test sets.
* **Token Length Filtering**: Managing input sizes for models.
* **Performance Logging**: Using Weights & Biases.

## Setup & Usage

1.  **Environment**: It's recommended to set up a Python virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
2.  **Dependencies**: Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    (Key libraries include `datasets`, `transformers`, `accelerate`, `bitsandbytes`, `peft`, `wandb`, `openai`, `scikit-learn`, `pandas`)
    *(Optional: If you plan to use `.env` files for managing environment variables, add `python-dotenv` to `requirements.txt` and install it.)*

3.  **Environment Variables**:
    * **Hugging Face Token**: For Gemma models, ensure you have accepted the license on Hugging Face and are authenticated. Set your `HF_TOKEN` environment variable or log in via `huggingface-cli login`.
    * **LM Studio Endpoint (for `gemma_3_27b_baseline.ipynb`)**:
        * Set the `LMSTUDIO_ENDPOINT` environment variable to your LM Studio server address (e.g., `http://localhost:1234/v1/`). The notebook defaults to `http://localhost:1234/v1/` if this variable is not set.
        * The `API_KEY` for LM Studio can also be set via the `LMSTUDIO_API_KEY` environment variable (defaults to "lm-studio" in the notebook).
4.  **Weights & Biases**:
    * The project is configured to log to W&B. Ensure you are logged in (`wandb login`).
    * W&B artifacts are stored in the `./wandb` directory by default.

## Directory Structure

* **`.venv/`**: Python virtual environment (intended to be gitignored).
* **`notebooks/`**: Contains the core Jupyter notebooks for EDA, finetuning, and evaluation. **This is the primary directory with the project's work.**
* **`outputs/`**: Default directory for saving finetuned models, evaluation results, and plots.
* **`wandb/`**: Contains local W&B logs (intended to be gitignored).
* **`models/`**: (Assumed) Potentially for storing base models if not dynamically downloaded (intended to be gitignored).
* **`requirements.txt`**: Python package dependencies.
* **`.gitignore`**: Specifies files and directories to be ignored by Git.

## Notes

* The `.gitignore` file is set up to primarily track the contents of the `notebooks/` directory and its subdirectories.
* Ensure `CUDA_LAUNCH_BLOCKING=1` is set if you encounter CUDA errors during training on relevant hardware.