# Project README

## Project Overview

This project aims to fine-tune the Mistral-7B-Instruct-v0.3 model on three
specialized tasks:

- **sa**: Salary prediction/classification
- **se**: Seniority prediction/classification
- **wa**: Work Arrangement prediction/classification

Using the provided sample scripts and notebooks, users can learn how to prepare
data, configure the model, run training and evaluation, and perform inference
with the fine-tuned models.

## Directory Structure

```
fine_tuning_evaluation/
├── __init__.py
├── common.py              # Common utilities and configurations
├── demo.py                # Inference example script
├── mistral_config.py      # Model and training hyperparameter settings
├── mistral_data.py        # Data loading and preprocessing
├── mistral_model.py       # Model definition and fine-tuning wrapper
├── mistral_raw.ipynb      # Exploratory data analysis notebook
├── mistral_demo.ipynb     # Interactive inference demonstration notebook
├── mistral_train.ipynb    # End-to-end training notebook
├── mistral_task_sa.ipynb  # Salary task notebook
├── mistral_task_se.ipynb  # Seniority task notebook
├── mistral_task_wa.ipynb  # Work Arrangement task notebook
```

## Environment Dependencies

```
# Core dependencies
datasets>=3.5.0
gensim>=4.3.3
ipykernel>=6.29.5
jupyter>=1.1.1
nltk>=3.9.1
numpy>=1.26.4
pandas>=2.2.3
peft>=0.15.1
scikit-learn>=1.6.1
scipy>=1.13.1
torch>=2.6.0
transformers>=4.51.1
trl>=0.16.1
protobuf>=5.29.4
sentencepiece>=0.2.0
markupsafe>=3.0.2
bitsandbytes>=0.41.5
gradio>=5.27.0
```

## Model & Training Configuration

1. Set the model name, training parameters, optimizer, and other hyperparameters
   in `mistral_config.py`.
2. Specify data paths and preprocessing logic in `mistral_data.py`.

## Data & Task Description

- **sa (Salary)**: Predict or classify salary.
- **se (Seniority)**: Predict or classify job seniority (e.g.,
  Junior/Senior/Lead).
- **wa (Work Arrangement)**: Predict or classify work arrangements (e.g.,
  Remote/OnSite/Hybrid).

Each task has a dedicated notebook (`mistral_task_sa.ipynb`,
`mistral_task_se.ipynb`, `mistral_task_wa.ipynb`) that can be run to see
step-by-step examples.

## Training & Evaluation

- Run the `mistral_train.ipynb` notebook for a full fine-tuning workflow.

## Inference Examples

- Use `demo.py` for command-line inference:
  ```bash
  python demo.py
  ```
- Or launch `mistral_demo.ipynb` for an interactive experience.

## Extensions & Customization

- To add a new task, refer to the task registration logic in `common.py` and
  create the corresponding DataLoader and notebook.
- Adjust hyperparameters such as learning rate and batch size in
  `mistral_config.py` to improve performance.

## References

- [Mistral-7B-Instruct-v0.3 Documentation](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- Hugging Face Transformers tutorials

---

_Author: Mingyuan Cui | Date: 2025-04-28_
