# SEEK + UNSW COMP6713 Group Project 2025 Term 1

**Team Name:** SEEKDeep  
**Project:** SEEK + UNSW COMP6713 – Group Project 2025 Term 1

## Team Members
- Yize Li (z5430483)
- Aoran Ni (z5533215)
- Mingyuan Cui (z5270593)
- Will Ren (z5429870)

---

## Table of Contents
1. [Overview](#overview)
2. [Problem Definition](#problem-definition)
3. [Dataset & Preprocessing](#dataset--preprocessing)
   - 3.1. Data Sources
   - 3.2. Data Preprocessing
   - 3.3. Annotation & Agreement
4. [Modeling Approach](#modeling-approach)
   - 4.1. Baseline Methods
   - 4.2. Fine-tuning & Extensions
     - 4.2.1. GPT‑2 Medium
     - 4.2.2. Mistral‑7B‑Instruct‑v0.3
   - 4.3. Proprietary & Open‑Weight LLMs
   - 4.4. Tools & Frameworks
5. [Evaluation & Results](#evaluation--results)
   - 5.1. Quantitative Metrics & Qualitative Analysis
     - 5.1.1. GPT‑2 Medium
     - 5.1.2. Mistral‑7B‑Instruct‑v0.3
     - 5.1.3. Proprietary Models
     - 5.1.4. Open‑Weight Models
   - 5.2. Command‑Line & Demo
6. [Discussion & Lessons Learned](#discussion--lessons-learned)
   - 6.1. Cost vs. Speed vs. Accuracy Trade‑offs
   - 6.2. Open‑Weight vs. Proprietary Models
   - 6.3. Is Fine‑tuning Worth It?
7. [Conclusion & Future Work](#conclusion--future-work)
8. [References](#references)

---

## Overview
This project, conducted in partnership with SEEK, explores generative AI methods for extracting key information from job advertisements. We focus on three attributes:

- **Salary** – currency, amount, and frequency
- **Work Arrangement** – remote, on‑site, or hybrid
- **Seniority** – seniority level of the position

We evaluate a wide range of models, from lightweight fine‑tuned models (GPT‑2, Mistral‑7B) to large proprietary (GPT‑4.1, Claude‑3.7) and open‑weight (Llama 3.1, DeepSeek) LLMs. Our goal is to understand the trade‑offs between cost, speed, and accuracy, and to provide actionable recommendations for real‑world deployment.

---

## Problem Definition
**Research Questions:**
1. What are the trade‑offs between cost, speed, and accuracy?
2. Should we use open‑weight or proprietary models?
3. Is fine‑tuning worth the effort?

**Tasks:**
- **Salary extraction** – structured output (e.g., `50000-70000-AUD-ANNUAL`)
- **Work arrangement classification** – one of `Remote`, `OnSite`, `Hybrid`
- **Seniority classification** – six categories after mapping (see Section 3.2)

---

## Dataset & Preprocessing

### 3.1. Data Sources
The dataset was provided by SEEK and consists of three parts:
- **Unlabelled development set**
- **Labelled development set** (for salary, work arrangement, seniority)
- **Labelled test set**

### 3.2. Data Preprocessing
- **HTML cleaning** – removed HTML tags using `BeautifulSoup`.
- **Translation** – an English‑translated version of the salary dataset was created to evaluate multilingual capability.
- **Seniority mapping** – the original diverse role titles (e.g., “assistant manager”, “senior lead”) were consolidated into six main categories:
  - Entry‑Level / Junior
  - Intermediate / Experienced
  - Senior Individual Contributor
  - Manager / Supervisor
  - Executive / Director
  - Internship / Trainee

Mapping was applied to both development and test sets. For unmatched strings, fuzzy matching was used.

### 3.3. Annotation & Agreement
To generate pseudo‑labels for the unlabelled dataset, we employed a consensus approach:
- **GPT‑4.1** and **Claude‑3.7‑Sonnet** first labelled the data.
- Instances where they disagreed were adjudicated by **Gemini‑2.5**.
- Final labels were accepted only when at least two models agreed; discordant samples were discarded.

**Agreement Metrics** (795 valid samples):
- Cohen’s Kappa: **0.6846** (substantial agreement)
- Overall agreement rate: **93.71%**

This provided a reliable training set for fine‑tuning.

---

## Modeling Approach

### 4.1. Baseline Methods
- **Work Arrangement** – tokenization + Word2Vec/GloVe embeddings + Random Forest / SVM. Achieved F1 = 0.45.
- **Seniority** – rule‑based keyword matching on concatenated job title, summary, and detail. Achieved 38% accuracy.
- **Salary** – regex patterns with language‑specific time‑unit mapping. Achieved 76% accuracy (58.66% after removing “None” labels).

### 4.2. Fine‑tuning & Extensions

#### 4.2.1. GPT‑2 Medium
We fine‑tuned `openai-community/gpt2‑medium` (345M parameters) for each task.  
**Custom extensions:**
- Stopping criteria: generation stops when a closing parenthesis `)` is encountered.
- Approximate numeric‑range matching (5% tolerance) for salary evaluation.
- Language detection to filter non‑English samples for separate metrics.

#### 4.2.2. Mistral‑7B‑Instruct‑v0.3
We applied supervised fine‑tuning (SFT) with **4‑bit quantization** (NF4) and **LoRA** (rank=64).  
- Only ~0.5% of parameters (≈35M) were trainable, reducing VRAM usage by ~68% compared to full 16‑bit training.
- This enabled training on a consumer‑grade RTX 4090 Laptop (16GB) for the work arrangement task, and on Colab A100 for salary and seniority.
- The model was fine‑tuned on the consensus‑labelled dataset.

### 4.3. Proprietary & Open‑Weight LLMs
**Proprietary models:**
- GPT‑4.1, GPT‑4o, GPT‑4o‑mini, GPT‑4.1‑mini (OpenAI)
- Gemini‑2.5‑flash‑preview (Google)
- Claude‑3.7‑Sonnet, Claude‑3‑Haiku (Anthropic)

**Open‑weight models:**
- Llama 3.1‑70B, Llama 3.1‑8B, Llama‑4‑Maverick (Meta)
- Gemma‑3‑27B (Google)
- DeepSeek‑Chat (DeepSeek)

**Prompt engineering** involved iterative refinement based on error analysis. For models supporting function calling, we used it to enforce structured output and reduce token usage.  
All experiments were tracked using **MLflow** (parameters, metrics, artifacts, traces).

### 4.4. Tools & Frameworks
- **Hugging Face Transformers / Datasets** – model loading, tokenization, training
- **PyTorch** – custom dataset and training loops
- **MLflow** – experiment tracking, model registry, UI
- **Gradio** – interactive demos
- **BeautifulSoup, regex, scikit‑learn, matplotlib** – preprocessing and evaluation
- **Google Colab / local GPU** – training environment

---

## Evaluation & Results

### 5.1. Quantitative Metrics & Qualitative Analysis

#### 5.1.1. GPT‑2 Medium (Fine‑tuned)
| Task               | Accuracy | Precision | Recall | F1    | Training Time |
|--------------------|----------|-----------|--------|-------|---------------|
| Salary             | 90.65%   | 0.893     | 0.907  | 0.897 | 32 min        |
| Seniority          | 74.75%   | 0.743     | 0.747  | 0.742 | 36 min        |
| Work Arrangement   | 80.81%   | 0.806     | 0.808  | 0.805 | 10 min        |

**Qualitative insights:**
- **Salary:** Struggles with numeric ranges and deeply hidden salary data; often defaults to “0‑0‑None‑None”.
- **Seniority:** Fixates on numeric/salary cues, misclassifies junior roles, and struggles with ambiguous titles.
- **Work Arrangement:** Misunderstands hybrid cues; biases toward majority class.

#### 5.1.2. Mistral‑7B‑Instruct‑v0.3 (Fine‑tuned)
| Task               | Accuracy | Precision | Recall | F1    |
|--------------------|----------|-----------|--------|-------|
| Salary (full)      | 91.89%   | 0.919     | 0.920  | 0.916 |
| Salary (currency)  | 98.24%   | 0.982     | 0.982  | 0.982 |
| Salary (frequency) | 98.41%   | 0.984     | 0.984  | 0.984 |
| Work Arrangement   | 84.85%   | 0.848     | 0.848  | 0.849 |
| Seniority*         | 70.68%   | –         | –      | –     |

*Seniority evaluated via semantic similarity (threshold = 0.001).

**Training times:** Salary: ~4h (A100), Seniority: ~6h (A100), Work Arrangement: ~5 min (RTX 4090).

#### 5.1.3. Proprietary Models
**Accuracy patterns:**
- **Salary** – GPT‑4.1, GPT‑4o, Claude‑3.7 > 90%; Gemini‑2.5 ~80%; Claude‑Haiku ~60%.
- **Seniority** – all models ~60%, indicating inherent difficulty.
- **Work Arrangement** – all models 85–96%.

**Latency:**
- GPT‑4.1 and GPT‑4o highest latency (especially for salary/seniority).
- Gemini‑2.5‑flash and Claude‑Haiku fastest (<1 sec/sample).

**Cost (per sample):**
- Lowest: GPT‑4o‑mini, Claude‑Haiku, Gemini‑2.5‑flash (~$0.0001–0.0004).
- Highest: Claude‑3.7‑Sonnet, GPT‑4o, GPT‑4.1 (~$0.0015–0.0020).

**Trade‑off analysis** (3D Pareto frontiers):
- **Salary**: Best overall = GPT‑4o‑mini; highest accuracy = Claude‑3.7‑Sonnet.
- **Seniority**: Gemini‑2.5‑flash‑preview is the only Pareto‑optimal solution.
- **Work Arrangement**: Best overall = GPT‑4o‑mini; accuracy‑focused = Claude‑3.7‑Sonnet or GPT‑4.1.

#### 5.1.4. Open‑Weight Models
**Accuracy patterns:**
- Work arrangement: 90–95% (similar to proprietary).
- Salary: wide variation (60–95%); larger models (70B) outperform 8B.
- Seniority: all models 54–62%.

**Cost efficiency:**
- DeepSeek‑Chat and Claude‑Haiku lowest (~$0.0002/sample).
- Llama models from official Meta not cheaper; using third‑party providers (e.g., Together AI) reduces cost to ~1/3 while preserving accuracy.

**Latency:**
- Llama models fastest among open‑weight (1–3 sec/sample).

**Observation:** For intermediate‑difficulty tasks (e.g., salary), the 70B model outperforms the 8B model. For very easy or very hard tasks, size matters less.

### 5.2. Command‑Line & Demo
- **Gradio demos** were built for each fine‑tuned model (GPT‑2, Mistral‑7B). Users can input a job ad and receive extracted information.
- **Command‑line interface** (`run_experiments.py`) allows batch evaluation with configurable model, task, and sample range.
- **MLflow UI** provides comprehensive tracking of experiments, metrics, and artifacts (see screenshot in report).

---

## Discussion & Lessons Learned

### 6.1. Cost vs. Speed vs. Accuracy Trade‑offs
| Model (Inference)        | Accuracy (avg) | Inference Time | Cost (all tasks) |
|--------------------------|----------------|----------------|------------------|
| GPT‑4.1 (API)            | Very High      | 67.5 min       | ~$3.72           |
| Llama 3.1‑70B (open)     | High           | 53.0 min       | ~$4.80           |
| Mistral‑7B (fine‑tuned)  | High (task‑dep)| ~15h training  | Free (training compute) |
| GPT‑2 Medium (fine‑tuned)| Moderate       | ~80 min training| Free (training compute) |

**Recommendations:**
- **High accuracy, low volume** – proprietary APIs (e.g., GPT‑4.1).
- **Cost‑sensitive, high volume** – self‑host open‑weight models (e.g., Llama 3.1‑70B on Together AI).
- **Rapid prototyping** – fine‑tuned smaller models (GPT‑2, Mistral‑7B).

### 6.2. Open‑Weight vs. Proprietary Models
| Aspect               | Open‑Weight                                   | Proprietary (e.g., GPT‑4.1)                  |
|----------------------|-----------------------------------------------|----------------------------------------------|
| Accuracy             | Competitive for simple/moderate tasks; lags for complex | Slight edge across all tasks                 |
| Speed                | Varies with infrastructure                    | Consistent low latency                       |
| Cost                 | Much lower per‑query at scale                 | Pay‑per‑use; can be high for large volumes   |
| Deployment           | Requires GPU setup & expertise                | Simple API integration, easy scaling         |
| Customization        | Full control (fine‑tuning, architecture)      | Limited to prompt tuning / optional fine‑tuning |
| Data Privacy         | Fully on‑premises                             | Data routed through provider                 |

**Balanced approach:** Use open‑weight models for large‑scale, cost‑sensitive workloads; use proprietary models for high‑accuracy, low‑volume, or when infrastructure is constrained.

### 6.3. Is Fine‑tuning Worth It?
| Task                  | GPT‑4.1 (no FT) | GPT‑2 Med (FT) | Mistral‑7B (FT) |
|-----------------------|-----------------|----------------|-----------------|
| Salary                | 93.5%           | 90.7%          | 91.9%           |
| Seniority             | 63.3%           | 74.8%          | 70.7%           |
| Work Arrangement      | 96.0%           | 80.8%          | 84.8%           |

- Fine‑tuning **significantly boosts** seniority performance, even surpassing proprietary models.
- For salary, fine‑tuned models approach proprietary performance.
- For work arrangement, the small fine‑tuning dataset (99 samples) limited gains; larger or higher‑quality data would help.

**Conclusion:** Fine‑tuning is highly worthwhile when high‑quality, task‑specific data is available. It enables smaller models to compete with much larger ones, especially for complex tasks like seniority classification.

---

## Conclusion & Future Work
We developed and evaluated a range of models for information extraction from job ads. Key findings:

- **Mistral‑7B‑Instruct v0.3** delivers strong accuracy for salary and work arrangement, with acceptable training time (4–6 hours).
- **GPT‑2** is ideal for rapid prototyping in resource‑constrained environments.
- **GPT‑4.1** is recommended for tasks demanding advanced contextual understanding (e.g., seniority) when API costs are acceptable.
- **DeepSeek‑Chat** and **Llama 3.1‑70B** are excellent open‑source alternatives, offering cost efficiency at scale.

**Future work directions:**
- Improve test set quality, especially for seniority.
- Integrate structured knowledge (e.g., industry‑specific role hierarchies) using the `classification_name` and `subclassification_name` columns.
- Enhance multilingual capabilities to support SEEK’s global operations.
- Explore advanced fine‑tuning methods (e.g., Chain‑of‑Thought instruction tuning, post‑processing) and Retrieval‑Augmented Generation (RAG) with a knowledge base.

---

## References
1. Hugging Face, 2025. *GPT‑2 model documentation (Transformers v4.50.0)*. [online] Available at: https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/gpt2 [Accessed 28 Apr. 2025].
2. Hugging Face, 2025. *openai‑community/gpt2 model card*. [online] Available at: https://huggingface.co/openai-community/gpt2 [Accessed 28 Apr. 2025].
3. Jiang, A.Q., Sablayrolles, A., Mensch, A., et al., 2023. *Mistral 7B*. [online] Available at: https://arxiv.org/abs/2310.06825 [Accessed 28 Apr. 2025].
4. Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L., 2023. *QLoRA: Efficient finetuning of quantized LLMs*. arXiv:2305.14314.
5. Dettmers, T. and Zettlemoyer, L., 2023. *The case for 4‑bit precision: k‑bit inference scaling laws*. In: *International Conference on Machine Learning (ICML)*, PMLR, pp.7750–7774.