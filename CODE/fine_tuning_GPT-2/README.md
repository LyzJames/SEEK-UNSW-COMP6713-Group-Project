# Fine-Tuned GPT-2 Models for Job Information Prediction

This project fine-tunes the GPT-2 language model (specifically **gpt2-medium**) to analyze job posting text and predict three different attributes:

- **Salary Range**
- **Seniority Level**
- **Work Arrangement** (e.g., remote, hybrid, on-site)

Each attribute has its own fine-tuning notebook and an accompanying Gradio web interface for interactive demos.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation Instructions](#installation-instructions)
3. [Model Training](#model-training)
4. [Running the Gradio Apps](#running-the-gradio-apps)
5. [Project Structure](#project-structure)
6. [Technologies Used](#technologies-used)
7. [Acknowledgments](#acknowledgments)

---

## Project Overview

This repository contains notebooks for fine-tuning GPT-2 on custom job posting datasets, each targeting a different prediction task:

- `fine-tuned_GPT2_salary.ipynb`: Salary range prediction
- `fine-tuned_GPT2_seniority.ipynb`: Seniority level prediction
- `fine-tuned_GPT2_work_arrangements.ipynb`: Work arrangement prediction

After training, each model can be served with a simple Gradio interface found in:

- `Gradio_Salary.ipynb`
- `Gradio_Seniority.ipynb`
- `Gradio_Work_Arrangements.ipynb`

---

## Installation Instructions

1. **Python Version**: Make sure you have Python 3.8 or higher.
2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install datasets transformers langdetect torch scikit-learn tqdm beautifulsoup4 matplotlib gradio
   ```
4. **Launch Jupyter**:
   ```bash
   jupyter notebook  # or jupyter lab
   ```
5. **Google Colab Setup**:  
   - This project is designed to run on **Google Colab**.  
   - Mount your Google Drive in Colab by running:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Place your dataset files in your Drive (e.g., under `MyDrive/job_data_files/`) and create a folder named `GPT2` in your Drive root to store model checkpoints (`.pth` files).  
   - By default, the notebooks save and load `.pth` files from `/content/drive/MyDrive/GPT2/`. You can edit the path in each notebook to point to a different Drive folder if desired.
   
---

## Model Training

For each task, open the corresponding notebook and run **all cells** in order:

1. **Load & preprocess** the dataset (clean HTML, detect language, etc.).
2. **Tokenize** text using the GPT-2 tokenizer.
3. **Fine-tune** the GPT-2 model on the labeled data.
4. **Validate** on a held-out split (accuracy, loss curves).
5. **Save** the fine-tuned model to disk (default path in notebook).

Repeat for each of:

- Salary: `fine-tuned_GPT2_salary.ipynb`
- Seniority: `fine-tuned_GPT2_seniority.ipynb`
- Work Arrangement: `fine-tuned_GPT2_work_arrangements.ipynb`

---

## Running the Gradio Apps

To interact with a trained model via a web UI:

1. Open the matching Gradio notebook, e.g. `Gradio_Salary.ipynb`.
2. Verify the model path points to your fine-tuned weights.
3. Run all cells. The last cell will launch a local server and display a URL (e.g., `http://127.0.0.1:7860`).
4. Navigate to the URL in your browser, enter a job description, and submit to see predictions.

_If ports collide when running multiple demos, notebooks will auto-increment ports (7861, 7862, ...)._

---

## Project Structure

- [fine-tuned_GPT2_salary.ipynb](https://colab.research.google.com/drive/1arrqlMhOUYFHv4slThrC0M_jLgUfA44C?usp=sharing)
- [fine-tuned_GPT2_seniority.ipynb](https://colab.research.google.com/drive/1gUevtTXpfBCaYjhuEGRMtQb27J49Hxcw?usp=sharing)
- [fine-tuned_GPT2_work_arrangements.ipynb](https://colab.research.google.com/drive/182ywgU2KC_2oT7iTczG4_BXVuEhubf4N?usp=sharing)
- [Gradio_Salary.ipynb](https://colab.research.google.com/drive/1r2u9iSXxn4ttsMD5EKXItH_CeV4waoTe?usp=sharing)
- [Gradio_Seniority.ipynb](https://colab.research.google.com/drive/1fw8wPMpSskAynETpttz3mEuVoe6JqZ93?usp=sharing)
- [Gradio_Work_Arrangements.ipynb](https://colab.research.google.com/drive/1zkVuPibNwYg8hz_QtjS5Kumlj6YrqnmN?usp=sharing)

---

## Technologies Used

- **Python 3.8+**
- **Jupyter Notebook**
- **PyTorch**
- **Hugging Face Transformers & Datasets**
- **scikit-learn**
- **langdetect**
- **BeautifulSoup4**
- **Matplotlib**
- **Gradio**

---

## Acknowledgments

- **OpenAI GPT-2**: Pre-trained model backbone.
- **Hugging Face**: `transformers` & `datasets` libraries.
- **Gradio**: Simplified web-app interfaces.
- **Open-source community**: Tutorials and example code that guided this implementation.


## Acknowledgments

- **OpenAI GPT-2**: Pre-trained model backbone.
- **Hugging Face**: `transformers` & `datasets` libraries.
- **Gradio**: Simplified web-app interfaces.
- **Open-source community**: Tutorials and example code that guided this implementation.
