# LLM Evaluation Framework

Framework for evaluating Large Language Models (LLMs) on job-related tasks including Accuracy, Speed and Cost.

## Overview

This framework evaluates LLMs (like Claude, LLAMA, etc.) on tasks such as:
- Work arrangement classification (Remote/Hybrid/Onsite)
- Salary information extraction
- Job seniority classification (Entry/Mid/Senior/Executive)

Results are tracked with MLflow for easy comparison and visualization.

## Folder Structure

```
llm_evaluation/
├── models/              # Model-specific implementations
│   ├── base_evaluator.py  # Base evaluator class
│   ├── claude_evaluator.py  # Claude implementation
│   ├── gpt_evaluator.py    # GPT implementation
│   ├── gemini_evaluator.py # Gemini implementation
│   ├── llama_evaluator.py  # LLaMA implementation
│   ├── deepseek_evaluator.py # DeepSeek implementation
│   └── __init__.py         # Package initialization
├── tasks/               # Task-specific implementations
│   ├── work_arrangement.py  # Work arrangement classification
│   ├── salary.py        # Salary extraction
│   ├── seniority.py     # Seniority classification
│   └── __init__.py      # Package initialization
├── prompts/             # Prompt templates
│   ├── templates.py     # Templates for all tasks
│   └── __init__.py      # Package initialization
├── config.py            # Configuration (tasks, models, paths)
└── run_experiments.py   # Main runner script
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: `pip install mlflow anthropic pandas scikit-learn matplotlib numpy openai ftfy emoji unicodedata`
- API keys for LLMs set as environment variables:
  - `ANTHROPIC_API_KEY` for Claude models
  - `OPENAI_API_KEY` for GPT models
  - `OPENROUTER_API_KEY` for Gemini models

### Running an Evaluation

1. Set required API keys:
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```

2. Run an experiment:
   run in directory CODE/llm_evaluation
   ```bash
   python run_experiments.py --model claude --task work_arrangement --limit 10
   ```

   Options:
   - `--model`: Model to use (claude, llama, etc.)
   - `--task`: Task to evaluate (work_arrangement, salary, seniority)
   - `--limit`: Limit the number of examples (optional)

3. View and compare results in MLflow UI:
   ```bash
   mlflow ui
   ```
   Then open http://localhost:5000 in your browser.

## Experimenting with Different Claude Models

The framework supports experimenting with different Claude model variants:

1. Run a specific Claude model variant:
   ```bash
   python run_experiments.py --model claude --claude-variant claude-3-opus --task work_arrangement --limit 10
   ```

2. Available Claude variants:
   - `claude-3-opus`: Claude 3 Opus (most capable)
   - `claude-3-sonnet`: Claude 3 Sonnet (balanced)
   - `claude-3-haiku`: Claude 3 Haiku (fastest)
   - `claude-3.5-sonnet`: Claude 3.5 Sonnet
   - `claude-3.7-sonnet`: Claude 3.7 Sonnet (latest version)

3. Compare models for cost and performance analysis:
   ```bash
   # Compare accuracy & cost of Claude 3 Sonnet vs. Haiku
   # Run the first model
   python run_experiments.py --model claude --claude-variant claude-3-sonnet --task salary --limit 20
   
   # Then run the second model
   python run_experiments.py --model claude --claude-variant claude-3-haiku --task salary --limit 20
   ```
   Then view the comparison in MLflow UI.

## Experimenting with Different GPT Models

The framework supports experimenting with different GPT model variants:

1. Run a specific GPT model variant:
   ```bash
   python run_experiments.py --model gpt --model-variant gpt-4o --task work_arrangement --limit 10
   ```

2. Available GPT variants:
   - `gpt-4.1`: GPT-4.1 (most capable)
   - `gpt-4.1-mini`: GPT-4.1 Mini (balanced capability and cost)
   - `gpt-4o`: GPT-4o (high capability)
   - `gpt-4o-mini`: GPT-4o Mini (fastest and most cost-effective)

## Experimenting with Different Gemini Models

The framework supports experimenting with different Gemini model variants:

1. Run a specific Gemini model variant:
   ```bash
   python run_experiments.py --model gemini --model-variant gemini-2.5-flash-preview --task work_arrangement --limit 10
   ```

2. Available Gemini variants:
   - `gemini-2.5-flash-preview`: Gemini 2.5 Flash (optimized for quick responses)
   - `gemma-3-4b`: Gemma 3 4B (compact model size)
   - `gemma-3-27b`: Gemma 3 27B (larger, more capable model)

3. Compare models for cost and performance analysis:
   ```bash
   # Compare accuracy & cost of Gemini vs. Gemma models
   # Run the first model
   python run_experiments.py --model gemini --model-variant gemini-2.5-flash-preview --task salary --limit 20
   
   # Then run the second model
   python run_experiments.py --model gemini --model-variant gemma-3-27b --task salary --limit 20
   ```
   Then view the comparison in MLflow UI.
