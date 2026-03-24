"""
Configuration settings for LLM evaluation.
"""

# Define tasks and their corresponding dataset paths
TASKS_DATAPATH = {    
    # "work_arrangement": "../../MISC/job_data_files/work_arrangements_development_set.csv",
    "work_arrangement": "../../MISC/job_data_files/work_arrangements_test_set.csv",
    # "work_arrangement": "../../MISC/label_unlabelled_data/lable_work_arrangement.csv",
    # "salary": "../../MISC/job_data_files/salary_labelled_development_set.csv", 
    "salary": "../../MISC/job_data_files/salary_labelled_test_set.csv",   
    # "seniority": "../../MISC/seniority_data_mapped_new/seniority_labelled_development_set_mapped.csv",
    "seniority": "../../MISC/seniority_data_mapped_new/seniority_labelled_test_set_mapped.csv",
    # "seniority": "../../MISC/seniority_data_mapped/seniority_labelled_test_set_mapped.csv"
}

# Define model variants available for testing
MODELS_VARIANTS = {
    # "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    # "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
    "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
    "llama3.1-8b": "llama3.1-8b",
    "llama3.1-70b": "llama3.1-70b",
    "llama4-maverick": "llama4-maverick",
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gemini-2.5-flash-preview": "gemini-2.5-flash-preview",
    "gemma-3-4b": "gemma-3-4b-it",
    "gemma-3-27b": "gemma-3-27b-it"
}

# Define model configurations
MODELS = {
    "claude": {
        "class": "ClaudeEvaluator",
        "name": "claude-3-7-sonnet-20250219",  # Default model
        "parameters": {}
    },
    "llama": {
        "class": "LlamaEvaluator",
        "name": "llama3.1-70b",
        "parameters": {}
    },
    "deepseek": {
        "class": "DeepSeekEvaluator",
        "name": "deepseek-chat",
        "parameters": {}
    },
    "gpt": {
        "class": "GptEvaluator",
        "name": "gpt-4o-mini",
        "parameters": {}
    },
    "gemini": {
        "class": "GeminiEvaluator",
        "name": "gemini-2.5-flash-preview",
        "parameters": {}
    },
}

# MLflow configuration
MLFLOW_CONFIG = {
    "tracking_uri": None,
    "experiment_prefix": "llm_eval"
} 