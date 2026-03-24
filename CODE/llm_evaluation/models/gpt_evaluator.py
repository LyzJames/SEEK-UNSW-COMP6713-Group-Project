import os
import time
from openai import OpenAI
from models.base_evaluator import LLMEvaluator
import json
from decimal import Decimal, ROUND_HALF_UP

class GptEvaluator(LLMEvaluator):
    """GPT-specific implementation of the LLM evaluator."""
    
    def __init__(self, model_variant="gpt-4o-mini-2024-07-18", task_name="work_arrangement"):
        """
        Initialize the GPT evaluator.
        Args:
            model_variant (str): model variant (such as gpt-4o-mini-2024-07-18)
            task_name (str): Task name (work_arrangement, salary, seniority)
        """
        super().__init__(model_variant, task_name) # call the parent class constructor  
        
        # Initialize GPT client  
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        
        # OpenAI API pricing per 1M tokens (https://openai.com/api/pricing/)

        self.pricing = {
            "gpt-4.1": {"input": 2, "output": 8},   # $2 per 1M input, $8 per 1M output
            "gpt-4.1-mini": {"input": 0.4, "output": 1.6},   # $0.4 per 1M input, $1.6 per 1M output
            "gpt-4o": {"input": 2.5, "output": 10},   # $2.5 per 1M input, $10 per 1M output
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},   # $0.15 per 1M input, $0.60 per 1M output
        }

        # Extract base model for pricing
        if "gpt-4.1-mini" in model_variant:
            self.base_model = "gpt-4.1-mini"
        elif "gpt-4.1" in model_variant:
            self.base_model = "gpt-4.1"
        elif "gpt-4o-mini" in model_variant:
            self.base_model = "gpt-4o-mini"
        elif "gpt-4o" in model_variant:
            self.base_model = "gpt-4o"
        else:
            # Default to gpt-4o-mini pricing if model not recognized
            self.base_model = "gpt-4o-mini"
    
    def call_api(self, prompt):
        """Call the OpenAI API with the given prompt.
        Args:
            prompt (str): The prompt to send to OpenAI
            
        Returns:
            dict: Results including prediction, latency, and token counts
        """
        start_time = time.time()

        # Get the function from the specific task
        function = self.task_module.create_tools()
        
        # Call the OpenAI API, reference at https://platform.openai.com/docs/api-reference/responses
        try:
            response = self.client.responses.create(
                model=self.model_variant,
                input=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            tools=[function],
            tool_choice="required",
            temperature=0,
        )   
            
            end_time = time.time()
            
            # Extract the prediction (first line of the response)
            prediction = self.post_process_predictions(response.output[0])
            
            result = {
                "prediction": prediction,
                "latency": round(end_time - start_time, 6),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
            # Calculate cost
            result["cost"] = self.calculate_cost(
                result["input_tokens"], 
                result["output_tokens"]
            )
            
            return result
            
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            # Return default values on error
            return {
                "prediction": "ERROR",
                "latency": round(time.time() - start_time, 6),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0
            }
    
    def calculate_cost(self, input_tokens, output_tokens):
        """Calculate the cost based on Claude pricing.
        Args:
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
            
        Returns:
            float: Calculated cost in USD, rounded to 6 decimal places
        Raises:
            ValueError: If pricing information for the base model is not found.
        """
        if self.base_model in self.pricing:
            input_cost = (input_tokens / 1_000_000) * self.pricing[self.base_model]["input"]
            output_cost = (output_tokens / 1_000_000) * self.pricing[self.base_model]["output"]
            return round(input_cost + output_cost, 6)
        else:
            # If pricing for the specific base model is not found, raise an error.
            raise ValueError(f"Pricing information not found for base model: {self.base_model}") 

    def post_process_predictions(self, response):
        """Post-process the predictions.
        Args:
            predictions (str): The predictions to post-process
            
        Returns:
            str: The post-processed predictions
        """
        if self.task_name == "work_arrangement":
            prediction = json.loads(response.arguments)["work_mode"]
        elif self.task_name == "salary":
            result = json.loads(response.arguments)
            upper = str(Decimal(result["upper"]).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
            lower = str(Decimal(result["lower"]).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
            curr_code = result["curr_code"]
            time_unit = result["time_unit"]
            if upper=="0" or lower=="0" or time_unit=="None":
                prediction = "0-0-None-None"
            else:
                prediction = f"{lower}-{upper}-{curr_code}-{time_unit}"
        elif self.task_name == "seniority":
            prediction = json.loads(response.arguments)["seniority"]  
        else:
            raise ValueError(f"Task name {self.task_name} not supported")
        return prediction
