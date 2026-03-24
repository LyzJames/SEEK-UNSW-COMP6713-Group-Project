import os
import time
from openai import OpenAI
from models.base_evaluator import LLMEvaluator
import json
from decimal import Decimal, ROUND_HALF_UP

class GeminiEvaluator(LLMEvaluator):
    """Gemini-specific implementation of the LLM evaluator."""
    
    def __init__(self, model_variant="gemini-2.5-flash-preview", task_name="work_arrangement"):
        """
        Initialize the Gemini evaluator.
        Args:
            model_variant (str): model variant (such as gemini-2.5-flash-preview)
            task_name (str): Task name (work_arrangement, salary, seniority)
        """
        super().__init__(model_variant, task_name) # call the parent class constructor  
        
        # Initialize OpenRouter client  
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        
        # Google pricing per 1M tokens (https://ai.google.dev/gemini-api/docs/pricing)

        self.pricing = {
            "gemini-2.5-flash-preview": {"input": 0.15, "output": 0.6},   # $0.15 per 1M input, $0.60 per 1M output
            "gemma-3-4b": {"input": 0.02, "output": 0.04},   # free model
            "gemma-3-27b": {"input": 0.1, "output": 0.2},   # free model
        }

        # Extract base model for pricing
        if "gemini-2.5-flash-preview" in model_variant:
            self.base_model = "gemini-2.5-flash-preview"
        elif "gemma-3-4b" in model_variant:
            self.base_model = "gemma-3-4b"
        elif "gemma-3-27b" in model_variant:
            self.base_model = "gemma-3-27b"
    
    def call_api(self, prompt):
        """Call the Gemini API with the given prompt.
        Args:
            prompt (str): The prompt to send to Gemini
            
        Returns:
            dict: Results including prediction, latency, and token counts
        """
        start_time = time.time()
        
        # Call the OpenRouter API with OpenAI SDK, reference at https://openrouter.ai/docs/api-reference/overview
        try:
            response = self.client.chat.completions.create(
                extra_body={"usage": True},
                model="google/" + self.model_variant,
                messages=[
                    {
                        "role": "user",
                        "content": self.preprocessing(prompt),
                    }
                ],
                temperature=0,
            )
            end_time = time.time()
            
            # Extract the prediction (first line of the response)
            prediction = response.choices[0].message.content
            
            result = {
                "prediction": prediction,
                "latency": round(end_time - start_time, 6),
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.prompt_tokens + response.usage.completion_tokens
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
        """Calculate the cost based on hardcoded pricing.
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
