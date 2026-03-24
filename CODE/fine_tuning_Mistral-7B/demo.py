import os
import re

import gradio as gr
import numpy as np
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common import SAVE_MODEL_PATH, MissionType, JobData

np.random.seed(42)


class MistralDemo:
    def __init__(self, device: torch.device = None):
        self.seed = 42
        self.base_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        self.device = device if device else torch.device("cpu")

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = None
        self.tokenizer = None

        self.wa_data = JobData(MissionType.WA)
        self.wa_model_path = os.path.join(SAVE_MODEL_PATH, "mistral-7b-lora-WA")
        self.sa_data = JobData(MissionType.SA)
        self.sa_model_path = os.path.join(SAVE_MODEL_PATH, "mistral-7b-lora-SA")
        self.se_data = JobData(MissionType.SE)
        self.se_model_path = os.path.join(SAVE_MODEL_PATH, "mistral-7b-lora-SE")

        # Create a Blocks app with three tabs
        self.demo = gr.Blocks()
        with self.demo:
            self.model_state = gr.State(value=None)
            with gr.Tabs():
                with gr.TabItem("About") as about_tab:
                    gr.Markdown("### Fine-tuning Mistral-7B-Instruct-v0.3")
                    gr.Markdown(
                        "This demo showcases the fine-tuning of the Mistral-7B-Instruct-v0.3 model for three tasks: Work Arrangement, Salary, and Seniority prediction."
                    )
                    gr.Markdown(
                        "Load Sample Button: loads a sample input for each task."
                    )

                with gr.TabItem("Work Arrangement") as wa_tab:
                    gr.Markdown("### Work Arrangement Prediction Task")
                    self.job_ad = gr.Textbox(label="Job Advertisement Details")

                    self.wa_load_sample_button = gr.Button("Load Sample Input")
                    self.wa_load_sample_button.click(
                        fn=self.__load_wa_sample_input,
                        inputs=None,
                        outputs=self.job_ad,
                    )

                    self.wa_output = gr.Textbox(label="Predicted Work Arrangement")
                    self.wa_predict_button = gr.Button("Predict Work Arrangement")

                with gr.TabItem("Salary") as sa_tab:
                    gr.Markdown("### Salary Prediction Task")
                    self.job_title = gr.Textbox(label="Job Title")
                    self.job_ad_details = gr.Textbox(label="Job Ad Details")
                    self.nation_short_desc = gr.Textbox(
                        label="Nation Short Description"
                    )
                    self.salary_additional_text = gr.Textbox(
                        label="Salary Additional Text"
                    )

                    self.sa_load_sample_button = gr.Button("Load Sample Input")
                    self.sa_load_sample_button.click(
                        fn=self.__load_sa_sample_input,
                        inputs=None,
                        outputs=[
                            self.job_title,
                            self.job_ad_details,
                            self.nation_short_desc,
                            self.salary_additional_text,
                        ],
                    )

                    self.sa_output = gr.Textbox(label="Predicted Salary")
                    self.sa_predict_button = gr.Button("Predict Salary")

                with gr.TabItem("Seniority") as se_tab:
                    gr.Markdown("### Seniority Prediction Task")
                    self.job_title = gr.Textbox(label="Job Title")
                    self.job_summary = gr.Textbox(label="Job Summary")
                    self.job_ad_details = gr.Textbox(label="Job Ad Details")
                    self.classification_name = gr.Textbox(label="Classification Name")
                    self.subclassification_name = gr.Textbox(
                        label="Subclassification Name"
                    )

                    self.se_load_sample_button = gr.Button("Load Sample Input")
                    self.se_load_sample_button.click(
                        fn=self.__load_se_sample_input,
                        inputs=None,
                        outputs=[
                            self.job_title,
                            self.job_summary,
                            self.job_ad_details,
                            self.classification_name,
                            self.subclassification_name,
                        ],
                    )

                    self.se_output = gr.Textbox(label="Predicted Seniority")
                    self.se_predict_button = gr.Button("Predict Seniority")

                wa_tab.select(
                    fn=self.__load_wa_model, inputs=None, outputs=self.model_state
                )
                self.wa_predict_button.click(
                    fn=self.__predict_work_arrangement,
                    inputs=[self.job_ad],
                    outputs=self.wa_output,
                )

                sa_tab.select(
                    fn=self.__load_sa_model, inputs=None, outputs=self.model_state
                )
                self.sa_predict_button.click(
                    fn=self.__predict_salary,
                    inputs=[
                        self.job_title,
                        self.job_ad_details,
                        self.nation_short_desc,
                        self.salary_additional_text,
                    ],
                    outputs=self.sa_output,
                )

                se_tab.select(
                    fn=self.__load_se_model, inputs=None, outputs=self.model_state
                )
                self.se_predict_button.click(
                    fn=self.__predict_seniority,
                    inputs=[
                        self.job_title,
                        self.job_summary,
                        self.job_ad_details,
                        self.classification_name,
                        self.subclassification_name,
                    ],
                    outputs=self.se_output,
                )
        self.demo.queue()

    def __load_sa_model(self) -> gr.State:
        self.release(self.model, self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.sa_model_path,
            quantization_config=self.bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.sa_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.model_state

    def __load_sa_sample_input(self) -> str:
        sample = self.sa_data.df_test.sample(1)
        job_title = sample["job_title"].values[0]
        job_ad_details = sample["job_ad_details"].values[0]
        nation_short_desc = sample["nation_short_desc"].values[0]
        salary_additional_text = sample["salary_additional_text"].values[0]
        return job_title, job_ad_details, nation_short_desc, salary_additional_text

    def __load_se_model(self):
        self.release(self.model, self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.se_model_path,
            quantization_config=self.bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        # self.tokenizer = AutoTokenizer.from_pretrained(self.se_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __load_se_sample_input(self) -> str:
        sample = self.se_data.df_test.sample(1)
        job_title = sample["job_title"].values[0]
        job_summary = sample["job_summary"].values[0]
        job_ad_details = sample["job_ad_details"].values[0]
        classification_name = sample["classification_name"].values[0]
        subclassification_name = sample["subclassification_name"].values[0]
        return (
            job_title,
            job_summary,
            job_ad_details,
            classification_name,
            subclassification_name,
        )

    def __load_wa_model(self):
        self.release(self.model, self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.wa_model_path,
            quantization_config=self.bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.wa_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __load_wa_sample_input(self) -> str:
        sample = self.wa_data.df_test.sample(1)
        job_ad = sample["job_ad"].values[0]
        return job_ad

    def __predict_salary(
        self,
        job_title: str,
        job_ad_details: str,
        nation_short_desc: str,
        salary_additional_text: str,
    ) -> str:
        """
        Predicts the salary for a given job title and job ad details using the Mistral-7B model.

        Args:
            job_title (str): The title of the job.
            job_ad_details (str): The details of the job advertisement.
            nation_short_desc (str): A short description of the nation.
            salary_additional_text (str): Additional text related to salary.

        Returns:
            str: The predicted salary.
        """
        prompt = self.format_prompt_jsonl_sa(
            job_title,
            job_ad_details,
            nation_short_desc,
            salary_additional_text,
        )
        return self.__predict(prompt)

    def __predict_seniority(
        self,
        job_title: str,
        job_summary: str,
        job_ad_details: str,
        classification_name: str,
        subclassification_name: str,
    ) -> str:
        """
        Predicts the seniority level for a given job title and job ad details using the Mistral-7B-Instruct-v0.3 model.

        Args:
            job_title (str): The title of the job.
            job_summary (str): A summary of the job.
            job_ad_details (str): The details of the job advertisement.
            classification_name (str): The classification name.
            subclassification_name (str): The subclassification name.

        Returns:
            str: The predicted seniority level.
        """
        prompt = self.format_prompt_jsonl_se(
            job_title,
            job_summary,
            job_ad_details,
            classification_name,
            subclassification_name,
        )
        return self.__predict(prompt)

    def __predict_work_arrangement(
        self,
        job_ad: str,
    ) -> str:
        """
        Predicts the work arrangement for a given job advertisement using the Mistral-7B-Instruct-v0.3 model.

        Args:
            job_ad (str): The job advertisement.

        Returns:
            str: The predicted work arrangement.
        """
        prompt = self.format_prompt_jsonl_wa(job_ad)
        return self.__predict(prompt)

    def __find_answer(self, text: str) -> str:
        return (
            re.search(r"Answer: (.*)", text).group(1).strip()
            if re.search(r"Answer: (.*)", text)
            else "None"
        )

    def format_prompt_jsonl_wa(self, job_ad: str) -> str:
        prompt_jsonl = {
            "instruction": (
                "You are a helpful assistant."
                "Your task is to classify which type of work arrangements the job advertisement belongs to."
                "There are three types: ['OnSite', 'Remote', 'Hybrid']."
            ),
            "input": job_ad,
        }
        return (
            f"<s>[INST] {prompt_jsonl['instruction']} {prompt_jsonl['input']} [/INST]"
        )

    def format_prompt_jsonl_sa(
        self,
        job_title: str,
        job_ad_details: str,
        nation_short_desc: str,
        salary_additional_text: str,
    ) -> str:
        prompt_jsonl = {
            "instruction": (
                "You are a helpful assistant."
                "Your task is to classify the salary level of the job advertisement. "
                "The format will be [NUMBER]-[NUMBER]-[CURRENCY SYMBOL]-[HOURLY|DAILY|MONTHLY|YEARLY|ANNUAL]."
            ),
            "input": (  # job_title job_ad_details nation_short_desc salary_additional_text
                f"The job title is: {job_title}."
                f"Further details: {job_ad_details}."
                f"Country Codes: {nation_short_desc}."
                f"Salary information: {salary_additional_text}."
            ),
        }
        return (
            f"<s>[INST] {prompt_jsonl['instruction']} {prompt_jsonl['input']} [/INST]"
        )

    def format_prompt_jsonl_se(
        self,
        job_title: str,
        job_summary: str,
        job_ad_details: str,
        classification_name: str,
        subclassification_name: str,
    ) -> str:
        prompt_jsonl = {
            "instruction": (
                "You are a helpful assistant."
                "Your task is to classify the seniority level of the job advertisement."
                "The format will be [SENIORITY LEVEL]."
            ),
            "input": (
                f"The job title is: {job_title}."
                f"To summarize the job: {job_summary}."
                f"Further details: {job_ad_details}"
                f"The classification is: {classification_name} / {subclassification_name}."
            ),
        }
        return (
            f"<s>[INST] {prompt_jsonl['instruction']} {prompt_jsonl['input']} [/INST]"
        )

    def __predict(self, prompt: str) -> str:
        """
        Predicts the output for a given prompt using the Mistral-7B-Instruct-v0.3 model.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The predicted output.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.device)

        attention_mask = self.tokenizer(prompt, return_tensors="pt").attention_mask
        attention_mask = attention_mask.to(self.device)

        self.model.gradient_checkpointing_enable()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=1,
        )

        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self.__find_answer(output_text)

    def release(self, model, tokenizer):
        """
        Release the model and tokenizer from memory.
        """
        try:
            if tokenizer is not None:
                # Delete tokenizer
                del tokenizer
            if model is not None:
                # Move model to CPU and delete
                model.cpu()
                del model
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: failed to unload model: {e}")

    def launch(self, **kwargs):
        self.demo.launch(**kwargs)


if __name__ == "__main__":
    print(f"torch version: {torch.__version__}")
    print(f"torch cuda version: {torch.version.cuda}")
    device = torch.device(
        torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    )
    print(f"torch device: {device}")
    print(
        f"device name: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'}"
    )
    mistral_demo = MistralDemo(device=device)
    mistral_demo.launch(share=True, debug=True)
