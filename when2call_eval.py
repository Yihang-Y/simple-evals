import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from . import common
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

# ==========================================
# 1. Judge Prompt (LLM-as-a-Judge)
# ==========================================
JUDGE_PROMPT = """You are an expert at classifying responses from AI models.

Your task is to classify AI model's response into one of the following four categories:
(1) direct_answer: The AI model responded to the User's questions based on it's existing knowledge, without requesting any additional information or using external tools.
(2) tool_call: The AI model decided to use a tool from the provided one's to help answer the question.
(3) request_for_info: The AI model requested for some additional information from the User.
(4) cannot_answer: The AI model refused to answer the User's questions by acknowledging the lack of required capabilities.

*You should not judge whether the AI model's response is accurate or not. Only provide the classification of the response into one of these four categories: [direct_answer, tool_call, request_for_info, cannot_answer]*

- The tools available to the AI model are given in <AVAILABLE_TOOLS> </AVAILABLE_TOOLS>
- The User's question is provided in <USER_QUESTION> </USER_QUESTION>
- The AI model's response is provided in <AI_MODEL_RESPONSE> </AI_MODEL_RESPONSE> which may or may not invlove a tool call

<AVAILABLE_TOOLS>
{}
</AVAILABLE_TOOLS>

<USER_QUESTION>
{}
</USER_QUESTION>

<AI_MODEL_RESPONSE>
{}
</AI_MODEL_RESPONSE>

Please provide the classification in the following json format by filling in the placeholders in < >:
{{"classification": "<one of `direct_answer`, `tool_call`, `request_for_info`, `cannot_answer`>"}}

Respond only in the prescribed json format with the placeholders filled in."""

class When2CallEval(Eval):
    def __init__(
        self,
        grader_model: SamplerBase,
        file_path: str,
        num_examples: int | None = None,
        n_repeats: int = 1,
    ):
        self.file_path = file_path
        self.grader_model = grader_model
        
        # Load Data
        data = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        if num_examples:
            data = data[:num_examples]
            
        self.examples = data * n_repeats
        # Define the fixed order of categories
        self.categories = ["direct_answer", "tool_call", "request_for_info", "cannot_answer"]

    def parse_tools(self, tools_raw):
        """Convert python-like tool strings into valid JSON schema dictionaries."""
        parsed_tools = [
            json.loads(
                t.replace("float", "string")
                .replace("integer", "string")
                .replace("dict", "object")
                .replace("tuple", "object")
            ) 
            for t in tools_raw
        ]
        formatted_tools = [{"type": "function", "function": t} for t in parsed_tools]
        for t in formatted_tools:
            t["function"]["name"] = t["function"]["name"].replace(".", "_")
            if "parameters" in t["function"]:
                t["function"]["parameters"]["type"] = "object"
                if "properties" in t["function"]["parameters"]:
                    for param in t["function"]["parameters"]["properties"]:
                        t["function"]["parameters"]["properties"][param]["type"] = "string"
        return formatted_tools

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            # --- 1. Prepare Inputs ---
            question = row["question"]
            correct_classification = row.get("correct_answer") 
            
            raw_tools = row.get("tools", [])
            tools_schema = self.parse_tools(raw_tools)
            tools_json_str = json.dumps(tools_schema, indent=2)
            
            # Construct Prompt for Subject Model
            subject_input_content = (
                f"You have access to the following tools:\n{tools_json_str}\n\n"
                f"User Question: {question}\n\n"
                "If you need to use a tool, output the function name and arguments in JSON format.\n"
                "/no_think"
            )
            
            # --- 2. Query Subject Model ---
            prompt_messages = [sampler._pack_message(content=subject_input_content, role="user")]
            sampler_response = sampler(prompt_messages)
            model_response_text = sampler_response.response_text.strip()
            
            # Remove thinking process
            if "</think>" in model_response_text:
                model_response_text = model_response_text.split("</think>")[-1].strip()

            # --- 3. Run Judge (LLM-as-a-Judge) ---
            formatted_judge_prompt = JUDGE_PROMPT.format(
                tools_json_str,                     # <AVAILABLE_TOOLS>
                question,                           # <USER_QUESTION>
                model_response_text                 # <AI_MODEL_RESPONSE>
            )
            
            judge_messages = [self.grader_model._pack_message(content=formatted_judge_prompt, role="user")]
            judge_output = self.grader_model(judge_messages).response_text.strip()
            
            # --- 4. Parse Judge Response ---
            clean_judge_output = judge_output.replace("```json", "").replace("```", "").strip()
            pred_classification = "unknown"
            try:
                judge_json = json.loads(clean_judge_output)
                pred_classification = judge_json.get("classification", "unknown")
            except:
                retry_messages = judge_messages + [
                    {"role": "assistant", "content": judge_output},
                    {"role": "user", "content": "Please output valid JSON only in the prescribed format."}
                ]
                retry_output = self.grader_model(retry_messages).response_text.strip()
                try:
                    clean_retry = retry_output.replace("```json", "").replace("```", "").strip()
                    judge_json = json.loads(clean_retry)
                    pred_classification = judge_json.get("classification", "unknown")
                except:
                    pred_classification = "parse_error"

            # --- 5. Basic Score ---
            is_correct = (pred_classification == correct_classification)
            score = 1.0 if is_correct else 0.0

            # HTML Visualization
            html_content = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=model_response_text, role="assistant"),
                score=score,
                correct_answer=f"GT: {correct_classification} | Tools: {len(raw_tools)}",
                extracted_answer=f"Pred: {pred_classification}"
            )

            return SingleEvalResult(
                html=html_content,
                score=score,
                convo=prompt_messages + [dict(content=model_response_text, role="assistant")],
                metrics={
                    "score": score,
                    "pred_class": pred_classification,  
                    "gt_class": correct_classification, 
                    "num_tools": len(raw_tools),
                    "is_correct": int(is_correct)
                }
            )

        # Run Eval Loop
        results = common.map_with_progress(fn, self.examples)
        
        # ==========================================
        # 6. Aggregation & Metrics Calculation
        # ==========================================
        total = len(results)
        if total == 0:
            return common.aggregate_results(results)

        # A. Vectors
        y_true = [r.metrics.get("gt_class", "unknown") for r in results]
        y_pred = [r.metrics.get("pred_class", "unknown") for r in results]

        # B. Hallucination Rate
        hallucination_subset = []
        hallucinated_calls = []
        for r in results:
            if r.metrics.get("gt_class") == "cannot_answer" and r.metrics.get("num_tools") == 0:
                hallucination_subset.append(r)
                if r.metrics.get("pred_class") == "tool_call":
                    hallucinated_calls.append(r)
        
        hallucination_rate = len(hallucinated_calls) / len(hallucination_subset) if len(hallucination_subset) > 0 else 0.0

        # C. Sklearn Metrics
        labels = self.categories
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"True:{x}" for x in labels], columns=[f"Pred:{x}" for x in labels])

        # D. Print Dashboard
        print("\n" + "="*80, flush=True)
        print("📊 WHEN2CALL EXTENDED EVALUATION REPORT", flush=True)
        print("="*80, flush=True)
        print(f"Total Samples:      {total}", flush=True)
        print(f"Overall Accuracy:   {sum(r.score for r in results)/total:.2%}", flush=True)
        print(f"Macro F1 Score:     {macro_f1:.4f}", flush=True)
        print(f"Hallucination Rate: {hallucination_rate:.2%}", flush=True)
        print("\n--- 📉 Confusion Matrix ---", flush=True)
        print(cm_df.to_string(), flush=True)
        print("\n--- 📈 Per-Category Breakdown ---", flush=True)
        print(f"{'Category':<20} | {'Prec.':<8} | {'Recall':<8} | {'F1':<8} | {'Support':<8}", flush=True)
        for i, cat in enumerate(labels):
            print(f"{cat:<20} | {precision[i]:<8.2f} | {recall[i]:<8.2f} | {f1[i]:<8.2f} | {support[i]:<8}", flush=True)
        print("="*80 + "\n", flush=True)

        # E. Clean up Strings
        for r in results:
            r.metrics.pop("pred_class", None)
            r.metrics.pop("gt_class", None)

        # F. Inject Globals and Detailed Metrics
        final_result_obj = common.aggregate_results(results)
        if final_result_obj.metrics is None: final_result_obj.metrics = {}

        # 1. 全局指标
        final_result_obj.metrics["macro_f1"] = macro_f1
        final_result_obj.metrics["hallucination_rate"] = hallucination_rate

        # 2. 分类详细指标
        for i, cat in enumerate(labels):
            # 原程序中的 "score" = Recall (该类别的准确率)
            final_result_obj.metrics[f"recall_{cat}"] = recall[i]  
            final_result_obj.metrics[f"f1_{cat}"] = f1[i]
            
            # 原程序中的 "response_distribution" (展平为 count_GT_x_PRED_y)
            # 只有数量 > 0 的才记录，防止 JSON 太长
            row_vals = cm[i]
            for j, pred_cat in enumerate(labels):
                count = int(row_vals[j])
                if count > 0:
                    key = f"count_true_{cat}_pred_{pred_cat}"
                    final_result_obj.metrics[key] = count

        return final_result_obj