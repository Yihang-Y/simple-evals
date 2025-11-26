import random
import re
import pandas
from . import common
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

# ==========================================
# 0. 被测模型 Prompt
# ==========================================
CLAMBER_SYSTEM_PROMPT = """You are a helpful assistant.
Your task is to analyze the User Query and decide if it is ambiguous.

Format your response exactly as follows:
1. Start with a decision tag: "[DECISION]: CLARIFY" (if ambiguous) or "[DECISION]: DIRECT" (if clear).
2. Then provide your content (the clarifying question OR the direct answer) on a new line.

Examples:
User: "Where is the best place to eat?"
Assistant: [DECISION]: CLARIFY
What type of cuisine are you looking for?

User: "What is 2+2?"
Assistant: [DECISION]: DIRECT
The answer is 4.
"""

# ==========================================
# 1. 评分 Prompt
# ==========================================
GRADER_TEMPLATE_CONTENT_ONLY = """
The AI assistant has correctly identified the type of response needed. Now evaluate the QUALITY of its content.

Scenario: {scenario_description}

User Query: {question}
Gold Target: {target}
Model Response: {predicted_answer}

Evaluation Criteria:
- If Clarification was needed: Is the model's question semantically similar to the Gold Target? Is it relevant?
- If Direct Answer was needed: Is the model's answer helpful and relevant? (Ignore factuality if Gold Target is N/A, just check if it's a direct answer).

Reply with:
A: CORRECT (Good quality)
B: INCORRECT (Bad quality / Irrelevant / Hallucinated)

Just return the letter "A" or "B".
""".strip()

class ClamberEval(Eval):
    def __init__(
        self,
        grader_model: SamplerBase,
        num_examples: int | None = None,
        n_repeats: int = 1,
        file_path: str | None = None,
    ):
        if not file_path:
            raise ValueError("file_path is required")
        
        df = pandas.read_json(file_path, lines=True)
        examples = [row.to_dict() for _, row in df.iterrows()]
        
        if num_examples:
            rng = random.Random(0)
            sample_size = min(num_examples, len(examples))
            examples = rng.sample(examples, sample_size)
            
        self.examples = examples * n_repeats
        self.grader_model = grader_model

    def grade_content(self, question, target, predicted_answer, require_clarification):
        """仅评估内容质量，不评估分类"""
        if require_clarification == 1:
            scenario = "Ambiguous Query -> Clarifying Question Required"
            if not target: target = "(Any reasonable clarifying question)"
        else:
            scenario = "Clear Query -> Direct Answer Required"
            target = "N/A (Direct Answer)"

        grader_prompt = GRADER_TEMPLATE_CONTENT_ONLY.format(
            scenario_description=scenario,
            question=question,
            target=target,
            predicted_answer=predicted_answer
        )

        prompt_messages = [self.grader_model._pack_message(content=grader_prompt, role="user")]
        sampler_response = self.grader_model(prompt_messages)
        grading_response = sampler_response.response_text

        # 只要不是 B 都算对 (容错 A, Correct, Yes 等)
        if "B" in grading_response and "A" not in grading_response:
            return "INCORRECT"
        return "CORRECT"

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            # --- 1. 数据解析 ---
            problem = row.get("question") or row.get("problem") or ""
            
            # 获取 Ground Truth (0=Direct, 1=Clarify)
            try:
                gt_require = int(row.get("require_clarification", 0))
            except:
                gt_require = 0

            # 获取 Gold Text
            gold_target = row.get("clarifying_question") or row.get("answer") or ""
            if isinstance(gold_target, list): gold_target = gold_target[0]

            # --- 2. 模型推理 ---
            full_prompt = f"{CLAMBER_SYSTEM_PROMPT}\n\nUser: {problem}\nAssistant:/no_think" # 可以在这直接把 think 干掉
            prompt_messages = [sampler._pack_message(content=full_prompt, role="user")]
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            
            # 去除 </think>
            if "</think>" in response_text:
                full_response = response_text.split("</think>")[-1].strip()
            else:
                full_response = response_text.strip()

            # --- 3. 结果解析 ---
            match = re.search(r"\[DECISION\]:\s*(CLARIFY|DIRECT)", full_response, re.IGNORECASE)
            
            is_format_error = False
            pred_require = -1
            decision_str = "PARSE_ERROR"
            content_only = full_response

            if match:
                decision_str = match.group(1).upper()
                content_only = full_response.replace(match.group(0), "").strip()
                pred_require = 1 if decision_str == "CLARIFY" else 0
            else:
                is_format_error = True

            # --- 4. 详细指标计算 ---
            
            # 初始化状态
            is_class_correct = False
            is_content_correct = False # 仅当分类正确且内容好时为 True
            grade_result = "SKIPPED"

            # 4.1 检查分类
            if not is_format_error and pred_require == gt_require:
                is_class_correct = True
                
                # 4.2 检查内容 (分类对才进 Judge)
                grade_result = self.grade_content(problem, gold_target, content_only, gt_require)
                if grade_result == "CORRECT":
                    is_content_correct = True
            
            # 4.3 最终得分 (0/1)
            score = 1.0 if is_content_correct else 0.0

            # 构造状态消息 (用于 HTML 展示)
            if is_format_error:
                status_msg = "⚠️ Format Error"
            elif not is_class_correct:
                status_msg = f"❌ Class Wrong (GT:{gt_require} vs Pred:{pred_require})"
            elif not is_content_correct:
                status_msg = f"⚠️ Class OK, Content Bad"
            else:
                status_msg = "✅ Perfect"

            # --- 5. 返回结果 (携带详细 Metrics) ---
            return SingleEvalResult(
                html=common.jinja_env.from_string(common.HTML_JINJA).render(
                    prompt_messages=sampler_response.actual_queried_message_list,
                    next_message=dict(content=response_text, role="assistant"),
                    score=score,
                    correct_answer=f"GT: {gt_require} | Gold: {gold_target}",
                    extracted_answer=f"{status_msg}\nDecision: {decision_str}\nContent: {content_only}",
                ),
                score=score,
                convo=sampler_response.actual_queried_message_list + [dict(content=response_text, role="assistant")],
                metrics={
                    "score": score,  # 总分 (分类+内容都对)
                    "is_format_valid": not is_format_error, # 格式是否正确
                    "is_class_correct": is_class_correct,   # 分类是否正确
                    "is_content_correct": is_content_correct, # 内容是否正确
                    
                    # 细分统计标志位
                    "gt_clarify": gt_require == 1,
                    "gt_direct": gt_require == 0,
                    "pred_clarify": pred_require == 1,
                    "pred_direct": pred_require == 0,
                }
            )

        # 执行评测
        results = common.map_with_progress(fn, self.examples)

        # ==========================================
        # 6. 聚合统计 (Dashboard)
        # ==========================================
        total = len(results)
        if total == 0: return common.aggregate_results(results)

        # 辅助函数：安全除法
        def safe_div(n, d): return n / d if d > 0 else 0.0

        # 提取 metrics 列表
        m = [r.metrics for r in results]

        # 1. 基础统计
        count_gt_clarify = sum(1 for x in m if x['gt_clarify'])
        count_gt_direct = sum(1 for x in m if x['gt_direct'])
        count_format_valid = sum(1 for x in m if x['is_format_valid'])

        # 2. 分类准确率 (Classification Metrics)
        # 只要分类对了就算对，不管内容好坏
        count_class_correct = sum(1 for x in m if x['is_class_correct'])
        
        # 针对 Need Clarify 的召回率 (Recall): GT=1 中，模型预测出 1 的比例
        # 注意：这里要求分类正确即可
        correct_clarify_class = sum(1 for x in m if x['gt_clarify'] and x['is_class_correct'])
        recall_clarify = safe_div(correct_clarify_class, count_gt_clarify)

        # 针对 Direct Answer 的特异度 (Specificity): GT=0 中，模型预测出 0 的比例
        correct_direct_class = sum(1 for x in m if x['gt_direct'] and x['is_class_correct'])
        recall_direct = safe_div(correct_direct_class, count_gt_direct)

        # 3. 内容生成质量 (Content Quality)
        # 条件概率：在分类正确的前提下，内容写得好的概率
        
        # 对于 Clarify 类：分类正确且问题提得好
        perfect_clarify = sum(1 for x in m if x['gt_clarify'] and x['is_content_correct'])
        quality_clarify_conditional = safe_div(perfect_clarify, correct_clarify_class)

        # 对于 Direct 类：分类正确且回答得好
        perfect_direct = sum(1 for x in m if x['gt_direct'] and x['is_content_correct'])
        quality_direct_conditional = safe_div(perfect_direct, correct_direct_class)

        # 4. 总体正确率 (End-to-End Accuracy)
        overall_acc = sum(r.score for r in results) / total

        # --- 打印报表 ---
        print("\n" + "="*50)
        print("📊 CLAMBER EVALUATION DASHBOARD")
        print("="*50)
        print(f"Total Samples: {total}")
        print(f"Format Adherence: {safe_div(count_format_valid, total):.2%} ({count_format_valid}/{total})")
        
        print("\n--- 🎯 Classification Performance (Task Understanding) ---")
        print(f"Overall Class Acc:  {safe_div(count_class_correct, total):.2%}")
        print(f"Recall (Need Clarify): {recall_clarify:.2%} ({correct_clarify_class}/{count_gt_clarify})")
        print(f"Recall (Need Direct):  {recall_direct:.2%} ({correct_direct_class}/{count_gt_direct})")

        print("\n--- ✍️  Generation Quality (Conditioned on Correct Class) ---")
        print(f"Clarify Q Quality:  {quality_clarify_conditional:.2%} (Is the question good?)")
        print(f"Direct Ans Quality: {quality_direct_conditional:.2%} (Is the answer helpful?)")

        print("\n--- 🏆 Overall End-to-End Metrics ---")
        print(f"Overall Accuracy:   {overall_acc:.2%} (Class + Content both correct)")
        print("="*50 + "\n")

        return common.aggregate_results(results)