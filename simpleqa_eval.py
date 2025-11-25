"""
SimpleQA: Measuring short-form factuality in large language models
Authors: Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, William Fedus
https://cdn.openai.com/papers/simpleqa.pdf
"""

import random
import re
import pandas
from . import common
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()


CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))


class SimpleQAEval(Eval):
    def __init__(
        self,
        grader_model: SamplerBase,
        num_examples: int | None = None,
        n_repeats: int = 1,
    ):
        # df = pandas.read_csv(
        #     "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
        # )
        df = pandas.read_json("./simple-evals/data/NQ-open.train.jsonl", lines=True)
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when max_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        self.examples = examples * n_repeats
        self.grader_model = grader_model

    def grade_sample(self, question: str, target: str, predicted_answer: str) -> str:
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )

        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user")
        ]
        sampler_response = self.grader_model(prompt_messages)
        grading_response = sampler_response.response_text

        match = re.search(r"(A|B|C)", grading_response)
        return (
            match.group(0) if match else "D"
        )  # Default to "NOT_ATTEMPTED" if no match

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            # key 可能是 problem 或者 question
            problem = row.get("problem", "")
            if problem == "":
                problem = row.get("question", "")

            answer = row.get("answer", "")
            if type(answer) != list:
                answer = answer[0]

            prompt_messages = [sampler._pack_message(content=problem, role="user")]
            # 这里模型去从 LLM 之中sample出来东西，
            # 一般是本地部署的服务
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = (
                sampler_response.actual_queried_message_list
            )

            # extract answer from response_text
            # 使用 re 提取</think>之后的内容
            pattern = r"</think>(.*)"
            match = re.search(pattern, response_text, re.DOTALL)
            extracted_answer = response_text.strip()
            grade_letter = "D"
            if match:
                extracted_answer = match.group(1).strip()
                grade_letter = self.grade_sample(problem, answer, extracted_answer)

                # Metrics based on grading response
                is_correct = grade_letter == "A"
                is_incorrect = grade_letter == "B"
                is_not_attempted = grade_letter == "C"
            else:
                # extracted_answer = response_text.strip()
                is_correct = False
                is_incorrect = True
                is_not_attempted = False

            score = is_correct

            # Create HTML for each sample result
            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=is_correct,
                correct_answer=answer,
                extracted_answer=grade_letter + "\n-----" + response_text,
            )
            convo = actual_queried_prompt_messages + [
                dict(content=response_text, role="assistant")
            ]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={
                    "is_correct": is_correct,
                    "is_incorrect": is_incorrect,
                    "is_not_attempted": is_not_attempted,
                },
            )

        # Run evaluation and collect results
        results = common.map_with_progress(fn, self.examples)

        # Aggregate metrics
        aggregate_metrics = {
            "is_correct": sum(result.metrics["is_correct"] for result in results)
            / len(results),
            "is_incorrect": sum(result.metrics["is_incorrect"] for result in results)
            / len(results),
            "is_not_attempted": sum(
                result.metrics["is_not_attempted"] for result in results
            )
            / len(results),
        }
        aggregate_metrics["is_given_attempted"] = (
            aggregate_metrics["is_correct"] + aggregate_metrics["is_incorrect"]
        )
        # Calculate accuracy_given_attempted
        aggregate_metrics["accuracy_given_attempted"] = (
            aggregate_metrics["is_correct"] / aggregate_metrics["is_given_attempted"]
            if aggregate_metrics["is_given_attempted"] > 0
            else 0
        )
        print("AGGREGATE METRICS")
        print(aggregate_metrics)
        print("##################")

        output_d = {
            "accuracy_given_attempted": aggregate_metrics["accuracy_given_attempted"],
            "f1": (
                2
                * aggregate_metrics["accuracy_given_attempted"]
                * aggregate_metrics["is_correct"]
                / (
                    aggregate_metrics["accuracy_given_attempted"]
                    + aggregate_metrics["is_correct"]
                )
                if (
                    aggregate_metrics["accuracy_given_attempted"]
                    + aggregate_metrics["is_correct"]
                )
                > 0
                else 0
            ),
        }

        print(f"Accuracy Given Attempted: {output_d['accuracy_given_attempted']:.3f}")
        print(f"F1 Score: {output_d['f1']:.3f}")

        return common.aggregate_results(results)
