import argparse
import json
import subprocess
from datetime import datetime
import os
import pandas as pd

from . import common

# from .common import common

# from .browsecomp_eval import BrowseCompEval
# from .drop_eval import DropEval
# from .gpqa_eval import GPQAEval
# from .healthbench_eval import HealthBenchEval
# from .healthbench_meta_eval import HealthBenchMetaEval
# from .math_eval import MathEval
# from .mgsm_eval import MGSMEval
# from .mmlu_eval import MMLUEval

# from .humaneval_eval import HumanEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
    VLLMChatCompletionSampler,
    GenChatCompletionSampler,
)
from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .sampler.responses_sampler import ResponsesSampler
from .simpleqa_eval import SimpleQAEval
from .hallu_eval import HalluLensEval
from .gsm8k_eval import GSM8KEval
from .nq_eval import NQEval

from .clamber_eval import ClamberEval
import dotenv

def main():
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Select a model by name. Also accepts a comma-separated list of models.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        help="Select an eval by name. Also accepts a comma-separated list of evals.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeats to run. Only supported for certain evals.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=120,
        help="Number of threads to run. Only supported for HealthBench and HealthBenchMeta.",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./simple-evals/output",
        help="Directory to save results",
    )

    args = parser.parse_args()
    models = {
        "qwen3": VLLMChatCompletionSampler(
            model="Qwen/Qwen3-8B",
            system_message="You are a helpful assistant.",
            temperature=0.5,
            port = 8000,
            max_tokens=32000
        ),
        "qwen3-sft-v4": VLLMChatCompletionSampler(
            model="Qwen/Qwen3-8B-SFT-V4",
            system_message="You are a helpful proactive assistant.",
            temperature=0.5,
            port = 8001,
            max_tokens=32000
        ),
    }

    if args.list_models:
        print("Available models:")
        for model_name in models.keys():
            print(f" - {model_name}")
        return

    if args.model:
        models_chosen = args.model.split(",")
        for model_name in models_chosen:
            if model_name not in models:
                print(f"Error: Model '{model_name}' not found.")
                return
        models = {model_name: models[model_name] for model_name in models_chosen}

    print(f"Running with args {args}")

    grading_sampler = GenChatCompletionSampler(
        model="gemini-2.5-flash-lite",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
    )
    equality_checker = GenChatCompletionSampler(
        model="gemini-2.5-flash-lite",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
    )

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "NQ":
                return NQEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    file_path="./simple-evals/data/NQ-open.train.jsonl",
                )
            case "gsm8k":
                return GSM8KEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case "hallu":
                return HalluLensEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case "clamber":
                return ClamberEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    file_path="./simple-evals/data/clamber_benchmark.jsonl"
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    if args.eval:
        evals_list = args.eval.split(",")
        evals = {}
        for eval_name in evals_list:
            try:
                evals[eval_name] = get_evals(eval_name, args.debug)
            except Exception:
                print(f"Error: eval '{eval_name}' not found.")
                return
    else:
        evals = {
            eval_name: get_evals(eval_name, args.debug)
            for eval_name in [
                "NQ",
                # "hallu",
                # "gsm8k",
            ]
        }

    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    print(f"Running the following evals: {list(evals.keys())}")
    print(f"Running evals for the following models: {list(models.keys())}")

    out_dir = args.output_dir
    if os.path.exists(out_dir) is False:
        os.makedirs(out_dir)
    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            # 这里传入的 sampler 是需要进行采样的模型，之后eval 来对于采样的结果做eval
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{model_name}"
            # file stem should also include the year, month, day, and time in hours and minutes
            file_stem += f"_{date_str}"
            report_filename = f"{out_dir}/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            # Sort metrics by key
            metrics = dict(sorted(metrics.items()))
            print(metrics)
            result_filename = f"{out_dir}/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")

            full_result_filename = (
                f"{out_dir}/{file_stem}{debug_suffix}_allresults.json"
            )
            with open(full_result_filename, "w") as f:
                result_dict = {
                    "score": result.score,
                    "metrics": result.metrics,
                    "htmls": result.htmls,
                    "convos": result.convos,
                    "metadata": result.metadata,
                }
                f.write(json.dumps(result_dict, indent=2))
                print(f"Writing all results to {full_result_filename}")

            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()
