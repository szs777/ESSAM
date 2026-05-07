import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import json
import time
import argparse
from typing import List, Dict, Any

import numpy as np
from vllm import LLM, SamplingParams

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import placement_group

from gsm8k.reward_function import reward_function


def parse_args():
    parser = argparse.ArgumentParser(description='vLLM evaluation for ES models on GSM8K')
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help='HF model name or base checkpoint path for vLLM')
    parser.add_argument('--trained_model_path', type=str, required=True,
                        help='Path to the fine-tuned model weight file (e.g., pytorch_model.pth)')

    parser.add_argument('--eval_data_path', type=str,
                        default='gsm8k/data/gsm8k/gsm8k_main_test.json',
                        help='Path to processed GSM8K evaluation JSON file (id/context/solution/answer)')
    parser.add_argument('--eval_samples', type=int, default=None,
                        help='Number of evaluation samples to evaluate (None = all)')
    parser.add_argument('--eval_offset', type=int, default=0,
                        help='Offset for evaluation data (negative means from end)')

    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--do_sample', action='store_true',
                        help='Whether to use sampling instead of greedy decoding')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p for nucleus sampling')

    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for prompts per vLLM.generate call (default: min(32, dataset_size))')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Tensor parallelism for vLLM')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Model dtype for vLLM')

    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save inference results (default: based on model name)')
    parser.add_argument('--save_responses', action='store_true',
                        help='Save individual responses to file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--show_examples', type=int, default=5,
                        help='Number of examples to show in detail')
    return parser.parse_args()


class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


def launch_engines(num_engines, model_name, tensor_parallel_size=1, dtype="float16"):
    pgs = [placement_group([{"GPU": 1, "CPU": 0}], lifetime="detached") for _ in range(num_engines)]
    ray.get([pg.ready() for pg in pgs])

    strategies = [
        PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        for pg in pgs
    ]

    engines = [
        ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(ESNcclLLM).remote(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="ray",
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype=dtype,
            enable_prefix_caching=False,
            enforce_eager=False,
            gpu_memory_utilization=0.6,
            max_model_len=32768,
        )
        for strategy in strategies
    ]
    return engines, pgs


def load_data(data_path: str, num_samples: int = None, offset: int = 0) -> List[Dict[str, Any]]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)

    if not isinstance(data_json, list):
        raise ValueError("Expected GSM8K data to be a list of dicts")

    dataset = data_json
    n = len(dataset)

    if offset < 0:
        start_idx, end_idx = max(0, n + offset), n
    else:
        start_idx, end_idx = offset, n
    dataset = dataset[start_idx:end_idx]

    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset[:num_samples]
    return dataset


def extract_model_response(generated_text: str) -> str:
    model_response = generated_text
    if "assistant:" in generated_text:
        model_response = generated_text.split("assistant:")[-1].strip()
    return model_response


def evaluate_batch_vllm(
    llm,
    batch_dataset: List[Dict[str, Any]],
    args,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    if verbose:
        print(f"Batch evaluating {len(batch_dataset)} samples...")

    temperature = args.temperature if args.do_sample else 0.0
    top_p = args.top_p if args.do_sample else 1.0
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=args.max_new_tokens,
    )

    input_texts = [item["context"] for item in batch_dataset]

    outputs = ray.get(llm.generate.remote(input_texts, sampling_params=sampling_params, use_tqdm=False))

    all_results = []
    for sample, out in zip(batch_dataset, outputs):
        completion_text = out.outputs[0].text if out.outputs else ""
        generated_text = f"{completion_text}"

        model_response = extract_model_response(generated_text)
        target_answer = sample["answer"]

        reward_result = reward_function(
            response=model_response,
            numbers=None,
            target=target_answer,
            end_token=None,
        )
        reward = reward_result["reward"]
        reward_info = reward_result["reward_info"]

        all_results.append({
            'id': sample.get('id', None),
            'input_text': sample['context'],
            'gt_solution': sample.get('solution', None),
            'gt_answer': target_answer,
            'generated_text': generated_text,
            'model_response': model_response,
            'reward': reward,
            'reward_info': reward_info
        })
    return all_results


def evaluate_dataset_vllm(
    llm,
    dataset: List[Dict[str, Any]],
    args,
    dataset_name: str,
    batch_size: int = None
) -> Dict[str, Any]:
    print(f"\n=== Evaluating on {dataset_name} dataset ({len(dataset)} samples) ===")
    if batch_size is None:
        batch_size = min(1024, len(dataset))
    print(f"Using batch size: {batch_size}")

    all_results = []
    total_reward = 0.0
    total_format_reward = 0.0
    total_answer_reward = 0.0

    start_time = time.time()
    num_batches = (len(dataset) - 1) // batch_size + 1
    for batch_idx, batch_start in enumerate(range(0, len(dataset), batch_size), start=1):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_dataset = dataset[batch_start:batch_end]

        if args.verbose:
            print(f"Processing batch {batch_idx}/{num_batches} (samples {batch_start+1}-{batch_end})...")

        batch_results = evaluate_batch_vllm(llm, batch_dataset, args, verbose=args.verbose)
        all_results.extend(batch_results)

        for result in batch_results:
            total_reward += result['reward']
            total_format_reward += result['reward_info']['format_reward']
            total_answer_reward += result['reward_info']['answer_reward']

        if batch_start == 0:
            for i, result in enumerate(batch_results[:args.show_examples]):
                print(f"\n--- Example {i+1} ---")
                print(f"ID: {result['id']}")
                print(f"Input: {result['input_text']}")
                print(f"GT Answer: {result['gt_answer']}")
                print(f"Model Response: {result['model_response']}")
                print(
                    "Reward: "
                    f"{result['reward']:.4f} "
                    f"(Format: {result['reward_info']['format_reward']:.4f}, "
                    f"Answer: {result['reward_info']['answer_reward']:.4f})"
                )

    eval_time = time.time() - start_time

    num_samples = len(dataset)
    avg_reward = total_reward / num_samples
    avg_format_reward = total_format_reward / num_samples
    avg_answer_reward = total_answer_reward / num_samples

    rewards = [r['reward'] for r in all_results]
    std_reward = float(np.std(rewards))
    min_reward = float(np.min(rewards))
    max_reward = float(np.max(rewards))

    high_reward_count = sum(1 for r in rewards if r >= 1.0)
    high_reward_percentage = high_reward_count / num_samples * 100.0

    answer_rewards = [r['reward_info']['answer_reward'] for r in all_results]
    correct_count = sum(1 for r in answer_rewards if r > 0)
    accuracy = correct_count / num_samples * 100.0

    stats = {
        'dataset_name': dataset_name,
        'num_samples': num_samples,
        'avg_reward': avg_reward,
        'avg_format_reward': avg_format_reward,
        'avg_answer_reward': avg_answer_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'high_reward_count': high_reward_count,
        'high_reward_percentage': high_reward_percentage,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'eval_time': eval_time,
        'all_results': all_results
    }

    print(f"\n=== {dataset_name} Results Summary ===")
    print(f"Number of samples: {num_samples}")
    print(f"Average reward: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"  - Format reward: {avg_format_reward:.4f}")
    print(f"  - Answer reward: {avg_answer_reward:.4f}")
    print(f"Accuracy (answer_reward > 0): {correct_count}/{num_samples} ({accuracy:.1f}%)")
    print(f"High reward samples (≥1.0): {high_reward_count}/{num_samples} ({high_reward_percentage:.1f}%)")
    print(f"Reward range: [{min_reward:.4f}, {max_reward:.4f}]")
    print(f"Evaluation time: {eval_time:.2f}s ({eval_time/num_samples:.3f}s per sample)")

    return stats


def save_results(results: Dict[str, Any], output_dir: str, args):
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        'model_id': args.model_id,
        'eval_stats': {k: v for k, v in results['eval_stats'].items() if k != 'all_results'},
        'generation_config': {
            'max_new_tokens': args.max_new_tokens,
            'do_sample': args.do_sample,
            'temperature': args.temperature if args.do_sample else None,
            'top_p': args.top_p if args.do_sample else None,
            'batch_size': args.batch_size,
            'tensor_parallel_size': args.tensor_parallel_size,
            'dtype': args.dtype,
        }
    }

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_path}")

    if args.save_responses:
        eval_details_path = os.path.join(output_dir, 'eval_detailed_results.json')
        with open(eval_details_path, 'w', encoding='utf-8') as f:
            json.dump(results['eval_stats']['all_results'], f, indent=2, ensure_ascii=False)
        print(f"Eval detailed results saved to: {eval_details_path}")


def main():
    args = parse_args()

    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    if args.output_dir is None:
        model_name = os.path.basename(args.model_id.rstrip('/'))
        batch_suffix = f"_batch{args.batch_size}" if args.batch_size else ""
        args.output_dir = f"./inference_results_vllm_gsm8k_{model_name}{batch_suffix}"

    print("=== vLLM ES Model Inference Script (GSM8K) ===")
    print(f"Base model path / HF id: {args.model_id}")
    print(f"Fine-tuned weights: {args.trained_model_path}")
    print(f"Eval data: {args.eval_data_path} (samples: {args.eval_samples}, offset: {args.eval_offset})")
    print(f"Output directory: {args.output_dir}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Dtype: {args.dtype}")

    engines, pgs = launch_engines(
        num_engines=1,
        model_name=args.model_id,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )
    llm = engines[0]
    llm.collective_rpc.remote("load_weights_from_disk", args=(args.trained_model_path,))
    print(f"vLLM LLM initialized from {args.trained_model_path}")

    eval_data_full_path = os.path.join(os.path.dirname(__file__), args.eval_data_path)
    eval_dataset = load_data(
        eval_data_full_path,
        num_samples=args.eval_samples,
        offset=args.eval_offset
    )
    print(f"Loaded {len(eval_dataset)} evaluation samples from {eval_data_full_path}")

    eval_stats = evaluate_dataset_vllm(llm, eval_dataset, args, "GSM8K-Eval", batch_size=args.batch_size)
    results = {'eval_stats': eval_stats}
    save_results(results, args.output_dir, args)

    for e in engines:
        try:
            ray.kill(e)
        except Exception:
            pass
    for pg in pgs:
        try:
            from ray.util.placement_group import remove_placement_group
            remove_placement_group(pg)
        except Exception:
            pass
    ray.shutdown()


if __name__ == "__main__":
    main()
