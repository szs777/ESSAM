import argparse
from datetime import datetime
import gc
import json
import os
import random
import shutil
import signal
import sys
import time

import numpy as np
import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port

from gsm8k.reward_function import reward_function

SIGMA = 0.001
ALPHA = 0.0005
RHO = 0.0005
POPULATION_SIZE = 30
NUM_ENGINES = 4
EXPERIMENT_DIR = "experiment"
CHUNK_SIZE = 200
EPOCHS = 40

def parse_args():
    parser = argparse.ArgumentParser(
        description="ES+SAM Fine-tuning for GSM8K Task with multi-engine NCCL sync"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--rho", type=float, default=RHO)
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    parser.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    parser.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument('--verbose', action='store_true', help='Print verbose logs')
    parser.add_argument(
        "--global_seed",
        type=int,
        help="Global random seed",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # set global random seed
    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)
        torch.cuda.manual_seed_all(args.global_seed)

    return args

class ESNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)

def launch_engines(num_engines, model_name):
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
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="utils.worker_extn.WorkerExtension",
            dtype="float16",
            enable_prefix_caching=False,
            enforce_eager=False,
        )
        for strategy in strategies
    ]
    return engines, pgs

def evaluate_gsm8k_handle(llm, task_datas):
    prompts = [d["context"] for d in task_datas]
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=42,
        max_tokens=1024,
    )
    handle = llm.generate.remote(prompts, sampling_params, use_tqdm=False)
    return handle, time.time()

def _postprocess_outputs(outputs, task_datas):
    rewards = []
    avg_rewards = []
    for output, data in zip(outputs, task_datas):
        response = output.outputs[0].text

        r = reward_function(
            response=response,
            numbers=None,
            target=data["answer"],
            end_token=None,
        )
        rewards.append(r)
        avg_rewards.append(r["reward"])
    return {
        "rewards": rewards,
        "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0,
    }

def save_hf_style_checkpoint_from_pth(pth_path: str, save_dir: str, base_ckpt_dir: str, tokenizer):
    try:
        state = torch.load(pth_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        hf_model = AutoModelForCausalLM.from_pretrained(
            base_ckpt_dir,
            torch_dtype=torch.float16,
        ).to("cpu")
        hf_model.eval()

        missing, unexpected = hf_model.load_state_dict(state, strict=False)

        if (len(missing) + len(unexpected)) > 0 and any(k.startswith("module.") for k in state.keys()):
            state2 = {k[len("module."):]: v for k, v in state.items()}
            missing, unexpected = hf_model.load_state_dict(state2, strict=False)

        os.makedirs(save_dir, exist_ok=True)
        try:
            hf_model.save_pretrained(save_dir, safe_serialization=True)
        except Exception:
            hf_model.save_pretrained(save_dir, safe_serialization=False)

        if getattr(hf_model, "generation_config", None) is not None:
            hf_model.generation_config.save_pretrained(save_dir)

        tokenizer.save_pretrained(save_dir)

        del hf_model
        gc.collect()

        print(f"[HF Save] HuggingFace-style checkpoint saved to: {save_dir}")
        if len(missing) + len(unexpected) > 0:
            print(f"[HF Save] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

    except Exception as e:
        print(f"[WARN] Failed to save HuggingFace-style checkpoint from {pth_path}: {repr(e)}")


def main(args):
    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True)

    logging_dir = f"{args.experiment_dir}/gsm8k_sam_nccl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=logging_dir)

    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16
    ).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    base_model_path = f"{model_saves_dir}/base_model"
    if os.path.exists(base_model_path):
        shutil.rmtree(base_model_path)
    os.makedirs(base_model_path, exist_ok=True)
    tokenizer.save_pretrained(base_model_path)
    base_model.save_pretrained(base_model_path)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data_path = "gsm8k/data/gsm8k_main_train.json"
    print(f"Loading GSM8K data from: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        task_datas = json.load(f)
    num_samples = len(task_datas)
    print(f"Loaded {num_samples} GSM8K training samples")

    chunk_size = args.chunk_size
    print(f"Will use chunk size = {chunk_size} samples per chunk.")

    engines, pgs = launch_engines(args.num_engines, base_model_path)

    master_address = get_ip()
    master_port = get_open_port()
    ray.get([
        engines[i].collective_rpc.remote(
            "init_inter_engine_group", args=(master_address, master_port, i, args.num_engines)
        )
        for i in range(args.num_engines)
    ])

    def cleanup():
        for llm in engines:
            try:
                ray.kill(llm)
            except Exception:
                pass
        for pg in pgs:
            try:
                remove_placement_group(pg)
            except Exception:
                pass
        ray.shutdown()

    def sig_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    def evaluate_population(seeds, iteration_idx, task_datas_for_iter, prefix="reward"):
        seeds_perf = {}
        seed_iter = iter(seeds)
        inflight = {}
        results_this_gen = []

        for eng_idx, llm in enumerate(engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            ray.get(llm.collective_rpc.remote(
                "perturb_self_weights",
                args=(seed, args.sigma, False)
            ))
            handle, start_ts = evaluate_gsm8k_handle(llm, task_datas_for_iter)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": eng_idx,
                "seed": seed,
                "start_ts": start_ts,
            }

        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            outputs = ray.get(h)
            metrics = _postprocess_outputs(outputs, task_datas_for_iter)
            elapsed = time.time() - meta["start_ts"]

            seeds_perf[meta["seed"]] = metrics
            results_this_gen.append(
                {"seed": meta["seed"], "avg_reward": metrics["avg_reward"], "time": elapsed}
            )

            llm = meta["engine"]
            ray.get(llm.collective_rpc.remote(
                "restore_self_weights",
                args=(meta["seed"], args.sigma)
            ))

            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue

            ray.get(llm.collective_rpc.remote(
                "perturb_self_weights",
                args=(next_seed, args.sigma, False)
            ))
            handle, start_ts = evaluate_gsm8k_handle(llm, task_datas_for_iter)
            inflight[handle] = {
                "engine": llm,
                "engine_idx": meta["engine_idx"],
                "seed": next_seed,
                "start_ts": start_ts,
            }
            if args.verbose:
                print(f"[{prefix}] Scheduled seed {next_seed} on engine {meta['engine_idx']}")

        all_avg_rewards = [v["avg_reward"] for v in seeds_perf.values()]
        mean_reward = float(np.mean(all_avg_rewards)) if all_avg_rewards else 0.0
        std_reward = float(np.std(all_avg_rewards)) if all_avg_rewards else 0.0
        min_reward = float(np.min(all_avg_rewards)) if all_avg_rewards else 0.0
        max_reward = float(np.max(all_avg_rewards)) if all_avg_rewards else 0.0

        print(f"[{prefix}] Mean reward: {mean_reward}, std: {std_reward}, min: {min_reward}, max: {max_reward}")
        for k in seeds_perf:
            seeds_perf[k]["norm_reward"] = (seeds_perf[k]["avg_reward"] - mean_reward) / (std_reward + 1e-8)
            if args.verbose:
                print(f"[{prefix}] Seed {k} normalized reward: {seeds_perf[k]['norm_reward']}")

        writer.add_scalar(f"{prefix}/mean", mean_reward, iteration_idx)
        writer.add_scalar(f"{prefix}/std", std_reward, iteration_idx)
        writer.add_scalar(f"{prefix}/min", min_reward, iteration_idx)
        writer.add_scalar(f"{prefix}/max", max_reward, iteration_idx)

        return seeds, seeds_perf, results_this_gen

    epochs = args.epochs
    global_iter = 0

    for epoch in range(epochs):
        print(f"\n========== Epoch {epoch + 1}/{epochs} ==========")

        indices = list(range(num_samples))
        random.shuffle(indices)

        task_chunks = []
        for start in range(0, num_samples, chunk_size):
            batch_indices = indices[start:start + chunk_size]
            chunk = [task_datas[idx] for idx in batch_indices]
            task_chunks.append(chunk)
        num_chunks = len(task_chunks)
        print(f"Epoch {epoch + 1}: num_chunks = {num_chunks}")

        chunk_indices = list(range(num_chunks))
        random.shuffle(chunk_indices)

        for pos_in_epoch, chunk_idx in enumerate(chunk_indices):
            current_tasks = task_chunks[chunk_idx]

            print(
                f"\n\n=== Generation {global_iter} "
                f"(epoch {epoch + 1}, chunk {pos_in_epoch + 1}/{num_chunks}) ==="
            )
            total_iter_start = time.time()

            base_seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
            base_seeds, base_seeds_perf, base_results = evaluate_population(
                base_seeds, iteration_idx=global_iter, task_datas_for_iter=current_tasks, prefix="reward"
            )

            sam_per_seed_coeffs = [
                (seed, (args.rho / args.population_size) * float(base_seeds_perf[seed]["norm_reward"]))
                for seed in base_seeds
            ]

            sam_perturb_start = time.time()
            handles = []
            for seed, coeff in sam_per_seed_coeffs:
                handles.append(
                    engines[0].collective_rpc.remote(
                        "perturb_self_weights",
                        args=(seed, coeff, True)
                    )
                )
            ray.get(handles)
            sam_perturb_elapsed = time.time() - sam_perturb_start
            if args.verbose:
                print(f"SAM perturbations applied in {sam_perturb_elapsed}s")
            writer.add_scalar("time/sam_perturbation", sam_perturb_elapsed, global_iter)

            sam_broadcast_start = time.time()
            ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])
            sam_broadcast_elapsed = time.time() - sam_broadcast_start
            if args.verbose:
                print(f"SAM broadcasted updated weights in {sam_broadcast_elapsed}s")
            writer.add_scalar("time/sam_broadcast", sam_broadcast_elapsed, global_iter)

            sam_seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
            sam_seeds, sam_seeds_perf, sam_results = evaluate_population(
                sam_seeds, iteration_idx=global_iter, task_datas_for_iter=current_tasks, prefix="sam_reward"
            )

            final_per_seed_coeffs = [
                (seed, (args.alpha / args.population_size) * float(sam_seeds_perf[seed]["norm_reward"]))
                for seed in sam_seeds
            ]

            sam_revert_start = time.time()
            handles = []
            for seed, coeff in sam_per_seed_coeffs:
                handles.append(
                    engines[0].collective_rpc.remote(
                        "perturb_self_weights",
                        args=(seed, coeff, False)
                    )
                )
            ray.get(handles)
            sam_revert_elapsed = time.time() - sam_revert_start
            if args.verbose:
                print(f"SAM perturbations reverted in {sam_revert_elapsed}s")
            writer.add_scalar("time/sam_revert", sam_revert_elapsed, global_iter)

            es_perturb_start = time.time()
            handles = []
            for seed, coeff in final_per_seed_coeffs:
                handles.append(
                    engines[0].collective_rpc.remote(
                        "perturb_self_weights",
                        args=(seed, coeff, False)
                    )
                )
            ray.get(handles)
            es_perturb_elapsed = time.time() - es_perturb_start
            if args.verbose:
                print(f"Applied final ES perturbations in {es_perturb_elapsed}s")
            writer.add_scalar("time/perturbation_application", es_perturb_elapsed, global_iter)

            broadcast_start = time.time()
            ray.get([e.collective_rpc.remote("broadcast_all_weights", args=(0,)) for e in engines])
            broadcast_elapsed = time.time() - broadcast_start
            if args.verbose:
                print(f"Broadcasted updated weights in {broadcast_elapsed}s")
            writer.add_scalar("time/broadcast", broadcast_elapsed, global_iter)

            eval_handle, _ = evaluate_gsm8k_handle(engines[0], current_tasks)
            eval_outputs = ray.get(eval_handle)
            eval_metrics = _postprocess_outputs(eval_outputs, current_tasks)
            updated_mean_reward = eval_metrics["avg_reward"]

            print(f"[Eval after update] Iteration {global_iter} mean train reward (chunk) = {updated_mean_reward}")

            if args.verbose:
                print("Base evaluation results:")
                for res_idx, res in enumerate(base_results):
                    print(f"[base] IDX:{res_idx} Seed {res['seed']} avg_reward: {res['avg_reward']}, time: {res['time']}s")
                print("SAM evaluation results:")
                for res_idx, res in enumerate(sam_results):
                    print(f"[sam ] IDX:{res_idx} Seed {res['seed']} avg_reward: {res['avg_reward']}, time: {res['time']}s")

            total_iter_end = time.time()
            iter_elapsed = total_iter_end - total_iter_start
            writer.add_scalar("time/iteration", iter_elapsed, global_iter)
            print(f"wall clock time for iteration {global_iter}: {iter_elapsed}s")
            print(f"=== Generation {global_iter} finished ===\n")

            global_iter += 1

        epoch_model_path = f"{model_saves_dir}/final_model_epoch_{epoch + 1}"
        os.makedirs(epoch_model_path, exist_ok=True)
        pth_path = f"{epoch_model_path}/pytorch_model.pth"
        ray.get(
            engines[0].collective_rpc.remote(
                "save_self_weights_to_disk", args=(pth_path,)
            )
        )
        print(f"Model after epoch {epoch + 1} saved to {epoch_model_path} (pth).")
        
        save_hf_style_checkpoint_from_pth(
            pth_path=pth_path,
            save_dir=epoch_model_path,
            base_ckpt_dir=base_model_path,
            tokenizer=tokenizer,
        )
        print(f"Model after epoch {epoch + 1} saved to {epoch_model_path} (hf-style).")

    print(f"Training finished. Models have been saved after each of {epochs} epochs.")

    cleanup()


if __name__ == "__main__":
    args = parse_args()
    main(args)