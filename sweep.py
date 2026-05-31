#!/usr/bin/env python3

import datetime
import os
import signal
import subprocess
import threading
import urllib.request
import argparse
import time
from pathlib import Path

VLLM_URL        = "http://localhost:8000"
METRICS_URL     = f"{VLLM_URL}/metrics"
SCRAPE_INTERVAL = 1.0
POLL_INTERVAL   = 0.5
OUTPUT_DIR      = Path("results/sweep")
LORA_HF_PATH    = "AMaslovskyi/qwen-devops-foundation-lora"

VLLM_CMD = [
    "vllm", "serve", "Qwen/Qwen3-8B",
    "--port", "8000",
    "--max-model-len", "1024",
    "--dtype", "auto",
]

BENCH_CMD = [
    "vllm", "bench", "serve",
    "--backend", "vllm",
    "--model", "Qwen/Qwen3-8B",
    "--endpoint", "/v1/completions",
    "--dataset-name", "random",
    "--save-result",
    "--ignore-eos",
    "--random-input-len", "512",
    "--random-output-len", "128",
    "--percentile-metrics", "ttft,tpot,itl,e2el",
]

SWEEP_CMD = [
    "vllm", "bench", "sweep", "serve",
    "--output-dir", "results/sweep",
    "--num-runs", "5",
    "--show-stdout",
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--experiment", choices=["1a","1b","2"], required=True)
    p.add_argument("--tenants", type=int, default=None)
    p.add_argument("--max-loras", type=int, default=None)
    p.add_argument("--max-concurrency", type=int, default=None)                   
    p.add_argument("--param-file", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--num-prompts", type=int, default=1000)

    args = p.parse_args()

    if args.experiment in ["1a", "1b"]:
        if args.max_concurrency is not None:
            p.error("--max-concurrency cannot be set for experiment 1a and 1b")
        if args.experiment == "1b":
            if args.max_loras is None:
                p.error("--max-loras is required for experiment 1b")
            if args.tenants is None:
                p.error("--tenants is required for experiment 1b")
    elif args.experiment == "2":
        if args.max_loras is not None:
            p.error("--max-loras cannot be set for experiment 2")
        if args.max_concurrency is None:
            p.error("--max-concurrency is required for experiment 2")
        if args.tenants is None:
            p.error("--tenants is required for experiment 2")

    return args

def prep_lora(args: argparse.Namespace) -> str:
    lora_modules = [f"lora_{i}={LORA_HF_PATH}" for i in range(args.tenants)]
    lora_names = [f"lora_{i}" for i in range(args.tenants)]
    experiment_name = None

    if args.experiment == "1b":

        experiment_name = f"{args.tenants}t_MULTI_LORA_MAX_CONCURRENCY_SWEEP_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H-%M-%S-%f')}"

        VLLM_CMD.extend([
            "--max-loras", str(args.max_loras),
        ])

        SWEEP_CMD.extend([
            "--bench-params", args.param_file,
            "--experiment-name", experiment_name
        ])
        
    else:
        experiment_name = f"{args.tenants}t_MULTI_LORA_MAX_LORAS_SWEEP_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H-%M-%S-%f')}"

        SWEEP_CMD.extend([
            "--serve-params", args.param_file,
            "--experiment-name", experiment_name
        ])

        BENCH_CMD.extend([
            "--max-concurrency", str(args.max_concurrency),
        ])
    
    VLLM_CMD.extend([
            "--enable-lora",
            "--max-cpu-loras", str(args.tenants),
            "--lora-modules", *lora_modules,
        ])    

    BENCH_CMD.extend([
        "--lora-modules", *lora_names,
        "--lora-assignment", "round-robin",
        "--num-warmups", str(args.tenants),
        "--num-prompts", str(args.num_prompts),
    ])

    return experiment_name
    
def prep_base(args: argparse.Namespace) -> str:
    experiment_name = f"BASE_MAX_CONCURRENCY_SWEEP_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H-%M-%S-%f')}"
    BENCH_CMD.extend([
        "--num-prompts", str(args.num_prompts),
        "--num-warmups", "5",
    ])

    SWEEP_CMD.extend([
        "--bench-params", args.param_file,
        "--experiment-name", experiment_name
    ]) 

    return experiment_name    

def stream_output(proc:subprocess.Popen, log_file):
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        log_file.write(line)
        log_file.flush()

def start_vllm(args: argparse.Namespace, log_path: Path) -> subprocess.Popen:
    vllm_cmds = ' '.join(VLLM_CMD)
    bench_cmds = ' '.join(BENCH_CMD)

    SWEEP_CMD.extend([
        "--serve-cmd", vllm_cmds,
        "--bench-cmd", bench_cmds,
    ])

    proc = subprocess.Popen(
        SWEEP_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    if args.dry_run:
        return proc

    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(log_path, "w")

    # Run in background thread so Popen returns immediately
    thread = threading.Thread(
        target=stream_output,
        args=(proc, log_file),
        daemon=True)
     
    thread.start()

    proc._log_file = log_file
    proc._log_thread = thread
    
    return proc

def wait_for_vllm(timeout: int = 300):
    print(f"Waiting for vLLM, timeout in {timeout} seconds")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        t0 = time.monotonic()
        try:
            urllib.request.urlopen(f"{VLLM_URL}/metrics", timeout=POLL_INTERVAL)
            print("vLLM is ready\n")
            return
        except Exception:
            pass

        time.sleep(max(0, POLL_INTERVAL - (time.monotonic() - t0)))

    raise TimeoutError("vLLM did not get ready in time")

def stop_process_group(proc: subprocess.Popen):
    if proc is None:
        return
    
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()

    log_thread = getattr(proc, "_log_thread", None)
    if log_thread:
        log_thread.join(timeout=2)

    log_file = getattr(proc, "_log_file", None)
    if log_file:
        log_file.close()

def main():
    args = parse_args()

    if args.experiment in ["1b", "2"]:
        experiment_name = prep_lora(args)
    else:
        experiment_name = prep_base(args)
        
    if args.dry_run:
        SWEEP_CMD.append("--dry-run")

    log_path = OUTPUT_DIR / "logs" / f"{experiment_name}_sweep.log"
    sweep_proc = None
        
    try:
        sweep_proc = start_vllm(args, log_path)

        if not args.dry_run:
            wait_for_vllm(timeout=300) 
 
        return_code = sweep_proc.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, SWEEP_CMD)
 
    except KeyboardInterrupt:
        print("\nStopping sweep")
 
    finally:
        stop_process_group(sweep_proc)
    
if __name__ == "__main__":
    main()