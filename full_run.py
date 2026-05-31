#!/usr/bin/env python3
import subprocess

COMMANDS = [
    ["uv", "run", "python", "sweep.py", "--experiment", "1a", "--param-file", "experiment_1_params.json"],
    ["uv", "run", "python", "sweep.py", "--experiment", "1b", "--tenants","1", "--max-loras", "1", "--param-file", "experiment_1_params.json"],
    ["uv", "run", "python", "sweep.py", "--experiment", "1b", "--tenants","8", "--max-loras", "8", "--param-file", "experiment_1_params.json"],
    ["uv", "run", "python", "sweep.py", "--experiment", "1b", "--tenants","16", "--max-loras", "16", "--param-file", "experiment_1_params.json"],
    ["uv", "run", "python", "sweep.py", "--experiment", "1b", "--tenants","32", "--max-loras", "32", "--param-file", "experiment_1_params.json"],
    ["uv", "run", "python", "sweep.py", "--experiment", "2", "--tenants", "32", "--max-concurrency", "32", "--param-file", "experiment_2_params.json"],
    ["uv", "run", "python", "sweep.py", "--experiment", "2", "--tenants", "32", "--max-concurrency", "64", "--param-file", "experiment_2_params.json"]
]

def main():
    for command in COMMANDS:
        print(f"\n$ {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)

    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    main()