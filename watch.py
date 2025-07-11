import argparse
import subprocess
import time
import torch
import torch.multiprocessing as mp
import signal
import sys
import os
import atexit

# -----------------------------------------------------------------------------
# Command-line argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Spawn dummy GPU workloads only on selected devices."
)
parser.add_argument(
    "--gpus",
    type=str,
    default=None,
    help="Comma-separated list of GPU IDs to manage (e.g. '0,2,3')."
)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Determine which GPUs to manage
# -----------------------------------------------------------------------------
# Fetch the total list of available GPU indices
all_devices = list(range(torch.cuda.device_count()))

if args.gpus:
    # Parse the user-provided GPU list
    try:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    except ValueError:
        print("⛔ Could not parse --gpus. Please provide comma-separated integers, e.g. '0,2'.")
        sys.exit(1)
else:
    # If no GPUs specified, default to all available devices
    gpu_ids = all_devices

# Dictionary to track our dummy-work processes per GPU
active_processes = {}

# -----------------------------------------------------------------------------
# Worker function: keeps a single GPU busy with repeated large matrix multiplications
# -----------------------------------------------------------------------------
def gpu_worker(gpu_id, size=(2**16, 2**16)):
    """
    Occupy the specified GPU with an infinite loop of large matmuls.
    This ensures sustained high utilization for testing purposes.
    """
    torch.cuda.set_device(gpu_id)
    a = torch.randn(size, device=f'cuda:{gpu_id}')
    b = torch.randn(size, device=f'cuda:{gpu_id}')
    while True:
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

# -----------------------------------------------------------------------------
# Utility: check which GPUs are currently in use by other processes
# -----------------------------------------------------------------------------
def get_gpu_usage():
    """
    Query nvidia-smi to determine whether each GPU has active compute apps.
    Returns:
        usage (dict): {gpu_id: bool} where True means the GPU is in use.
    """
    # Query all running compute-apps (PID + GPU UUID)
    raw_output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        stderr=subprocess.DEVNULL
    ).decode().strip()

    usage = {}
    for gpu_id in all_devices:
        # Query this GPU's unique identifier
        uuid = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu=uuid", "--format=csv,noheader", f"--id={gpu_id}"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Determine if any line in the compute-apps output matches this GPU's UUID
        in_use = any(
            line.split(",")[1].strip() == uuid 
            for line in raw_output.splitlines()
        ) if raw_output else False

        usage[gpu_id] = in_use

    return usage

# -----------------------------------------------------------------------------
# Cleanup: terminate all dummy-work processes on exit or signal
# -----------------------------------------------------------------------------
def shutdown_all(*_):
    """
    Gracefully terminate all spawned GPU worker processes
    and exit the program.
    """
    print("\nShutting down all dummy GPU workers…")
    for gid, proc in active_processes.items():
        if proc.is_alive():
            print(f" • Terminating GPU {gid} (PID {proc.pid})")
            proc.terminate()
    # Ensure all processes have fully exited
    for proc in active_processes.values():
        proc.join()
    sys.exit(0)

# -----------------------------------------------------------------------------
# Main supervision loop
# -----------------------------------------------------------------------------
def main():
    # Exit early if no GPUs were selected
    if not gpu_ids:
        print("No GPUs selected. Exiting.")
        sys.exit(1)

    print(f"Managing GPUs: {gpu_ids}")

    # Use 'spawn' to safely fork processes when using CUDA
    mp.set_start_method("spawn", force=True)

    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, shutdown_all)
    signal.signal(signal.SIGTERM, shutdown_all)
    atexit.register(shutdown_all)

    # Periodically check GPU usage and spawn dummy workloads as needed
    while True:
        usage = get_gpu_usage()
        for gid in gpu_ids:
            if gid not in all_devices:
                print(f"[WARN] GPU {gid} does not exist—skipping.")
                continue

            if usage.get(gid, False):
                # GPU is already busy with other work
                print(f"[INFO] GPU {gid} is currently in use.")
            else:
                # Launch workload if not already running
                if gid in active_processes and active_processes[gid].is_alive():
                    print(f"[INFO] Dummy workload already running on GPU {gid}.")
                else:
                    print(f"[INFO] Launching dummy workload on free GPU {gid}.")
                    proc = mp.Process(target=gpu_worker, args=(gid,))
                    proc.start()
                    active_processes[gid] = proc

        # Wait before re-checking (5 minutes)
        time.sleep(5 * 60)

# Entry point guard
if __name__ == "__main__":
    main()
