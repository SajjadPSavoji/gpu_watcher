import subprocess
import time
import torch
import torch.multiprocessing as mp
import signal
import sys
import os
import atexit

# Keeps track of running processes per GPU
active_processes = {}

def gpu_worker(gpu_id, size=(2**16, 2**16)):
    """Keep a single GPU busy by performing repeated large matmuls."""
    torch.cuda.set_device(gpu_id)
    a = torch.randn(size, device=f'cuda:{gpu_id}')
    b = torch.randn(size, device=f'cuda:{gpu_id}')
    while True:
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

def get_gpu_usage():
    """
    Returns a dict {gpu_id: True/False} indicating whether each GPU has any processes.
    """
    gpu_usage = {}
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--query-compute-apps=pid,gpu_uuid",
                    "--format=csv,noheader"
                ],
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()

            gpu_uuid = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--query-gpu=uuid",
                    "--format=csv,noheader",
                    f"--id={gpu_id}"
                ],
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()

            gpu_in_use = any(line.split(",")[1].strip() == gpu_uuid for line in output.splitlines()) if output else False

            gpu_usage[gpu_id] = gpu_in_use

        except Exception as e:
            print(f"Error checking GPU {gpu_id}: {e}")
            gpu_usage[gpu_id] = False

    return gpu_usage

def shutdown_all():
    print("\nShutting down all dummy GPU workers...")
    for gpu_id, proc in active_processes.items():
        if proc.is_alive():
            print(f"Terminating process for GPU {gpu_id} (PID {proc.pid})...")
            proc.terminate()
    for gpu_id, proc in active_processes.items():
        proc.join()
    print("All workers terminated.")
    sys.exit(0)

def main():
    print("GPU utilization supervisor started.")
    mp.set_start_method('spawn', force=True)

    # Register clean shutdown hooks
    signal.signal(signal.SIGINT, lambda s, f: shutdown_all())
    signal.signal(signal.SIGTERM, lambda s, f: shutdown_all())
    atexit.register(shutdown_all)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found.")
        sys.exit(1)

    while True:
        usage = get_gpu_usage()

        for gpu_id in range(num_gpus):
            in_use = usage.get(gpu_id, False)

            if in_use:
                print(f"[INFO] GPU {gpu_id} is in use.")
            else:
                if gpu_id in active_processes and active_processes[gpu_id].is_alive():
                    print(f"[INFO] GPU {gpu_id} is already occupied by our dummy workload.")
                else:
                    print(f"[INFO] GPU {gpu_id} is free. Launching dummy workload.")
                    p = mp.Process(target=gpu_worker, args=(gpu_id,))
                    p.start()
                    active_processes[gpu_id] = p

        time.sleep(5 * 60)  # 5 minutes

if __name__ == "__main__":
    main()
