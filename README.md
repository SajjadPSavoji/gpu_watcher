# gpu_watcher

**gpu_watcher** is a Python utility that monitors your system's NVIDIA GPUs and automatically keeps them occupied with dummy workloads when they are idle. This can help prevent GPU auto-suspend, maintain thermals, or keep resources reserved.

---

## 📋 Features

- **Automatic GPU monitoring**: Checks GPU usage periodically.
- **Dummy workload launcher**: Starts a matrix multiplication workload to keep the GPU active.
- **Multi-GPU support**: Works across all detected GPUs.
- **Clean shutdown**: Gracefully terminates workloads on exit or interruption.

---

## ⚙️ Installation

This project uses [uv](https://github.com/astral-sh/uv) as the Python environment and package manager.  
Make sure you have `uv` installed (`pip install uv`) before you start.

1. **Create and activate a virtual environment**

   ```bash
   uv venv .venv --python=python3.10
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   uv pip install -e .
   ```

---

## ▶️ Usage

Once installed, simply run:

```bash
python watch.py
```

By default, the script:

- Checks every **5 minutes** to see whether any GPU is idle.
- Starts a dummy process on each free GPU to occupy it.
- Monitors and logs the status of each GPU.

To stop the script and clean up all running processes, press `Ctrl+C` or send a termination signal. All dummy workloads will be terminated gracefully.

---

## 📂 Project Structure

```
gpu_watcher/
├── watch.py      # Main script for monitoring and occupying GPUs
├── pyproject.toml
└── README.md
```

---

## 🖥️ How It Works

1. **Monitoring**  
   `nvidia-smi` is called to query which processes are attached to each GPU.

2. **Decision Logic**  
   - If a GPU is in use by any process, it is left alone.
   - If a GPU is idle, a new `torch` process is started to run repeated large matrix multiplications on that GPU.

3. **Shutdown Handling**  
   Signal handlers (`SIGINT`, `SIGTERM`) and `atexit` hooks ensure that any spawned processes are killed cleanly.

---

## 🛑 Clean Shutdown

The script listens for interrupt and termination signals. When it exits:

- All spawned dummy processes are terminated.
- Resources are released properly.

---

## 🧩 Dependencies

- Python ≥ 3.10
- [PyTorch](https://pytorch.org/)
- NumPy

These are declared in `pyproject.toml`.

---

## ✏️ Example Output

When running, you will see logs similar to:

```
GPU utilization supervisor started.
[INFO] GPU 0 is in use.
[INFO] GPU 1 is free. Launching dummy workload.
```

---

## 📜 License

This project is provided under the MIT License.

---

## 🙋 Author

**Sajjad Pakdamansavoji**  
📧 [sj.pakdaman.edu@gmail.com](mailto:sj.pakdaman.edu@gmail.com)

---

## ⭐️ Contributing

Feel free to open issues or submit pull requests to improve the project.
