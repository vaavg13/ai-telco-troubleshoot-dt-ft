ai-telco-troubleshoot-ch
========================

Project scripts to fine-tune and evaluate `Qwen/Qwen2.5-1.5B-Instruct` on the `netop/TeleLogs` dataset.

Setup
-----
- Python 3.10+ with PyTorch (distributed) available and a GPU that supports `bfloat16`.
- Install deps: `pip install -r requirements.txt`.

Training
--------
Launch two processes on one node with OpenMP threads capped at 4:
- `OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 train.py`

Evaluation
----------
Runs greedy generation against the test split and reports accuracy for checkpoint.
- `python eval_qwen.py --model-path`
