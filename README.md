# ESSAM
A novel zeroth-order fine-tuning method for improving the mathematical reasoning ability of large language models, which combines Evolution Strategies (ES) with Sharpness-Aware Maximization (SAM).

### Create a Python environment

Using conda:

```bash
conda create -n essam_env python=3.10
conda activate essam_env
```

Or using venv:

```bash
python -m venv essam_env
source essam_env/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

Run ESSAM on the GSM8K dataset:

```bash
bash essam_run.sh
```

Try using the accelerated version of ESSAM, namely ESSAM-F, which can achieve approximately 2× speedup while still obtaining competitive performance:

```bash
bash essam-fen_run.sh
```

---

## Citation

If you find this work helpful in your research, please cite:

```bibtex
@misc{sun2026essamnovelcompetitiveevolution,
      title={ESSAM: A Novel Competitive Evolution Strategies Approach to Reinforcement Learning for Memory Efficient LLMs Fine-Tuning}, 
      author={Zhishen Sun and Sizhe Dang and Guang Dai and Haishan Ye},
      year={2026},
      eprint={2602.01003},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.01003}, 
}
```
