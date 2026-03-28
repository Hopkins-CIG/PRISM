# PRISM: Probabilistic and Robust Inverse Solver with Measurement-Conditioned Diffusion Prior for Blind Inverse Problems (ICASSP 2026 Oral)

This repository contains the official implementation of **PRISM**, a plug-and-play diffusion framework for blind inverse problems (e.g., blind motion deblurring) with a measurement‑conditioned diffusion prior.

Paper: https://arxiv.org/pdf/2509.16106

---

### 1) Environment setup

We recommend using Conda:

```bash
conda create -n prism python=3.10
conda activate prism
pip install -r requirements.txt
```

Make sure your CUDA / PyTorch versions are compatible with the wheels specified in `requirements.txt`.

---

### 2) Download pretrained checkpoints

Download the corresponding pretrained models and place them under the `models/` directory as indicated in the config files.

- **Image prior (FFHQ color)**  
	[link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing)

- **Kernel prior (motion blur)**  
	[link](https://drive.google.com/drive/folders/15YbB385ZyqHxArU6MVUCOjKTZWMPDHYg?usp=sharing)

- **Kernel bank for evaluation (optional, used by `kernel_test_path`)**  
	[link](https://drive.google.com/drive/folders/1Q-rY7IUZjGeYtifi_XZT4YfUTQATrKlT?usp=sharing)

Example expected paths (you can change them in the config):

- `models/ffhq_10m.pt` – image prior checkpoint  
- `models/kernel_prior.pt` – kernel prior checkpoint

---

### 3) Configure data and models

Main configuration for FFHQ blind motion deblurring:

- [`config/ffhq_edm_motion_deblur256.yaml`](config/ffhq_edm_motion_deblur256.yaml)

In this file, adjust at least the following paths to your environment:

- `data.root` – path to your FFHQ test set (or your own dataset)
- `model.model_path` – path to the image prior checkpoint
- `kernel_model.model_path` – path to the kernel prior checkpoint
- `task.operator.kernel_test_path` – optional path to a `.npy` kernel bank (e.g. `models/kernel_test100.npy`)

If `kernel_test_path` is set, the motion blur operator will take kernels from the bank; otherwise, it will randomly generate motion‑blur kernels.

---

### 4) Run experiments

The main script for posterior sampling is [`posterior_sample.py`](posterior_sample.py).  
You can either use the provided shell script or call the Python script directly.

**Option A: use the shell script (recommended default config)**

```bash
bash prism.sh
```

**Option B: run directly with a specific config**

```bash
torchrun --nproc_per_node=1 posterior_sample.py \
    --config config/ffhq_edm_motion_deblur256.yaml \
    --output_dir ./results2/\
    --num_runs 1
```

Key arguments:

- `--config` – path to the YAML config file
- `--output_dir` – directory where results (gt, meas, recon, kernels_est, etc.) are saved
- `--num_runs` – number of posterior samples per image
- `--record` – if set, saves all intermediate diffusion samples

Outputs for each experiment are saved under `results/<exp_name>/`, where `exp_name` encodes the operator, noise, and sampler.

---
