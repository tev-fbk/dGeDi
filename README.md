<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0+cu118-ee4c2c?logo=pytorch&logoColor=white)
![IROS](https://img.shields.io/badge/IROS-2025-4b4b4b)

<h1>⚗️ Distilling 3D distinctive local descriptors for 6D pose estimation</h1>

[Amir Hamza](https://scholar.google.com/citations?user=GYZWQrAAAAAJ&hl=en&authuser=1)<sup>1,2</sup> ·
[Andrea Caraffa](https://scholar.google.com/citations?user=ARB9u4cAAAAJ&hl=en&authuser=1)<sup>1</sup> ·
[Davide Boscaini](https://davideboscaini.github.io/)<sup>1</sup> ·
[Fabio Poiesi](https://fabiopoiesi.github.io/)<sup>1</sup>

<sup>1</sup>TeV – Fondazione Bruno Kessler &nbsp;|&nbsp; <sup>2</sup>University of Trento

{ahamza, acaraffa, dboscaini, poiesi}@fbk.eu

[![arXiv](https://img.shields.io/badge/arXiv-2503.15106-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.15106v3)
[![Website](https://img.shields.io/badge/Project-Website-blue?logo=google-chrome&logoColor=white)](https://tev-fbk.github.io/dGeDi/)
[![YouTube](https://img.shields.io/badge/Video-YouTube-red?logo=youtube&logoColor=white)](https://youtu.be/jPz__8H0csc)
[![HuggingFace](https://img.shields.io/badge/Checkpoints-HuggingFace-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/ahamza848/dGeDi)


<img src="./static/IROS25_dGeDi-teaser.avif" width="750" alt="dGeDi teaser">

</div>

---

## Table of Contents

- [Abstract](#abstract)
- [Environment Setup](#environment-setup)
- [Checkpoints](#checkpoints)
- [Data Layout](#data-layout)
- [Running the Demo](#running-the-demo)
- [Config Reference](#config-reference)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## Abstract

Three-dimensional local descriptors are crucial for encoding geometric surface properties, making them essential for various point cloud understanding tasks. Among these descriptors, GeDi has demonstrated strong zero-shot 6D pose estimation capabilities but remains computationally impractical for real-world applications due to its expensive inference process.

*Can we retain GeDi's effectiveness while significantly improving its efficiency?* In this work, we explore this question by introducing a knowledge distillation framework that trains an efficient student model to regress local descriptors from a GeDi teacher. Our key contributions include: an efficient large-scale training procedure that ensures robustness to occlusions and partial observations while operating under compute and storage constraints, and a novel loss formulation that handles weak supervision from non-distinctive teacher descriptors.

We validate our approach on five BOP Benchmark datasets and demonstrate a significant reduction in inference time while maintaining competitive performance with existing methods, bringing zero-shot 6D pose estimation closer to real-time feasibility — over **170× faster** than GeDi.

---

## Environment Setup

> **Tested configuration:** Ubuntu 20.04, CUDA 11.8, Python 3.10
>
> **Minimum requirements:** Ubuntu 18.04+, CUDA 11.3+, PyTorch 1.10.0+

```bash
git clone https://github.com/tev-fbk/dGeDi.git
cd dGeDi
```

### 1. Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 2. Create and activate the virtual environment
```bash
export UV_CACHE_DIR=/dGeDi/.cache/uv
uv venv /dGeDi/envs/dGeDi_env --python 3.10
source /dGeDi/envs/dGeDi_env/bin/activate
```

### 3. PyTorch 2.1.0 + CUDA 11.8
```bash
uv pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

### 4. Core scientific stack
```bash
uv pip install numpy==1.26.4 scipy h5py==3.14.0 pyyaml
```

### 5. Build tools
```bash
uv pip install setuptools wheel
```

### 6. SharedArray
```bash
uv pip install SharedArray==3.2.0 --no-build-isolation
```

### 7. Utilities
```bash
uv pip install ninja tensorboard==2.11.0 tensorboardX==2.6.2.2 \
               timm==0.6.13 addict einops plyfile termcolor yapf \
               loguru opencv-python-headless==4.10.0.84
```

### 8. PyTorch Geometric
```bash
uv pip install torch-geometric
uv pip install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

### 9. spconv
```bash
uv pip install spconv-cu118==2.3.8
```

### 10. Open3D
```bash
uv pip install open3d==0.17.0
```

### 11. Flash Attention

**Option A — prebuilt wheel (recommended, no compilation):**
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
uv pip install flash_attn-2.5.9.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

**Option B — build from source:**

Follow the official installation instructions at https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features

> **No Flash Attention?** Set `enable_flash: False` in `config_dgedi.yaml` — the model falls back to standard attention.
> For more details see the [PointTransformerV3 Flash Attention notes](https://github.com/pointcept/pointtransformerv3#flash-attention).

### 12. Other packages
```bash
uv pip install huggingface_hub tqdm scikit-learn
```

---

## Checkpoints

### Option A — automatic download
```bash
python download_ckpts.py
```
Fetches both weights from [HuggingFace `ahamza848/dGeDi`](https://huggingface.co/ahamza848/dGeDi) into `checkpoints/`.

### Option B — manual download
Download the `.pth` files from https://huggingface.co/ahamza848/dGeDi and place them as:
```
checkpoints/
├── dGeDi_single_scale.pth
└── dGeDi_multi_scale.pth
```

---

## Data Layout

Place your point clouds as follows:
```
pcds/
├── query_pcd/
│   └── obj_000001.ply          # query object model
└── target_pcd/
    ├── target_pcd_00001.ply    # scene / observation
    └── ...
```

---

## Running the Demo

```bash
# Multi-scale (default)
python demo.py --mode multi_scale

# Single-scale
python demo.py --mode single_scale
```
Note : All paths assume you are running from inside the dGeDi/ repo root.

Full argument reference:

| Argument | Default | Description |
|---|---|---|
| `--query_pcd` | `pcds/query_pcd/obj_000001.ply` | Path to query point cloud |
| `--target_dir` | `pcds/target_pcd` | Directory of target point clouds |
| `--config` | `config_dgedi.yaml` | Path to YAML config |
| `--mode` | `multi_scale` | `single_scale` or `multi_scale` |
| `--out_dir` | `./dgedi_pca_out` | Output directory |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--ransac_threshold` | `0.03` | Max correspondence distance for RANSAC |
| `--icp_threshold` | `0.03` | Max correspondence distance for ICP refinement |

### Output

Each run saves the following files in `--out_dir`:

| File | Description |
|---|---|
| `<query>_PCA.ply` | Query cloud coloured by PCA of dGeDi features |
| `<target>_PCA.ply` | Target cloud coloured in the same PCA space |
| `<query>_to_<target>_registered_RANSAC_ICP.ply` | Registered query + target overlay |

---

## Config Reference

`config_dgedi.yaml` controls model architecture and inference behaviour.

```yaml
single_scale:
  weights_path: 'checkpoints/dGeDi_single_scale.pth'
  model_config:
    enc_channels: [32, 64, 128, 256, 512] 
    enc_num_head: [2, 4, 8, 16, 32]          
    dec_channels: [32, 64, 128, 256]         
    dec_num_head: [4, 4, 8, 16]              
    enable_flash: True    # set False if Flash Attention is not installed
    normalize_features: True  # L2-normalise output descriptors

multi_scale:
  weights_path: 'checkpoints/dGeDi_multi_scale.pth'
  model_config:
    enc_channels: [32, 64, 128, 256, 512]
    enc_num_head: [2, 4, 8, 16, 32]
    dec_channels: [64, 64, 128, 256]        
    dec_num_head: [4, 4, 8, 16]
    enable_flash: True
    normalize_features: True
```

**Key options:**

- **`enable_flash`** — enables [Flash Attention](https://github.com/Dao-AILab/flash-attention) for faster, memory-efficient attention. Set to `False` if Flash Attention is not installed:
  ```yaml
  enable_flash: False
  ```
- **`normalize_features`** — L2-normalises descriptors before matching. Recommended `True` for RANSAC-based registration.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{hamza2025distilling3ddistinctivelocal,
  author  = {Hamza, Amir and Caraffa, Andrea and Boscaini, Davide and Poiesi, Fabio},
  title   = {Distilling 3D distinctive local descriptors for 6D pose estimation},
  journal = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year    = {2025},
}
```

---

## Acknowledgement

This study was funded by the European Union – NextGenerationEU, in the framework of the iNEST – Interconnected Nord-Est Innovation Ecosystem (Piano Nazionale di Ripresa e Resilienza (PNRR) – Missione 4, Componente 2, Investimento 1.5, D.D. 1058 23/06/2022, iNEST ECS00000043 – Spoke3, CUP E63C22001030007). The views and opinions expressed are solely those of the authors and do not necessarily reflect those of the European Union, nor can the European Union be held responsible for them. We acknowledge ISCRA for awarding this project access to the LEONARDO supercomputer, owned by the EuroHPC Joint Undertaking, hosted by CINECA (Italy). We also acknowledge the authors of [Point Transformer V3](https://github.com/pointcept/pointtransformerv3) for making their implementation publicly available, which facilitated this research.

---

## License

### Code License

This code is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

You may use this code **for academic purposes only**. Any commercial or military use is strictly forbidden.

Full license: https://creativecommons.org/licenses/by-nc/4.0/

### Website License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />The project website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>. Website template from <a href="https://github.com/nerfies/nerfies.github.io">Nerfies</a>.
