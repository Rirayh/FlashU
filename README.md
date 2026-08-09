# Flash-Unified: A Training-Free and Task-Aware Acceleration Framework for Native Unified Models

[![CVPR 2026](https://img.shields.io/badge/CVPR%202026-Findings-blue.svg)](https://arxiv.org/abs/2603.15271)
[![arXiv](https://img.shields.io/badge/arXiv-2603.15271-b31b1b.svg)](https://arxiv.org/abs/2603.15271)

🎉 **This work has been accepted to Findings of CVPR 2026!** 🎉

FlashU is a **training-free, plug-and-play acceleration framework** for unified multimodal model. By applying a simple `apply_flashu_patch()` call, you can achieve **~2x speedup** on text-to-image generation with negligible quality loss.

## 🚀 Quick Start 

```python
from models import Showo2Qwen2_5  # Original Show-o2 baseline
from flashu import apply_flashu_patch, FlashUConfig

model = Showo2Qwen2_5.from_pretrained("showlab/show-o2-1.5B")
apply_flashu_patch(model, FlashUConfig())  # ⚡ That's it!

# Use the accelerated model exactly as before
images = model.t2i_generate(...)
```

## 🛠️ Installation

```bash
# Clone this repository
git clone https://github.com/Nicerova7/FlashU-Lab.git
cd FlashU-Lab

# Install dependencies (same as Show-o2)
pip install \
  transformers==4.46.3 \
  tokenizers==0.20.3 \
  diffusers==0.30.1 \
  accelerate \
  einops \
  omegaconf \
  tqdm \
  pillow \
  decord

# Download WAN 2.1 VAE weights
wget -c \
  https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1_VAE.pth \
  -O Wan2.1_VAE.pth
```

**Requirements:**
- Python >= 3.10
- PyTorch >= 2.4.0 (with FlexAttention support)
- CUDA >= 12.1
- GPU: >= 24GB VRAM

## 📖 Usage

### Basic Usage

```python
from models import Showo2Qwen2_5
from flashu import apply_flashu_patch, FlashUConfig

# Load baseline model
model = Showo2Qwen2_5.from_pretrained("showlab/show-o2-1.5B").cuda().eval()

# Apply FlashU acceleration
config = FlashUConfig(
    r_p=0.20,           # FFN pruning ratio
    r_LS=0.20,          # Layer skipping ratio
    T_LS=10,            # Layer importance recalc interval
    T_cache=5,          # Diffusion head cache interval
    tau=10,             # Hybrid FFN final steps
    schedule=[(4, 5.0), (8, 7.5), (20, 11.0)],  # Adaptive guidance
)
apply_flashu_patch(model, config)

# Use model as normal
images = model.t2i_generate(prompts, ...)
```

### DPG-Bench Evaluation

```bash
# Quick demo (5 prompts)
bash scripts/run_dpg_accelerated.sh --demo

# Full benchmark (1065 prompts)
bash scripts/run_dpg_accelerated.sh
```

The script automatically runs **both baseline and FlashU**, then prints a comparison table.

### Custom Configuration

```python
from flashu import FlashUConfig

# Aggressive speedup (lower quality)
fast_config = FlashUConfig(r_p=0.30, r_LS=0.30, tau=5)

# Conservative (higher quality)
quality_config = FlashUConfig(r_p=0.10, r_LS=0.10, tau=15)
```

### Reverting to Baseline

```python
from flashu import remove_flashu_patch

remove_flashu_patch(model)  # Restores original forward passes
```


## 🔬 Analysis Tools

### OBD Importance Score Computation — Generation

```bash
python -m evaluation.calculate_obd_cache_new \
    config=configs/showo2_1.5b_demo_432x432.yaml \
    validation_prompts_file=prompts/custom_prompts_20.json \
    start_sample_index=0 \
    num_prompts=20 \
    batch_size=4 \
    log_dir=logs_obd_cache
```

### OBD Importance Score Computation — Understanding

The understanding data file must contain JSON records with `image_path` and
`prompt` fields.

```bash
python -m evaluation.calculate_obd_cache_understanding \
    config=configs/showo2_1.5b_demo_432x432.yaml \
    understanding_data_file=prompts/understanding_calibration.json \
    start_sample_index=0 \
    num_prompts=5 \
    batch_size=1 \
    log_dir=logs_obd_cache
```

### Importance Score Visualization

```bash
python -m evaluation.visualize_obd_scores \
    --scores_dir <path_to_scores>
```

### Activation Analysis

```bash
python -m evaluation.analyze_t2i_activations1 \
    config=configs/showo2_1.5b_demo_432x432.yaml
```

## 🎓 Citation

```bibtex
@article{ke2026flashunified,
  title={Flash-Unified: A Training-Free and Task-Aware Acceleration Framework for Native Unified Models},
  author={Ke, Junlong and Wen, Zichen and Yang, Boxue and Yang, Yantai and Liu, Xuyang and Liao, Chenfei and Chen, Zhaorun and Wang, Shaobo and Zhang, Linfeng},
  journal={arXiv preprint arXiv:2603.15271},
  year={2026}
}
```

## 📜 License

Apache License 2.0. See headers in source files for details.

## 🙏 Acknowledgements

- **[Show-o2](https://github.com/showlab/Show-o)** by NUS Show Lab - The amazing baseline model
- **[Wanda](https://arxiv.org/abs/2306.11695)** - Structural pruning method
- **FlexAttention** (PyTorch 2.4+) - Efficient attention implementation

