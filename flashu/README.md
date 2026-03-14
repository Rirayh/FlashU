# FlashU Plugin Architecture

FlashU is a training-free acceleration framework for unified multimodal models.
It is implemented as a plugin that monkey-patches a baseline Showo2Qwen2_5 model
without modifying any baseline source files.

## Quick Start

```python
from models import Showo2Qwen2_5
from flashu import apply_flashu_patch, FlashUConfig

# Load baseline model
model = Showo2Qwen2_5(**config.model.showo).to(device)
model.load_state_dict(state_dict)

# Configure and apply FlashU
flashu_config = FlashUConfig(
    r_p=0.20,       # FFN pruning ratio (Sec. 3.2)
    r_LS=0.20,      # Layer-skipping ratio (Sec. 3.3)
    T_LS=10,        # Layer-skip recalculation interval (Sec. 3.3)
    tau=10,          # Hybrid FFN final steps (Sec. 3.5)
    T_cache=5,       # Diffusion head cache interval (Sec. 3.4)
)
apply_flashu_patch(model, flashu_config)

# Use model normally -- acceleration is transparent
```
