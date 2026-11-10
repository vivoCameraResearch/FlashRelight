# flashrelight/training/lora.py

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f = base.in_features
        out_f = base.out_features
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: x must be Tensor here
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def apply_lora_to_wan(
    wan_model: nn.Module,
    target_substrings: list[str],
    rank: int = 32,
    alpha: float = 32.0,
    dropout: float = 0.0,
):
    """
    Safe LoRA injection for Wan2.2:
    - Only inject into wan_model.blocks.* modules (avoid list-based text_embedding path)
    - Only inject into nn.Linear
    - Match by name contains any of target_substrings (e.g. q,k,v,o,ffn.0,ffn.2)
    - Skip cond_cross_attn to allow full training of that branch
    - Skip text cross_attn (keep text path unchanged)
    """
    # Build a map from module full name -> module object
    name_to_module = dict(wan_model.named_modules())

    replaced = 0
    for module_name, module in list(name_to_module.items()):
        # 1) whitelist: only inside transformer blocks
        if not module_name.startswith("blocks."):
            continue

        # 2) only Linear
        if not isinstance(module, nn.Linear):
            continue

        # # 2.5) skip cond_cross_attn (new branch should be fully trained)
        # if ".cond_cross_attn." in module_name:
        #     continue

        # # 2.6) skip text cross_attn
        # if ".cross_attn." in module_name:
        #     continue

        # 3) must match target substrings
        if not any(s in module_name for s in target_substrings):
            continue

        # 4) replace this Linear in its parent
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = name_to_module[parent_name]
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced += 1

    if replaced == 0:
        raise RuntimeError(
            f"LoRA injection replaced 0 modules. Check target_substrings={target_substrings} "
            f"and WanModel module naming."
        )
    return replaced
