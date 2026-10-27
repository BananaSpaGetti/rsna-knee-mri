"""Shared model definition for RSNA Knee training and inference kernels.

forward() accepts either a (N,H,W) grayscale batch (stage A: expanded to 3
identical channels) or a (N,3,H,W) window batch (stage B: 2.5D windows of
consecutive cached slices, see study_windows). Stage A weights load and
produce identical outputs through either code path.
`spatial_pool` ("gap" default | "label_attn") is carried in run_config.json and
read back by train/eval/infer; checkpoints trained with "label_attn" carry an
extra `spatial_query` tensor and need the switch to load strictly.

Checkpoint compatibility: coatnet2 checkpoints carry `arch`/`family` in their
run_config so inference can pick the right ARCH_SPECS entry; v4b and earlier
(DINOv2-S only) load with arch=None, which resolves ARCH at call time."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import LABELS, N_SLOTS, find_input  # noqa: F401 (re-export)

ARCH = "vit_small_patch14_reg4_dinov2.lvd142m"
IMG = 224
UNFREEZE_BLOCKS = 6
HIDDEN = 256
# per-label spatial attention (spatial_pool="label_attn"): attention/GAP mix and
# logit cap, see docs/competitors-code/mattiaangeli-d4-serving.md section 2.2
SPATIAL_MIX = 0.85
SPATIAL_CAP = 2.0

# ImageNet constants — the fallback when timm cannot resolve the pretrained
# config offline (pretrained=False). Matches DINOv2's preprocessor contract.
_FALLBACK_MEAN = (0.485, 0.456, 0.406)
_FALLBACK_STD = (0.229, 0.224, 0.225)
# Qwen VL image processors normalise with the OpenAI CLIP statistics.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Per-architecture knobs. "stride" is the backbone's input stride (None for
# ViT: patch embed accepts img_size directly, no padding needed). "unfreeze"
# is "blocks" (freeze all but the last UNFREEZE_BLOCKS transformer blocks,
# the DINOv2-S recipe) or "all" (leave everything trainable, nothing frozen).
# Unknown arch names (e.g. the dry-run's vit_tiny) fall back to the ViT entry.
ARCH_SPECS = {
    "vit_small_patch14_reg4_dinov2.lvd142m": {
        "family": "dinov2s", "stride": None, "unfreeze": "blocks",
        "fallback_mean": _FALLBACK_MEAN, "fallback_std": _FALLBACK_STD,
        "create_kwargs": {},
    },
    "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k": {
        "family": "coatnet2", "stride": 32, "unfreeze": "all",
        "fallback_mean": (0.5, 0.5, 0.5), "fallback_std": (0.5, 0.5, 0.5),
        "create_kwargs": {"drop_path_rate": 0.1},
    },
    # Raptor's backbone (round 10, docs/raptor-diff.md item 1): the same CoAtNet-2
    # but fine-tuned at 384 px. "input_size" (optional) upsamples the padded
    # cache window to that side before the backbone so the pretrained scale is
    # matched; without it the 288 px window goes in as is.
    "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k": {
        "family": "coatnet2", "stride": 32, "unfreeze": "all",
        "fallback_mean": (0.5, 0.5, 0.5), "fallback_std": (0.5, 0.5, 0.5),
        "create_kwargs": {"drop_path_rate": 0.1},
    },
    "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k@384": {
        "family": "coatnet2", "stride": 32, "unfreeze": "all",
        "fallback_mean": (0.5, 0.5, 0.5), "fallback_std": (0.5, 0.5, 0.5),
        "create_kwargs": {"drop_path_rate": 0.1},
        "timm_arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "input_size": 384,
    },
    # Pure-CNN arm (round 10): a different model class from the hybrid
    # CoAtNet and the ViT DINOv2, added as a diversity teacher for the next
    # label round. timm's ConvNeXt constructor takes no img_size (fully
    # convolutional), so the spec opts out of that kwarg; padding to the
    # stride still applies.
    "convnext_small.fb_in22k_ft_in1k_384": {
        "family": "convnexts", "stride": 32, "unfreeze": "all",
        "fallback_mean": _FALLBACK_MEAN, "fallback_std": _FALLBACK_STD,
        "create_kwargs": {"drop_path_rate": 0.1}, "img_size_kwarg": False,
    },
    # Round 10 step A: the vision tower of the Qwen3.5-2B VLM (24-layer ViT,
    # patch 16, 2x2 spatial merge, 2048-d merged tokens) as a plain backbone
    # for the same per-label attention head. Loaded through transformers, not
    # timm ("loader"); stride 32 = patch * merge so the padded 288 px input
    # gives an 18x18 patch grid / 9x9 = 81 merged tokens, mean-pooled.
    "qwen3.5-2b-vision": {
        "family": "qwenv", "stride": 32, "unfreeze": "all",
        "fallback_mean": _CLIP_MEAN, "fallback_std": _CLIP_STD,
        "create_kwargs": {}, "loader": "qwen_vision", "hf_id": "Qwen/Qwen3.5-2B",
    },
    "qwen3.5-2b-vision-patch": {
        "family": "qwenvp", "stride": 32, "unfreeze": "all",
        "fallback_mean": _CLIP_MEAN, "fallback_std": _CLIP_STD,
        "create_kwargs": {}, "loader": "qwen_vision", "hf_id": "Qwen/Qwen3.5-2B",
        "feature": "patch",
    },
    # Round 10 step B: the whole Qwen3.5-2B VLM. Every window of a study goes
    # into ONE token sequence (slot tag + image tokens per window), so the LM
    # does the cross-window fusion; a linear head reads the last hidden state.
    # LoRA on the LM (attention, linear-attention and MLP projections) and
    # on the vision tower; everything else frozen. "unfreeze" is informational.
    "qwen3.5-2b-vlm": {
        "family": "qwenvlm", "stride": 32, "unfreeze": "lora",
        "fallback_mean": _CLIP_MEAN, "fallback_std": _CLIP_STD,
        "create_kwargs": {}, "loader": "qwen_vlm", "hf_id": "Qwen/Qwen3.5-2B",
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
                 "targets": ["qkv", "proj", "linear_fc1", "linear_fc2",
                             "q_proj", "k_proj", "v_proj", "o_proj",
                             "in_proj_qkv", "in_proj_z", "out_proj",
                             "gate_proj", "up_proj", "down_proj"]},
    },
    "qwen3.5-2b-vlm-meanimg-r32": {
        "family": "qwenvlm", "stride": 32, "unfreeze": "lora",
        "fallback_mean": _CLIP_MEAN, "fallback_std": _CLIP_STD,
        "create_kwargs": {}, "loader": "qwen_vlm", "hf_id": "Qwen/Qwen3.5-2B",
        "pool": "mean_image",
        "lora": {"r": 32, "alpha": 64, "dropout": 0.05,
                 "targets": ["qkv", "proj", "linear_fc1", "linear_fc2",
                             "q_proj", "k_proj", "v_proj", "o_proj",
                             "in_proj_qkv", "in_proj_z", "out_proj",
                             "gate_proj", "up_proj", "down_proj"]},
    },
    "qwen3.5-2b-vlm-meanimg": {
        "family": "qwenvlm", "stride": 32, "unfreeze": "lora",
        "fallback_mean": _CLIP_MEAN, "fallback_std": _CLIP_STD,
        "create_kwargs": {}, "loader": "qwen_vlm", "hf_id": "Qwen/Qwen3.5-2B",
        "pool": "mean_image",
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
                 "targets": ["qkv", "proj", "linear_fc1", "linear_fc2",
                             "q_proj", "k_proj", "v_proj", "o_proj",
                             "in_proj_qkv", "in_proj_z", "out_proj",
                             "gate_proj", "up_proj", "down_proj"]},
    },
}
_DEFAULT_SPEC = ARCH_SPECS[ARCH]  # keyed off ARCH, not a second literal copy


def _mark_grad(module, inputs, output):
    """Forward hook for a frozen-then-checkpointed stage: reentrant grad
    checkpointing only backprops through a segment whose input requires
    grad, so mark this stage's output as requiring grad during training to
    keep the checkpointed segments downstream trainable."""
    # No `module.training` test here: the frozen stage is kept in eval() by
    # Model.train(), so that flag is always False and would make the hook inert.
    if torch.is_grad_enabled() and not output.requires_grad:
        output.requires_grad_(True)


class QwenVisionBackbone(nn.Module):
    """The vision tower of a Qwen3.5 VLM behind a timm-like interface:
    forward((N,3,H,W) normalised) -> (N, num_features). Patchification
    reproduces transformers' Qwen2VL image processor (merge-major patch
    order, temporal_patch_size copies of the frame), so the pretrained patch
    embedding sees exactly the layout it was trained on. Each image is its
    own attention sequence (cu_seqlens); the 2x2-merged tokens from the
    merger are mean-pooled. H and W must be multiples of patch*merge."""

    def __init__(self, hf_id: str, pretrained: bool = True, feature: str = "merged"):
        super().__init__()
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(hf_id)
        vcfg = cfg.vision_config
        # "merged": the 2x2-merged 2048-d tokens the LM reads; "patch": the
        # 1024-d per-patch ViT states before the merger (a plain ViT output).
        self.feature = feature
        if pretrained:
            from transformers import Qwen3_5ForConditionalGeneration
            full = Qwen3_5ForConditionalGeneration.from_pretrained(
                hf_id, dtype=torch.float32, device_map="cpu")
            self.visual = full.model.visual
            del full
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
            self.visual = Qwen3_5VisionModel(vcfg)
        self.patch = vcfg.patch_size
        self.merge = vcfg.spatial_merge_size
        self.t_patch = vcfg.temporal_patch_size
        self.num_features = vcfg.out_hidden_size if feature == "merged" else vcfg.hidden_size
        if feature == "patch":  # the merger is bypassed: freeze it so no trainable param goes gradient-less
            for p in self.visual.merger.parameters():
                p.requires_grad = False

    def set_grad_checkpointing(self, enable: bool = True):
        if enable:
            self.visual.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.visual.gradient_checkpointing_disable()

    def patchify(self, x):
        n, c, h, w = x.shape
        p, m = self.patch, self.merge
        gh, gw = h // p, w // p
        assert gh % m == 0 and gw % m == 0, (h, w)
        x = x.reshape(n, c, gh // m, m, p, gw // m, m, p)
        x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)  # n, gh/m, gw/m, m, m, c, p, p
        x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, self.t_patch, -1, -1)
        return x.reshape(n * gh * gw, c * self.t_patch * p * p), gh, gw

    def forward(self, x):
        n = x.shape[0]
        flat, gh, gw = self.patchify(x)
        grid = torch.tensor([[1, gh, gw]], device=x.device).expand(n, -1)
        out = self.visual(flat, grid)
        feats = out.pooler_output if self.feature == "merged" else out.last_hidden_state
        return feats.reshape(n, -1, self.num_features).mean(1)


def _qwen_patchify(x, patch: int, merge: int, t_patch: int):
    """(N,3,H,W) -> (N*gh*gw, 3*t_patch*patch*patch) in the Qwen2VL image
    processor's merge-major order, plus the patch grid (gh, gw)."""
    n, c, h, w = x.shape
    gh, gw = h // patch, w // patch
    assert gh % merge == 0 and gw % merge == 0, (h, w)
    x = x.reshape(n, c, gh // merge, merge, patch, gw // merge, merge, patch)
    x = x.permute(0, 2, 5, 3, 6, 1, 4, 7)  # n, gh/m, gw/m, m, m, c, p, p
    x = x.unsqueeze(6).expand(-1, -1, -1, -1, -1, -1, t_patch, -1, -1)
    return x.reshape(n * gh * gw, c * t_patch * patch * patch), gh, gw


class QwenVLMCore(nn.Module):
    """Qwen3.5 VLM (vision tower + language model, no lm_head) with LoRA
    adapters, scoring a batch of studies: forward(x, slots, owner, n_studies)
    -> (n_studies, hidden) = the LM's last hidden state at the final prompt
    token of each study's sequence. Each study's sequence is
    `prefix + sum_over_its_windows(slot tag + <vision_start> <image_pad>*T
    <vision_end>) + suffix`; windows are taken in row order, and rows with
    owner < 0 (collate padding) are dropped. The base weights are bf16 and
    frozen; only LoRA parameters train."""

    def __init__(self, hf_id: str, lora: dict, pretrained: bool = True,
                 grad_ckpt: bool = True, pool: str = "last"):
        super().__init__()
        # "last": hidden state at the final prompt token; "mean_image": mean of
        # the hidden states at every image-token position of the study.
        self.pool = pool
        from transformers import AutoConfig, AutoTokenizer
        from transformers import Qwen3_5ForConditionalGeneration
        cfg = AutoConfig.from_pretrained(hf_id)
        if pretrained:
            full = Qwen3_5ForConditionalGeneration.from_pretrained(
                hf_id, dtype=torch.bfloat16, device_map="cpu")
        else:
            full = Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16)
        core = full.model
        del full
        for p in core.parameters():
            p.requires_grad = False
        if grad_ckpt:
            core.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        from peft import LoraConfig, inject_adapter_in_model
        lcfg = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"],
                          lora_dropout=lora["dropout"],
                          target_modules=list(lora["targets"]), bias="none")
        self.core = inject_adapter_in_model(lcfg, core)
        n_lora = 0
        for n, p in self.core.named_parameters():
            is_lora = "lora_" in n
            p.requires_grad = is_lora
            if is_lora:
                p.data = p.data.float()  # adapters train in fp32 under bf16 autocast
                n_lora += p.numel()
        self.n_lora = n_lora
        vcfg = cfg.vision_config
        self.patch, self.merge, self.t_patch = (
            vcfg.patch_size, vcfg.spatial_merge_size, vcfg.temporal_patch_size)
        self.hidden = cfg.text_config.hidden_size
        self.image_id = cfg.image_token_id
        self.vs_id, self.ve_id = cfg.vision_start_token_id, cfg.vision_end_token_id
        tok = AutoTokenizer.from_pretrained(hf_id)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

        def enc(t):
            return tok.encode(t, add_special_tokens=False)
        self.prefix = enc("Knee MRI study. One image per slice window, grouped by slot.")
        self.suffix = enc("\nFindings:")
        self.slot_tags = [enc(f"\nslot {k}:") for k in range(16)]

    def forward(self, x, slots, owner, n_studies: int):
        keep = owner >= 0
        if not bool(keep.all()):
            x, slots, owner = x[keep], slots[keep], owner[keep]
        flat, gh, gw = _qwen_patchify(x, self.patch, self.merge, self.t_patch)
        n_tok = (gh * gw) // (self.merge * self.merge)
        seqs, order = [], []
        owner_l, slots_l = owner.tolist(), slots.tolist()
        for i in range(n_studies):
            idx = [j for j, o in enumerate(owner_l) if o == i]
            ids = list(self.prefix)
            for j in idx:
                ids += (self.slot_tags[slots_l[j]] + [self.vs_id]
                        + [self.image_id] * n_tok + [self.ve_id])
            ids += self.suffix
            seqs.append(ids)
            order += idx
        n_rows = x.shape[0]
        order_t = torch.tensor(order, device=x.device, dtype=torch.long)
        flat = flat.view(n_rows, gh * gw, -1)[order_t].reshape(-1, flat.shape[-1])
        grid = torch.tensor([[1, gh, gw]], device=x.device).expand(n_rows, -1)
        L = max(len(q) for q in seqs)
        input_ids = torch.full((n_studies, L), self.pad_id, dtype=torch.long, device=x.device)
        attn = torch.zeros((n_studies, L), dtype=torch.long, device=x.device)
        for i, q in enumerate(seqs):
            input_ids[i, :len(q)] = torch.tensor(q, device=x.device)
            attn[i, :len(q)] = 1
        # mm_token_type_ids: 0 text, 1 image (M-RoPE needs it alongside input_ids)
        out = self.core(input_ids=input_ids, attention_mask=attn,
                        mm_token_type_ids=(input_ids == self.image_id).long(),
                        pixel_values=flat.to(torch.bfloat16), image_grid_thw=grid)
        hs = out.last_hidden_state
        if self.pool == "mean_image":
            img = (input_ids == self.image_id).unsqueeze(-1).to(hs.dtype)
            pooled = (hs * img).sum(1) / img.sum(1).clamp_min(1.0)
        else:
            last = attn.sum(1) - 1
            pooled = hs[torch.arange(n_studies, device=x.device), last]
        return pooled.float()


class Model(nn.Module):
    """forward()/embed()/head() as documented on the methods below.

    Reentrant-checkpoint trap: with `unfreeze="stem_stage0"`, the backbone's
    stem and first stage are frozen and grad-checkpointed. Reentrant
    checkpointing (older timm / torch default) back-propagates through a
    checkpointed segment only if that segment's INPUT tensor requires grad;
    since the frozen stem/stage0 params don't require grad and the input
    image doesn't either, the segment silently produces no gradient for any
    trainable param inside it (or anything downstream of it that relies on
    the checkpoint boundary re-running with grad). A forward hook on
    `backbone.stages[0]` marks that stage's output `requires_grad_(True)` so
    checkpointing downstream of the frozen prefix still tracks gradients.
    Frozen submodules are kept in eval() (BN running stats frozen) even
    across `.train()` calls via the `train()` override below."""

    def __init__(self, pretrained: bool = True, img_size: int = IMG,
                arch: str | None = None, grad_ckpt: bool = False,
                unfreeze: str | None = None, drop_path: float | None = None,
                n_slots: int | None = None, spatial_pool: str | None = None):
        super().__init__()
        import timm
        import math
        arch = arch or ARCH
        spec = ARCH_SPECS.get(arch)
        if spec is None:
            print(f"WARN arch {arch!r} not in ARCH_SPECS, using the ViT default spec")
            spec = _DEFAULT_SPEC
        self.arch = arch
        self.family = spec["family"]
        stride = spec["stride"]
        self.pad = 0 if stride is None else (
            math.ceil(img_size / stride) * stride - img_size)
        create_kwargs = dict(spec["create_kwargs"])
        if drop_path is not None:
            create_kwargs["drop_path_rate"] = drop_path
        self.unfreeze = unfreeze or spec["unfreeze"]
        self.spatial_pool = spatial_pool or "gap"
        if self.spatial_pool not in ("gap", "label_attn"):
            raise ValueError(f"unknown spatial_pool {self.spatial_pool!r} (gap | label_attn)")
        self.mode = "vlm" if spec.get("loader") == "qwen_vlm" else "backbone"
        if self.mode == "vlm":
            mean, std = spec["fallback_mean"], spec["fallback_std"]
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
            self.backbone = QwenVLMCore(spec["hf_id"], spec["lora"], pretrained,
                                        grad_ckpt=grad_ckpt,
                                        pool=spec.get("pool", "last"))
            self.vlm_head = nn.Sequential(nn.LayerNorm(self.backbone.hidden),
                                          nn.Dropout(0.1),
                                          nn.Linear(self.backbone.hidden, len(LABELS)))
            self._frozen = []
            if self.spatial_pool != "gap":
                raise ValueError(f"spatial_pool={self.spatial_pool!r} is not supported in VLM mode")
            return
        if spec.get("loader") == "qwen_vision":
            self.backbone = QwenVisionBackbone(spec["hf_id"], pretrained,
                                               feature=spec.get("feature", "merged"))
            mean, std = spec["fallback_mean"], spec["fallback_std"]
        else:
            # optional resize of the padded window before the backbone (e.g. to
            # the checkpoint's native 384 px); the timm arch may differ from the
            # spec key when one checkpoint is registered at several input sizes
            self.input_size = spec.get("input_size")
            timm_arch = spec.get("timm_arch", arch)
            if spec.get("img_size_kwarg", True):
                create_kwargs["img_size"] = self.input_size or (img_size + self.pad)
            self.backbone = timm.create_model(
                timm_arch, pretrained=pretrained, num_classes=0, **create_kwargs)
            try:
                cfg = timm.data.resolve_model_data_config(self.backbone)
                mean, std = cfg["mean"], cfg["std"]
            except Exception:
                mean, std = spec["fallback_mean"], spec["fallback_std"]
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        if grad_ckpt:
            self.backbone.set_grad_checkpointing(True)
        d = self.backbone.num_features
        # per-label attention: project backbone features, add a slot embedding,
        # then let each of the 12 labels attend to the study's slices separately.
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, HIDDEN), nn.GELU())
        # n_slots (v5: 5, the raptor5 layout) picks the slot embedding size;
        # None (v4b/v6c and earlier checkpoints) keeps the plane6 default.
        self.slot_emb = nn.Embedding(n_slots or N_SLOTS, HIDDEN)
        self.query = nn.Parameter(torch.randn(len(LABELS), HIDDEN) * 0.02)
        self.drop = nn.Dropout(0.2)
        self.out_weight = nn.Parameter(torch.randn(len(LABELS), HIDDEN) * 0.02)
        self.out_bias = nn.Parameter(torch.zeros(len(LABELS)))
        if self.spatial_pool == "label_attn":
            if self.family != "coatnet2":
                raise ValueError(f"spatial_pool='label_attn' is implemented for the coatnet2 family only (4-D post-norm feature map), not {self.family!r}")
            # per-label spatial queries over the patch grid; zero-init = uniform attention = today's GAP model
            self.spatial_query = nn.Parameter(torch.zeros(len(LABELS), d))

        self._frozen = []
        if self.unfreeze == "blocks":
            for p in self.backbone.parameters():
                p.requires_grad = False
            for blk in self.backbone.blocks[-UNFREEZE_BLOCKS:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True
        elif self.unfreeze == "all":
            pass
        elif self.unfreeze == "stem_stage0":
            if not (hasattr(self.backbone, "stem") and hasattr(self.backbone, "stages")):
                raise ValueError(
                    f"unfreeze='stem_stage0' needs stem/stages, arch {arch!r} has none")
            for p in self.backbone.stem.parameters():
                p.requires_grad = False
            for p in self.backbone.stages[0].parameters():
                p.requires_grad = False
            self._frozen = [self.backbone.stem, self.backbone.stages[0]]
            self.backbone.stages[0].register_forward_hook(_mark_grad)
        else:
            raise ValueError(f"unknown unfreeze mode {self.unfreeze!r}")

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self._frozen:
            m.eval()
        return self

    def _label_attn_pool(self, fmap):
        """(N, D, H, W) backbone feature map -> (N, 12, D): per-label attention over
        the H*W patch tokens (LayerNorm'd for scoring, raw for pooling), logits
        capped at SPATIAL_CAP*tanh(s/SPATIAL_CAP), output SPATIAL_MIX of the
        attention-pooled token + (1-SPATIAL_MIX) of the GAP token."""
        t = fmap.flatten(2).transpose(1, 2)                      # (N, P, D)
        tn = F.layer_norm(t, (t.shape[-1],))                     # no affine
        s = torch.einsum("npd,od->nop", tn, self.spatial_query) / t.shape[-1] ** 0.5
        a = torch.softmax(SPATIAL_CAP * torch.tanh(s / SPATIAL_CAP), dim=-1)  # (N, 12, P)
        g = t.mean(1, keepdim=True)                              # (N, 1, D) = the GAP token
        return g + SPATIAL_MIX * (torch.einsum("nop,npd->nod", a, t) - g)

    def embed(self, x, slots):
        """Backbone + proj + slot embedding. (N,H,W) or (N,3,H,W) -> (N, HIDDEN), or
        (N, 12, HIDDEN) per-label window features when spatial_pool="label_attn".
        VLM mode: no compute here - returns the padded, normalised pixels and
        slots for head(), which needs the pixels grouped per study."""
        if x.dim() == 3:  # (N,H,W) grey -> 3ch (stage A / older weights)
            x = x.unsqueeze(1).expand(-1, 3, -1, -1)
        if self.pad:
            lo = self.pad // 2
            hi = self.pad - lo
            x = F.pad(x, (lo, hi, lo, hi))
        x = (x - self.mean) / self.std
        if self.mode == "vlm":
            return (x, slots)
        if getattr(self, "input_size", None) and x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        if self.spatial_pool == "label_attn":
            v = self._label_attn_pool(self.backbone.forward_features(x))      # (N, 12, D)
            return self.proj(v) + self.slot_emb(slots).unsqueeze(1)            # (N, 12, HIDDEN)
        return self.proj(self.backbone(x)) + self.slot_emb(slots)  # (N, HIDDEN)

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if getattr(self, "mode", "backbone") != "vlm":
            return sd
        # VLM mode: checkpoint only what trains (LoRA + head) - the 2B base is
        # re-loaded from the HF weights; 4 GB per fold would be pointless.
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        prefix = kwargs.get("prefix", "")
        return type(sd)((k, v) for k, v in sd.items()
                        if k[len(prefix):] in keep or k.startswith(prefix + "vlm_head."))

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        if getattr(self, "mode", "backbone") == "vlm":
            strict = False
        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def head(self, h, owner, n_studies):
        """Per-label attention over rows sharing an `owner`, then dropout and
        readout. h: (N, HIDDEN) from embed(), or (N, 12, HIDDEN) per-label window
        features (spatial_pool="label_attn": label l attends over the study's
        windows using its own feature). VLM mode: h = (pixels, slots)
        from embed(); the LM scores each owner group as one sequence."""
        if self.mode == "vlm":
            x, slots = h
            return self.vlm_head(self.backbone(x, slots, owner, n_studies))
        ctxs = []
        for i in range(n_studies):
            m = owner == i
            hs = h[m]  # (S, HIDDEN) slices belonging to this study
            if hs.dim() == 3:  # (S, 12, HIDDEN): label l attends over windows with its own feature (spatial_pool="label_attn")
                logits = torch.einsum("soh,oh->os", hs, self.query) / HIDDEN ** 0.5
                a = torch.softmax(logits, dim=-1)
                ctxs.append(torch.einsum("os,soh->oh", a, hs))
                continue
            logits = torch.einsum("sh,oh->os", hs, self.query) / HIDDEN ** 0.5
            a = torch.softmax(logits, dim=-1)  # per-label distribution over slices
            ctxs.append(torch.einsum("os,sh->oh", a, hs))  # (12, HIDDEN)
        ctx = self.drop(torch.stack(ctxs))  # (n_studies, 12, HIDDEN)
        return (ctx * self.out_weight.unsqueeze(0)).sum(-1) + self.out_bias

    def forward(self, x, slots, owner, n_studies):
        return self.head(self.embed(x, slots), owner, n_studies)


def study_batch(vol, device):
    """(6,9,H,W) uint8 -> model inputs using every non-empty slice."""
    import numpy as np
    picks, slots = [], []
    for s in range(vol.shape[0]):
        filled = vol[s].reshape(vol.shape[1], -1).max(axis=1) > 0
        for j in np.flatnonzero(filled):
            picks.append(vol[s, j])
            slots.append(s)
    if not picks:
        picks, slots = [vol[0, 0]], [0]
    x = torch.from_numpy(np.stack(picks)).float().div_(255.0).to(device)
    slots_t = torch.tensor(slots, device=device)
    owner = torch.zeros(len(slots), dtype=torch.long, device=device)
    return x, slots_t, owner


def study_windows(vol, group: int = 3, stride: int | None = None):
    """(6,n_slices,H,W) uint8 -> `group`-slice windows as (N_windows,3,H,W)
    float in [0,1] (no device placement; caller moves it).

    stride=None (the training-time grouping, and the default every caller
    used before round 10): for each slot, `n_slices // group` non-overlapping
    windows of `group` consecutive cached slices are formed.

    stride=k (dense, inference-only): for each slot, one window centred on
    every k-th slice index; a window centred on slice i spans
    [i - group//2, i - group//2 + group), and indices outside [0, n_slices)
    are zero-padded (the cache's own "all-zero = no data" convention).
    stride=1 is Raptor's K_EVAL-style dense scoring; measured on the v6c
    fold checkpoints unchanged it is +0.0071 +/- 0.0007 paired OOF macro
    AUC over stride=None (src/eval_windows.py, docs/methodology.md), at
    ~group x the backbone work per study. The checkpoints were trained on
    the non-overlapping grouping, so this is a test-time change only.

    Either way a window is kept only if at least one of its slices is
    non-empty (all-zero slices are padding). If no window survives at all,
    returns one all-zero window at slot 0 (mirrors the study_batch
    fallback)."""
    import numpy as np
    n_slices = vol.shape[1]
    windows, slots = [], []
    if stride is None:
        n_groups = n_slices // group
        for s in range(vol.shape[0]):
            for g in range(n_groups):
                chunk = vol[s, g * group:(g + 1) * group]
                if chunk.any():
                    windows.append(chunk)
                    slots.append(s)
    else:
        assert isinstance(stride, int) and stride >= 1, f"stride={stride!r}"
        half = group // 2
        zero_slice = np.zeros(vol.shape[2:], dtype=vol.dtype)
        for s in range(vol.shape[0]):
            for i in range(0, n_slices, stride):
                chunk = np.stack([
                    vol[s, j] if 0 <= j < n_slices else zero_slice
                    for j in range(i - half, i - half + group)
                ])
                if chunk.any():
                    windows.append(chunk)
                    slots.append(s)
    if not windows:
        windows, slots = [np.zeros_like(vol[0, :group])], [0]
    x = torch.from_numpy(np.stack(windows)).float().div_(255.0)
    slots_t = torch.tensor(slots, dtype=torch.long)
    owner = torch.zeros(len(slots), dtype=torch.long)
    return x, slots_t, owner


def _tta_views(x, slots, tta: bool):
    """[x] or [x, flipped]. Sagittal-only flip (slots 0,1): a horizontal flip
    on coronal/axial slices mirrors medial/lateral anatomy and corrupts those
    labels. Do not "fix" this to a full-tensor flip. Shared by predict_study
    so predict_volume and predict_windows cannot diverge on TTA."""
    if not tta:
        return [x]
    x2 = x.clone()
    mask = slots < 2
    x2[mask] = x2[mask].flip(-1)
    return [x, x2]


def predict_study(models, vol, device, use_amp: bool, tta: bool = False,
                  window: int | None = None, stride: int | None = None):
    """One study volume -> (p_attn, win_probs, slots), running the backbone
    (Model.embed) exactly once per (model, TTA view) - predict_volume and
    predict_windows are thin wrappers over this so a study is never embedded
    twice per model.

    p_attn (12,): sigmoid of the study-level attention head (owner=zeros,
    n_studies=1), averaged over models AND TTA views in one flat mean (the
    same "for m in models: for v in views:" order predict_volume has always
    used) - a nested per-model-then-model mean is a different float sequence
    and would move the shipped submission's last decimal.

    win_probs (n_models, n_windows, 12) or None when window is None:
    sigmoid of the same h scored with each window ALONE (owner=arange,
    n_studies=n_windows: no cross-window attention), averaged over TTA views
    per model first, then stacked over models - the nested reduction
    predict_windows has always used, needed because callers pool per-model
    columns across windows themselves.

    slots: (n_rows,) numpy, from the shared batch build (study_batch when
    window is None, else study_windows(vol, window, stride)).

    stride: forwarded to study_windows (None = the non-overlapping
    training-time grouping; an int = dense windows every `stride` slices).
    Ignored when window is None."""
    import numpy as np
    if window is None:
        x, slots, owner = study_batch(vol, device)
    else:
        x, slots, owner = study_windows(vol, window, stride)
        x, slots, owner = x.to(device), slots.to(device), owner.to(device)
    views = _tta_views(x, slots, tta)
    n_rows = x.shape[0]
    win_owner = torch.arange(n_rows, device=device) if window is not None else None
    win_probs = None
    with torch.no_grad(), torch.amp.autocast(device, enabled=use_amp):
        attn_flat = []  # flat "for m: for v:" order, matches predict_volume today
        per_model_windows = [] if window is not None else None
        for m in models:
            view_windows = [] if window is not None else None
            for v in views:
                h = m.embed(v, slots)
                attn_flat.append(torch.sigmoid(m.head(h, owner, 1)).float())
                if window is not None:
                    view_windows.append(
                        torch.sigmoid(m.head(h, win_owner, n_rows)).float())
            if window is not None:
                per_model_windows.append(torch.stack(view_windows).mean(dim=0))
        p_attn = torch.stack(attn_flat).mean(dim=0).cpu().numpy()[0]
        if window is not None:
            win_probs = (torch.stack(per_model_windows).cpu().numpy()
                        .astype(np.float32))
    return p_attn, win_probs, slots.cpu().numpy()


def predict_windows(models, vol, device, use_amp: bool, tta: bool = False,
                    window: int = 3, stride: int | None = None):
    """Per-window sigmoid probs, each window scored ALONE (owner=arange,
    n_studies=n_windows: no attention pooling across windows happens here -
    callers pool across windows themselves). Same sagittal-only per-window
    TTA flip as predict_volume, averaged within each model. Thin wrapper over
    predict_study; shared by train_dinov2.gold_window_probs and
    infer_submit's window pooling so the two computations cannot drift apart.

    Returns (probs (n_models, n_windows, 12) float32 numpy, slots
    (n_windows,) numpy)."""
    _, win_probs, slots = predict_study(models, vol, device, use_amp,
                                        tta=tta, window=window, stride=stride)
    return win_probs, slots


def predict_volume(models, vol, device, use_amp: bool, tta: bool = False,
                   window: int | None = None, stride: int | None = None):
    """One study volume -> mean sigmoid probs over `models` (and over the
    horizontally flipped view when tta). The single rule both the training
    kernel (GOLD-58 measurement) and the submission kernel apply, so the
    measured TTA verdict is about the code path that ships. Thin wrapper over
    predict_study.

    window=None (default) is the stage A decode path (study_batch); this is
    NOT byte-for-byte unchanged for every stage-A-era checkpoint - weights-v3
    and earlier were trained before this per-label attention head existed
    and load only against src/model.py at commit 70c1954 or earlier.
    window=<int> is the stage B 2.5D path: study_windows(vol, window) builds
    the (N,3,H,W) batch. The TTA flip is sagittal-only (slots 0,1) since
    round 4; flip(-1) is the last spatial axis either way."""
    p_attn, _, _ = predict_study(models, vol, device, use_amp, tta=tta,
                                 window=window, stride=stride)
    return p_attn


def _pool_mean(probs, axis):
    return np.nanmean(probs, axis=axis)


def _pool_max(probs, axis):
    return np.nanmax(probs, axis=axis)


def _pool_top2(probs, axis):
    """Mean of the two largest values along `axis`, NaN-tolerant (the single
    real value when only one window is real)."""
    filled = np.where(np.isnan(probs), -np.inf, probs)
    sorted_ = np.sort(filled, axis=axis)
    idx = [slice(None)] * sorted_.ndim
    idx[axis] = slice(-2, None)
    top2 = sorted_[tuple(idx)]
    top2 = np.where(np.isfinite(top2), top2, np.nan)
    return np.nanmean(top2, axis=axis)


# window pooling modes, shared by tune_pooling.py (gold-58 AUC sweep, axis=2
# over (n_gold, n_folds, n_windows, 12)) and infer_submit.pool_windows
# (axis=1 over (n_models, n_windows, 12)) so the two cannot drift apart.
POOLERS = {"mean": _pool_mean, "max": _pool_max, "top2": _pool_top2}
