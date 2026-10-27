# Spec: `spatial_pool="label_attn"` - per-label patch-grid attention for the CoAtNet-2 arm (arm `coatnet2_pa`)

Repo: `C:\Users\Windows-11\Claude\Projects\Kaggle\rsna-67knee\` (git top-level is the parent `C:\Users\Windows-11\Claude\Projects\Kaggle`, so git paths are `rsna-67knee/...`). Written 2026-10-01 (planner pass; saved verbatim by the main session).

## 0. Measured facts the design rests on (all verified this session unless marked)

- timm 1.0.30, torch 2.11.0+cu128 (`py -c` on this PC).
- `Model.embed` already zero-pads the cached 280 px window to 288 px (`self.pad = 8`, lo=4/hi=4) because the arch spec has stride 32. `backbone.forward_features` at 288 px returns **(N, 1024, 9, 9) = 81 patch tokens**. Feeding a raw 280 px tensor raises `RuntimeError: The size of tensor a (289) must match the size of tensor b (324)` inside the relative-position MLP attention, so the pad is mandatory and the feature map carries ~0.25 patch of zero-padding on each border. Nothing new needed: the option reuses the padded tensor.
- `MaxxVit.forward_features` = `stem -> stages -> self.norm`; `backbone.norm` is `LayerNorm2d(1024, eps=1e-6, affine)` (pretrained). The timm head of both the 224 and 384 variants is a plain `ClassifierHead(global_pool=avg, drop, fc=Identity, flatten)` - **there is no `head.norm` on this timm version**, so the competitor's "pretrained head norm" is, for us, `backbone.norm` applied to every token inside `forward_features`. The current GAP path `self.backbone(x)` = `head(forward_features(x))`; measured `forward_features(x).mean((2,3))` vs `backbone(x)`: max abs diff **0.0** on CPU fp32.
- Mean token L2 norm after `backbone.norm` at random init: 31.8 (~sqrt(1024)), relevant for the `/sqrt(D)` scaling choice below.
- `set_grad_checkpointing(True)` only sets `stage.grad_checkpointing` flags; `forward_features` runs the same `self.stages`, so checkpointing and the `_mark_grad` hook on `stages[0]` work unchanged.
- Current head parameters (option off): `query (12,256)`, `out_weight (12,256)`, `out_bias (12,)`, `proj.0.{weight,bias} (1024,)x2`, `proj.1.weight (256,1024)`, `proj.1.bias (256,)`, `slot_emb.weight (6,256)` = **8 tensors, 272,140 params**; backbone 72,856,264. The train log prints `8 head params`.
- Batch geometry (train): 12 studies x up to 6 slots x `WINDOWS_PER_SLOT=2` = up to 144 windows, padded to a multiple of `WINDOW_BUCKET=16` (owner -1 rows are skipped by `head`).
- No pytest on this PC. Existing test convention: a plain script `src/test_geometry_v4g.py`.
- Local launch pattern (verified from `C:\kaggle\staging\local-coatnet-t4-f0\train_dinov2.py` diff vs repo): `NUM_WORKERS = 0`, `MAX_HOURS = 24.0`, `FOLDS_TO_TRAIN = (0,)`, `OUT = "C:/kaggle/working-coatnet-t4-f0"`, run with `HF_HUB_OFFLINE=1 py train_dinov2.py > train.log 2>&1` from the staging dir; the HF cache holds `models--timm--coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`. The script finds `C:\kaggle\input\rsna-knee-src` via its `/kaggle/input/rsna-knee-src` candidate (drive-relative path), and that mirror's `model.py` is currently byte-identical to the repo's.
- `C:\kaggle\input` holds only the v4 cache parts (`rsna-knee-build-cache`, `-b`), the labels (`consensus_v1.csv` present), `rsna-knee-src`. No v4g cache yet.
- weights-rs (`C:\kaggle\staging\weights-rs`): 5 folds, seed 7, `fold_auc_last` fold 0 = 0.8743 (stride 3), `oof_stride1_eval.npy` present (all-fold dense macro 0.8778; dense eval of all 4349 studies took 8.9 min on the 5090). Epoch time locally 156-182 s -> fold 0 of 20 epochs ~55-60 min.
- GPU at writing time: 18.9 GB / 32.6 GB used, 86 % util by another training job. Memory numbers below are analytic and flagged.

## 1. Approach and why

Add one switch, `spatial_pool` (`"gap"` default = today's code, `"label_attn"` = the option), to `Model`. With the option on, `embed` runs `backbone.forward_features` (N,1024,9,9), applies the competitor's per-label spatial reader (LN'd tokens, per-label 1024-d query, `2*tanh(s/2)` cap, `0.85*attn + 0.15*GAP`) to get 12 per-label 1024-d tokens per window, then pushes each through the **existing shared** `proj` (LN(1024) -> Linear 1024->256 -> GELU) and adds the slot embedding, returning **(N, 12, 256)** instead of (N, 256). `head` gains a 3-D branch that uses label `l`'s own window feature with query `l`. Only one new parameter tensor (`spatial_query`, 12x1024, zero-init) is added, so at init the option-on model computes exactly today's GAP model (uniform attention -> `v_l = g`), i.e. the experiment measures only what label-specific spatial reading adds, with no change to the window head's width, init or lr.

Rejected: 1024-d per-label window features with a 1024-wide (or gated-MIL) window head, as in the reference. Memory does not decide it - per-label tokens at batch 12 are 144 x 12 x 1024 x 2 B = 3.5 MB fp16 either way - but it would change two things at once (spatial reader + window-head width/init/params, +2.1 M params if the gated MLP is copied), lose the zero-init identity with `coatnet2_rs`, and need its own 1024-wide GAP control to attribute the effect. Also rejected: applying the pretrained norm after pooling as the reference does - on this timm build the norm sits inside `forward_features`, the current GAP already pools post-norm tokens, and keeping that preserves option-off/zero-query equivalence (stated deviation, see 2.1).

## 2. Equations (option on), per window n

Tokens `t_p in R^1024`, p = 1..81, from `forward_features` (already `backbone.norm`'d, pretrained affine LN).

```
t~_p   = LayerNorm(t_p)                       # no affine, eps 1e-5 (F.layer_norm default), 0 params
s_lp   = <q_l, t~_p> / sqrt(1024)             # q_l = spatial_query[l], l = 1..12 (reference 2.2 keeps the sqrt; FSX dropped it - we keep it)
s^_lp  = 2 * tanh(s_lp / 2)                   # cap 2.0
a_lp   = softmax_p(s^_lp)
g      = mean_p t_p                           # == today's GAP vector (verified identical to backbone(x))
v_l    = g + 0.85 * (sum_p a_lp t_p - g)      # (12, 1024)
h_l    = GELU(W proj LN_affine(v_l) + b) + slot_emb[slot_n]      # existing proj, shared across l -> (12, 256)
```
`embed` returns h of shape (N, 12, 256). Per-label window attention for study i with windows S:
```
e_ls   = <h_{s,l}, query_l> / sqrt(256);  alpha_ls = softmax_s(e_ls)
ctx_l  = sum_s alpha_ls h_{s,l}                                   # (12, 256)
logit_l = <Dropout_0.2(ctx_l), out_weight_l> + out_bias_l          # unchanged
```
Zero-init identity: `q_l = 0` -> `a_lp = 1/81` -> `v_l = g` -> `h_l = proj(g) + slot_emb` for every l -> `ctx` and logits equal today's model (up to fp rounding of the uniform-weighted sum vs `.mean`; assert `allclose(atol=1e-5)`, not `equal`). Gradient still reaches `q_l` at zero init (softmax Jacobian at uniform is non-zero when tokens differ).

Parameter counts: +12,288 (`spatial_query` 12x1024). Head 272,140 -> 284,428 (9 tensors); backbone unchanged 72,856,264; total 73,140,692. The train log line becomes `... 9 head params (lr=0.001), 382 trainable total` (was 8 / 381).

Memory (analytic, at N=144 windows, fp16 autocast, grad ckpt; baseline peak 7.1 GiB is the round-6 measurement on a T4 from memory notes, not re-measured here): new tensors kept for backward - LN'd tokens (144,81,1024) fp32 under autocast ~48 MB, scores+softmax (144,12,81) fp32 ~1.1 MB, `v` (144,12,1024) fp16 3.5 MB, proj-LN out fp32 7 MB, Linear/GELU/slot-add outputs (1728,256) ~3 x 0.9 MB. Total ~65 MB ~0.06 GiB on top of the baseline - predicted peak ~7.2 GiB. Extra compute ~144x12x81x1024x2 ~0.3 GFLOP per batch, negligible against the backbone. **Verify with the SMOKE probe (Task 8a) which prints `peak X GiB`.**

## 3. Constraints that are fixed

- `spatial_pool` absent/None/`"gap"` must be byte-identical to today: same forward math, same `state_dict` keys (no new parameter registered), same RNG consumption in `__init__` (the new parameter is `torch.zeros`, registered last, only when the option is on).
- `HIDDEN=256`, `proj`, `slot_emb`, `query`, `out_weight`, dropout 0.2, `unfreeze="stem_stage0"`, drop_path 0.1, seed 7, random window starts, 20 epochs, batch 12, label file `consensus_v1.csv`, fold split on `SEED=42` - all as in `coatnet2_rs`.
- Option restricted to `family == "coatnet2"` (ViT returns 3-D tokens; ConvNeXt's `forward_features` is pre-norm so GAP-equivalence would not hold). Raise `ValueError` otherwise.
- Do not touch `src/preprocess.py`. Copy only `model.py` to the local src mirror.
- No commits unless the caller asks.

## 4. Out of scope

Gated tanh x sigmoid window MIL, slot-count normalisation, depth/stride metadata, the FSX slot stream / FiLM, BN recalibration / SWA, the 384 backbone, extending the option to ConvNeXt/ViT, regenerating `notebooks/fusion/fusion_submit.py` via `bundle.py` (needed only if a label_attn checkpoint ships through the fusion kernel), pushing the Kaggle `rsna-knee-src` dataset (needed only before a Kaggle kernel loads a label_attn checkpoint; the new `model.py` is backward compatible so the push is safe whenever it happens), docs beyond the ARMS comment and the plan log line.

## 5. Tasks

### Task 1 - `src/model.py`: the option
**Files**: `src/model.py`
**Change**:
1. Module constants after `HIDDEN = 256`: `SPATIAL_MIX = 0.85` and `SPATIAL_CAP = 2.0`, with a one-line comment pointing to `docs/competitors-code/mattiaangeli-d4-serving.md` section 2.2.
2. `Model.__init__` signature: add `spatial_pool: str | None = None` after `n_slots`. Right after `self.unfreeze = ...` set `self.spatial_pool = spatial_pool or "gap"`; `if self.spatial_pool not in ("gap", "label_attn"): raise ValueError(...)`. In the backbone (non-VLM) branch, after `self.out_bias = ...` (keep it last so existing init order is untouched):
   ```python
   if self.spatial_pool == "label_attn":
       if self.family != "coatnet2":
           raise ValueError(f"spatial_pool='label_attn' is implemented for the coatnet2 family only (4-D post-norm feature map), not {self.family!r}")
       # per-label spatial queries over the patch grid; zero-init = uniform attention = today's GAP model
       self.spatial_query = nn.Parameter(torch.zeros(len(LABELS), d))
   ```
   In the VLM branch (`self.mode == "vlm"`), raise `ValueError` if `self.spatial_pool != "gap"` before `return`.
3. New method on `Model`:
   ```python
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
   ```
4. `embed`: replace the final `return self.proj(self.backbone(x)) + self.slot_emb(slots)` with
   ```python
   if self.spatial_pool == "label_attn":
       v = self._label_attn_pool(self.backbone.forward_features(x))      # (N, 12, D)
       return self.proj(v) + self.slot_emb(slots).unsqueeze(1)            # (N, 12, HIDDEN)
   return self.proj(self.backbone(x)) + self.slot_emb(slots)  # (N, HIDDEN)
   ```
   Update the docstring: `-> (N, HIDDEN), or (N, 12, HIDDEN) per-label window features when spatial_pool="label_attn"`.
5. `head`: inside the `for i in range(n_studies)` loop, after `hs = h[m]`, insert
   ```python
   if hs.dim() == 3:  # (S, 12, HIDDEN): label l attends over windows with its own feature (spatial_pool="label_attn")
       logits = torch.einsum("soh,oh->os", hs, self.query) / HIDDEN ** 0.5
       a = torch.softmax(logits, dim=-1)
       ctxs.append(torch.einsum("os,soh->oh", a, hs))
       continue
   ```
   and leave the existing 2-D lines textually unchanged. Docstring: note the 3-D form.
6. Class docstring / module header: one sentence that `spatial_pool` is carried in `run_config.json` and read back by train/eval/infer.
**Verification**: `cd rsna-67knee && py -c "import sys; sys.path.insert(0,'src'); import torch; from model import Model; torch.manual_seed(0); m=Model(pretrained=False,img_size=280,arch='coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k',unfreeze='stem_stage0',drop_path=0.1,n_slots=6,spatial_pool='label_attn').eval(); x=torch.rand(5,3,280,280); s=torch.tensor([0,1,2,3,0]); h=m.embed(x,s); print(h.shape, m.head(h,torch.tensor([0,0,0,1,1]),2).shape, m.head(h,torch.arange(5),5).shape, sum(p.numel() for n,p in m.named_parameters() if not n.startswith('backbone.')))"` prints `torch.Size([5, 12, 256]) torch.Size([2, 12]) torch.Size([5, 12]) 284428`. Task 2 is the full check.
**Depends on**: nothing.

### Task 2 - unit test `src/test_spatial_pool.py`
**Files**: `src/test_spatial_pool.py` (new; plain script with asserts; no pytest available).
**Change**: CPU-only script that
1. Materialises the pre-change model: `subprocess.check_output(["git","show","HEAD:rsna-67knee/src/model.py"], cwd=<git top-level>)` -> write to a temp dir -> `importlib.util.spec_from_file_location("model_head", path)` with `src/` on `sys.path` (so its `from preprocess import ...` resolves). Skip this half with a printed WARN if `git show` fails (the reference half is meaningful only while HEAD predates Task 1).
2. `KW = dict(pretrained=False, img_size=280, arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", grad_ckpt=False, unfreeze="stem_stage0", drop_path=0.1, n_slots=6)`. Build three models, each preceded by `torch.manual_seed(0)`: `old = model_head.Model(**KW)`, `off = model.Model(**KW)`, `on = model.Model(**KW, spatial_pool="label_attn")`; all `.eval()`.
3. Assert `old.state_dict().keys() == off.state_dict().keys()` and every tensor `torch.equal`. Assert `set(on.state_dict()) - set(off.state_dict()) == {"spatial_query"}`.
4. Input: `torch.manual_seed(1); x = torch.rand(5,3,280,280); slots = tensor([0,1,2,3,0]); owner = tensor([0,0,0,1,1]); n=2`. Assert `torch.equal(old(x,slots,owner,n), off(x,slots,owner,n))` (**byte-identical option off**). Also assert equality on the grey path `x[:,0]` (N,H,W).
5. Option on: `h = on.embed(x, slots)`; assert shape (5,12,256); `L = on.head(h, owner, 2)` shape (2,12) finite; `on.head(h, torch.arange(5), 5)` shape (5,12). Assert `torch.allclose(L, off(x,slots,owner,n), atol=1e-5)` (zero-query identity). Grey path with option on -> (5,12,256).
6. Gradient at zero init: `on.train(); L = on(x,slots,owner,n); L.sum().backward(); assert on.spatial_query.grad is not None and on.spatial_query.grad.abs().max() > 0`.
7. Non-trivial attention: `with torch.no_grad(): on.spatial_query.normal_(0, 0.5)`; `on.eval()`; assert `(on(x,slots,owner,n) - off(x,slots,owner,n)).abs().max() > 1e-4`.
8. Round trip: `sd = on.state_dict(); torch.manual_seed(0); on2 = model.Model(**KW, spatial_pool="label_attn"); on2.load_state_dict(sd, strict=True); on2.eval(); assert torch.equal(on2(x,slots,owner,n), on(x,slots,owner,n))`.
9. Negative: `model.Model(**KW, spatial_pool="bogus")` raises ValueError; `model.Model(pretrained=False, img_size=224, spatial_pool="label_attn")` (ViT default arch) raises ValueError.
10. Print `PASS test_spatial_pool`.
**Verification**: `cd rsna-67knee && py src/test_spatial_pool.py` ends with `PASS test_spatial_pool` (CPU, ~1-2 min).
**Depends on**: Task 1. Run it BEFORE committing Task 1 so `git show HEAD:` is the pre-change file.

### Task 3 - train script: arm row, switch, run_config
**Files**: `notebooks/train/train_dinov2.py`.
**Change**:
1. ARMS, directly after the `coatnet2_rs2` row (before the `coatnet2_v4g` comment block):
   ```python
   # coatnet2_pa: coatnet2_rs unchanged (seed 7, random window starts) + spatial_pool="label_attn":
   # per-label attention over the 9x9 patch grid (LN'd tokens, cap-2 tanh, 0.85 attn + 0.15 GAP;
   # docs/competitors-code/mattiaangeli-d4-serving.md 2.2) replaces GAP before the shared 1024->256
   # projection; zero-init queries = the rs model at step 0. One factor vs coatnet2_rs: paired fold-0 dense OOF.
   "coatnet2_pa": dict(arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
                       family="coatnet2", batch=12, epochs=20,
                       folds_to_train=(0, 1, 2, 3, 4), lr_backbone=5e-5,
                       lr_head=1e-3, grad_ckpt=True,
                       unfreeze="stem_stage0", warmup_epochs=1, drop_path=0.1,
                       pos_weight=False, swa_last=0, full_data=False,
                       window_starts="random", train_seed=7, spatial_pool="label_attn"),
   ```
2. After `TRAIN_SEED = _arm.get("train_seed", SEED)` add `SPATIAL_POOL = _arm.get("spatial_pool", "gap")  # "gap" (GAP -> proj, every arm so far) | "label_attn" (model.py per-label patch attention)`.
3. `Model(img_size=CACHE_CFG["size"], arch=ARCH, grad_ckpt=GRAD_CKPT, unfreeze=UNFREEZE, drop_path=DROP_PATH, n_slots=N_SLOTS_CFG, spatial_pool=SPATIAL_POOL)`.
4. `run_cfg`: add `"spatial_pool": SPATIAL_POOL,` right after `"train_seed": TRAIN_SEED,`.
**Verification**: `py -c "import ast,sys; ast.parse(open('rsna-67knee/notebooks/train/train_dinov2.py',encoding='utf-8').read()); print('ok')"` and `grep -n "SPATIAL_POOL\|coatnet2_pa" rsna-67knee/notebooks/train/train_dinov2.py` shows the 5 sites.
**Depends on**: Task 1.

### Task 4 - `tools/stage_train.py`: arm registered
**Change**: add `"coatnet2_pa": "v4"` to `ARM_CACHE` right after `"coatnet2_rs2": "v4"`.
**Verification**: `py rsna-67knee/tools/stage_train.py --user patimakornsrinamen --arm coatnet2_pa --folds 0 --out C:/kaggle/staging/local-coatnet-pa-f0` prints `ARM=coatnet2_pa FOLDS_TO_TRAIN=(0,) ... cache=v4` (this also performs the staging for Task 7).
**Depends on**: Task 3.

### Task 5 - consumers read the switch back
**Files**: `src/eval_windows.py` (line ~247), `notebooks/infer/infer_submit.py` (line ~258).
**Change**: eval_windows: `Model(pretrained=False, img_size=img_size, arch=arch, grad_ckpt=False, unfreeze=unfreeze, drop_path=drop_path, n_slots=None, spatial_pool=run_cfg.get("spatial_pool"))`. infer_submit `load_family`: `Model(pretrained=False, img_size=cfg["size"], arch=cfg.get("arch"), n_slots=cfg.get("n_slots"), spatial_pool=cfg.get("spatial_pool"))`. Nothing else.
**Verification**: `cd rsna-67knee && py src/eval_windows.py --folds 0 --limit 5 --weights C:/kaggle/staging/weights-rs --labels C:/kaggle/input/rsna-knee-consensus-labels/consensus_v1.csv --strides 3 --device cpu` runs to completion on an old run_config (no `spatial_pool` key -> gap) and reports `stride-3 vs saved oof.npy: ... max abs diff` below 0.002. infer_submit: `py -c "import ast; ast.parse(open('rsna-67knee/notebooks/infer/infer_submit.py',encoding='utf-8').read()); print('ok')"` plus a grep for `spatial_pool=cfg.get`.
**Depends on**: Task 1.

### Task 6 - local src mirror
**Change**: `cp C:/Users/Windows-11/Claude/Projects/Kaggle/rsna-67knee/src/model.py C:/kaggle/input/rsna-knee-src/model.py` - model.py ONLY.
**Verification**: `diff` prints nothing; `grep -c label_attn C:/kaggle/input/rsna-knee-src/model.py` > 0.
**Depends on**: Tasks 1-2 passing.

### Task 7 - stage the fold-0 run
**Change** (from `C:\Users\Windows-11\Claude\Projects\Kaggle`):
```
py rsna-67knee/tools/stage_train.py --user patimakornsrinamen --arm coatnet2_pa --folds 0 --out C:/kaggle/staging/local-coatnet-pa-f0
sed -i 's/^NUM_WORKERS = 2$/NUM_WORKERS = 0/; s#^OUT = "/kaggle/working"#OUT = "C:/kaggle/working-coatnet-pa-f0"#; s/^MAX_HOURS = 8.4$/MAX_HOURS = 24.0/' C:/kaggle/staging/local-coatnet-pa-f0/train_dinov2.py
```
LABEL_FILE stays the repo default `consensus_v1.csv`. The staged copy carries `CACHE_VERSION = "v4"`.
**Verification**: `grep -n "^ARM = \|^FOLDS_TO_TRAIN\|^NUM_WORKERS\|^OUT = \|^MAX_HOURS\|^LABEL_FILE\|^CACHE_VERSION\|^SMOKE" C:/kaggle/staging/local-coatnet-pa-f0/train_dinov2.py` shows `ARM = "coatnet2_pa"`, `FOLDS_TO_TRAIN = (0,)`, `NUM_WORKERS = 0`, `OUT = "C:/kaggle/working-coatnet-pa-f0"`, `MAX_HOURS = 24.0`, `LABEL_FILE = "consensus_v1.csv"`, `SMOKE = 0`.
**Depends on**: Tasks 3, 4, 6.

### Task 8a - SMOKE memory/throughput probe (3 min)
Stage a second copy with `--smoke 120 --out C:/kaggle/staging/local-coatnet-pa-smoke`, same sed with `OUT = "C:/kaggle/working-coatnet-pa-smoke"`, then - only when `nvidia-smi` shows no other training job - `cd C:/kaggle/staging/local-coatnet-pa-smoke && HF_HUB_OFFLINE=1 py train_dinov2.py > train.log 2>&1`.
**Verification**: `train.log` contains `optimizer groups: 373 backbone params (lr=5e-05), 9 head params (lr=0.001), 382 trainable total`, no `no gradient reached` assertion, and a `SMOKE: 120 studies in ...s = X studies/s, peak Y GiB` line with Y <= 8 GiB. Record Y in the plan log.

### Task 8b - the fold-0 run
`cd C:/kaggle/staging/local-coatnet-pa-f0 && HF_HUB_OFFLINE=1 py train_dinov2.py > train.log 2>&1` (GPU free first). Expected ~1 h.
**Verification**: `head -12 train.log` shows the two v4 cache parts, `CACHE_CFG: {... 'version': 'v4' ...}`, `labels: ...consensus_v1.csv (weighted)`, `9 head params ... 382 trainable total`; the log ends with `fold 0 macro AUC vs LLM labels (last): 0.8xxx`, `GOLD-58 ...`; `C:/kaggle/working-coatnet-pa-f0/` holds `coatnet2_fold0.pt`, `oof.npy`, `fold_of.json`, `run_config.json` (with `"spatial_pool": "label_attn"`), `gold_windows.npz`; `torch.load(...)['spatial_query'].abs().max() > 0`.

### Task 9 - dense OOF and the paired judgement vs weights-rs fold 0
```
py src/eval_windows.py --folds 0 --weights C:/kaggle/working-coatnet-pa-f0 --labels C:/kaggle/input/rsna-knee-consensus-labels/consensus_v1.csv --strides 3,1 --out C:/kaggle/working-coatnet-pa-f0/eval_windows.md
```
then the paired bootstrap on fold-0 rows vs `C:/kaggle/staging/weights-rs/oof_stride1_eval.npy` (assert `fold_of.json` equality). If the v4g cache lands under `C:/kaggle/input` before this step, pass `--cache-root` or move it aside (eval_windows asserts on differing cache_config).
**Decision rule** (docs/methodology.md: paired single-fold bar 0.004): mean >= +0.004 -> queue folds 1-4 and the 5-fold paired read; 0 < mean < 0.004 -> inconclusive (seed-11 replicate `coatnet2_pa2` is the tie-breaker); mean <= 0 -> drop, log the number.

## 6. What `run_config.json` must record
`"spatial_pool": "label_attn"` (new key; older configs lack it -> every consumer's `cfg.get("spatial_pool")` -> None -> gap). Everything else inherited from `coatnet2_rs`. The checkpoint gains the key `spatial_query` (12,1024); `load_state_dict(strict=True)` on a model built with the switch from run_config loads it; a model built without the switch would fail strict loading - the intended guard against mixing up checkpoints.

## 7. Unverified / flagged
- Memory and throughput of the option are analytic (Task 8a measures them).
- Reference deviation: the competitor applies the pretrained norm after pooling and then its own LN; we pool tokens that are already `backbone.norm`'d and rely on `proj[0]` as the own LN. Chosen for exact option-off equivalence.
- The reference's `q_l` init is unknown; zero init is our choice.
- `/sqrt(1024)` scaling follows the Global96 baseline reader; the FSX variant removed it. With zero init the queries need norm ~2-4 to reach the cap; at lr_head 1e-3 that is reachable but the attention could stay soft for the first epochs - a risk, not a bug.
