# notebooks/fusion/fusion_submit.py
#
# RSNA Knee, round 7 task 7 (A2): our own fusion kernel, built from
# self-contained public arms plus our two arms, rank-blended with the
# public "the-bee-s-knees-final-rsna-push" recipe's weights (prvsiyan,
# kernel prvsiyan/the-bee-s-knees-final-rsna-push, pulled 2026-09-09,
# SHA-256 6b03919bbca5fe61e37c5ea6546609a771cd7db6503121f3e2dacdd8c9b6b052).
#
# Attribution for the bundled public arms (see raptor_infer.py and
# resgated_runner.py's own headers for the full licence/checkpoint detail):
#   Raptor CoAtNet v5 -- dreaddevelopment (Johnathan Wagner), dataset
#     `raptor-knee-maxspan`, CC0-1.0 weights.
#   Residual-gated CoAtNet -- Mattia Angeli, dataset
#     `rsna-knee-coat-resgated-ep10-top3`, CC0-1.0 weights; offline opencv
#     wheel `opencv-python-headless-4120088-x86`, Apache-2.0.
#   Both assembled/reproduced (per the source kernel's own licence table)
#   from Mattia Angeli's V27 notebook, Apache-2.0.
#
# A Kaggle script kernel is a single code file, so the two task-6 helper
# scripts and infer_submit.py (none of the three lives in the rsna-knee-src
# dataset) are embedded below as string literals (_RAPTOR_SRC, _RESGATED_SRC,
# _INFER_SRC), regenerated from the standalone files by bundle.py -- do not
# hand-edit the BEGIN/END BUNDLED blocks. At runtime they are written out to
# /kaggle/working/ and run: raptor_infer.py and resgated_runner.py as
# sequential subprocesses (so the GPU is fully freed between them, per their
# own env contracts, documented in their headers); infer_submit.py is
# imported instead, so this kernel can call its load_family/decode/score
# path directly for our own arms (OUR_ARMS below, keyed by arm id -- v6c and
# pseudo1 are both coatnet2, trained on different label files, and
# _run_our_arms loads/scores them one at a time so they don't collide).
#
# fuse() is the pure blend math, driven entirely by the BLEND dict below:
# CoAt branch = rank_blend of BLEND["coat"]'s named arms (any mix of public
# and our own, e.g. Raptor/resgated/v6c/pseudo1); preset "full" additionally
# mixes in a transformer branch (BLEND["transformer"], v4b alone) per label,
# (1-w)*transformer_rank + w*coat_rank, w=BLEND["w_coat"]
# (w_coat_lateral_meniscus for Lateral Meniscus), re-ranked; preset
# "coat_only" is the CoAt branch alone, re-ranked, no transformer branch;
# preset "public_only" drops our own arms entirely, using
# BLEND["public_coat"] (Raptor 0.60 / resgated 0.40, the
# submission_public_only.csv weights, unchanged). The legacy "full" weights
# (0.375 Raptor / 0.375 v6c / 0.25 resgated, w 0.60/1.00) are the public
# kernel's own blend recipe (user, 2026-09-09), reused as-is; gold-58 stays
# untouched by them. A staged copy of this file flips ONE dict (BLEND) to
# change preset or weights for an LB submission.
#
# BLEND["preset"] selects which variant is ALSO copied into the literal
# submission.csv (the file Kaggle grades) -- both submission.csv (the
# active-preset variant) and submission_public_only.csv are always written,
# so a run under preset "public_only" still produces both files, just with
# submission.csv holding the public-only blend (the "LB probe without our
# arms" run named in the round-7 decision), and never loads any of our own
# arms (see _required_our_arm_ids).

import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

_SRC_CANDIDATES = (
    "/kaggle/input/rsna-67knee-src",
    "/kaggle/input/datasets/patimakornsrinamen/rsna-67knee-src",
    # pre-rename slug, kept so this kernel runs against either name
    "/kaggle/input/rsna-knee-src",
    "/kaggle/input/datasets/patimakornsrinamen/rsna-knee-src",
)
for _src in _SRC_CANDIDATES:
    if os.path.exists(_src):
        sys.path.insert(0, _src)
        break
else:
    raise FileNotFoundError(f"rsna-knee-src not mounted; tried {_SRC_CANDIDATES}")
from preprocess import LABELS
from blend_eval import pct_rank, rank_blend

WORKING = "/kaggle/working"
OUT = "/kaggle/working/submission.csv"
PUBLIC_ONLY_OUT = "/kaggle/working/submission_public_only.csv"
RAPTOR_SCRIPT = os.path.join(WORKING, "raptor_infer.py")
RESGATED_SCRIPT = os.path.join(WORKING, "resgated_runner.py")
INFER_SUBMIT_SCRIPT = os.path.join(WORKING, "infer_submit.py")
RAPTOR_CSV = "/kaggle/working/_raptor.csv"
# Raptor sub-arms to run. The verbatim script defaults to all four (maxspan-v5,
# native384dense-v10, maxspan-v5-reverse, native384-v8), which need three
# weight datasets; this kernel attaches only `raptor-knee-maxspan` and runs
# the single 0.914-0.924 arm the plan settled on (v1 of this kernel died on
# the missing v10 checkpoint).
RAPTOR_ARM_NAMES = "maxspan-v5"
RESGATED_CSV = "/kaggle/working/_resgated.csv"

# BEGIN BUNDLED RAPTOR
_RAPTOR_SRC = '# notebooks/fusion/raptor_infer.py\n#\n# Standalone extraction of the `_KE_SRC` cell from the public kernel\n# "the-bee-s-knees-final-rsna-push" (Kaggle kernel\n# prvsiyan/the-bee-s-knees-final-rsna-push, pulled to\n# C:/kaggle/staging/competitors/the-bee-s-knees-final-rsna-push/ on 2026-09-09).\n# Source authenticated by the pulled notebook\'s own SHA-256\n# 6b03919bbca5fe61e37c5ea6546609a771cd7db6503121f3e2dacdd8c9b6b052.\n#\n# Original Raptor CoAtNet inference recipe and weights: dreaddevelopment\n# (Johnathan Wagner), datasets `raptor-knee-maxspan`, `raptor-knee-native384`,\n# `raptor-knee-native384dense`, `raptor-knee-arms` -- all four CC0-1.0.\n# Assembled/reproduced (per the source kernel\'s licence table) from Mattia\n# Angeli\'s V27 notebook, Apache-2.0. This file keeps that licence for the\n# reproduced code; the CC0-1.0 terms above cover the checkpoint weights only.\n#\n# Everything between the BEGIN/END sentinels below is byte-identical to the\n# `_KE_SRC` string literal in the source notebook\'s cell 32 (verified by\n# diffing this block against `ast.literal_eval()` of that assignment) -- do\n# not edit it. It defines Raptor\'s own\n# 4-sub-arm (maxspan-v5 / native384dense-v10 / maxspan-v5-reverse /\n# native384-v8) rank blend and writes `/kaggle/working/_raptor.csv` in\n# `sample_submission` column order; no cross-arm fusion (Raptor vs. our arms\n# vs. resgated) happens here -- that lives in `fusion_submit.py`.\n#\n# `find_test_root`/`find_weight_file` inside the verbatim body already walk\n# all of /kaggle/input recursively, so they resolve both the flat\n# /kaggle/input/<name>/... mount layout and the nested\n# /kaggle/input/datasets/<owner>/<name>/... layout without needing any\n# changes here.\n#\n# Argument/env contract (all optional; defaults reproduce the source kernel\'s\n# hardcoded behaviour exactly):\n#   RAPTOR_ARM_NAMES   comma-separated subset of the ARMS[*]["name"] values to\n#                      run, e.g. "maxspan-v5,native384-v8" (default: all four)\n#   RAPTOR_CKPT_DIRS   os.pathsep-separated extra directories searched for a\n#                      checkpoint file before the built-in\n#                      /kaggle/input/raptor-knee-* search list\n#   RAPTOR_OUTPUT_CSV  output path (default: /kaggle/working/_raptor.csv, the\n#                      path the source kernel hardcodes)\n# No other parameterisation is added: the slot layout, window sampling, and\n# per-arm blend weights are exactly as in the source cell.\n#\n# Run standalone: `python raptor_infer.py` (writes _raptor.csv and exits;\n# weights absent -> RuntimeError naming the missing checkpoint file).\n# Import-safe: `import raptor_infer` defines everything below without\n# running main().\n\n# ==== BEGIN VERBATIM _KE_SRC (source notebook cell 32) ====\nimport os, glob, time, gc, hashlib\nos.environ.setdefault(\'HF_HUB_OFFLINE\', \'1\')\nos.environ.setdefault(\'TRANSFORMERS_OFFLINE\', \'1\')\nos.environ.setdefault(\'HF_HUB_DISABLE_TELEMETRY\', \'1\')\nimport numpy as np\nimport torch, torch.nn as nn, torch.nn.functional as F\nimport timm\ntorch.backends.cudnn.benchmark = True\ntorch.backends.cuda.matmul.allow_tf32 = True\nIMG = 336\nCROP_MM = 140.0\nSPAN_LO, SPAN_HI = 0.02, 0.98\nSLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14),\n         ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]\nMAXS = sum(slot[2] for slot in SLOTS)\nK_EVAL = 62\nNORM = "imagenet"\nLAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",\n       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker\'s",\n       "Contusion", "Fracture"]\n_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)\n_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)\n_SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14),\n            ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]\n_SLOTS44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10),\n            ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]\nARMS = [\n    {"name": "maxspan-v5", "file": "raptor_ft_coatnet_v5_full_swa.pt",\n     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,\n     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,\n     "reverse": False, "w": 0.60},\n    {"name": "native384dense-v10", "file": "raptor_ft_coatnet_v10_full.pt",\n     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,\n     "img": 384, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,\n     "reverse": False, "w": 0.10},\n    {"name": "maxspan-v5-reverse", "file": "raptor_ft_coatnet_v5_full_swa.pt",\n     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,\n     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,\n     "reverse": True, "w": 0.10},\n    {"name": "native384-v8", "file": "raptor_ft_coatnet_v8_full_swa.pt",\n     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,\n     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 42,\n     "reverse": False, "w": 0.20},\n]\n\ndef build_backbone(arch, pretrained=False):\n    hybrid = arch.startswith((\'maxvit\', \'maxxvit\', \'coatnet\', \'coat_\', \'convnext\'))\n    is_vit = not hybrid and any((k in arch for k in (\'vit\', \'deit\', \'dinov2\', \'eva\', \'beit\')))\n    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)\n    if is_vit:\n        kw.update(global_pool=\'token\', dynamic_img_size=True)\n    else:\n        kw.update(global_pool=\'avg\')\n    return timm.create_model(arch, **kw)\n\nclass RaptorClassifier(nn.Module):\n\n    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):\n        super().__init__()\n        self.backbone = backbone\n        self.norm = nn.LayerNorm(F_dim)\n        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))\n        self.clsW = nn.Parameter(torch.zeros(n, F_dim))\n        self.clsb = nn.Parameter(torch.zeros(n))\n        nn.init.trunc_normal_(self.clsW, std=0.02)\n        self.n = n\n\n    def encode(self, x):\n        B, K = x.shape[:2]\n        f = self.backbone(x.flatten(0, 1))\n        return f.view(B, K, -1)\n\n    def head(self, feats):\n        h = self.norm(feats)\n        a = self.att(h)\n        a = torch.softmax(a, dim=1)\n        pooled = torch.einsum(\'bkn,bkf->bnf\', a, h)\n        logits = (pooled * self.clsW).sum(-1) + self.clsb\n        return logits\n\n    def forward(self, x):\n        return self.head(self.encode(x))\n\ndef load_model(pt_path, arch_default, res_default, device, ngpu=1):\n    ck = torch.load(pt_path, map_location=\'cpu\', weights_only=False)\n    arch = ck.get(\'arch\', arch_default)\n    ck_res = int(ck.get(\'res\', res_default))\n    bb = build_backbone(arch, pretrained=False)\n    model = RaptorClassifier(bb, F_dim=bb.num_features)\n    model.load_state_dict(ck[\'model\'], strict=True)\n    model.eval().to(device)\n    del ck\n    gc.collect()\n    return (model, ck_res)\n\ndef load_refit_head(pt_path, feature_dim, device):\n    ck = torch.load(pt_path, map_location=\'cpu\', weights_only=False)\n    state = ck.get(\'model\', ck)\n    head = RaptorClassifier(nn.Identity(), F_dim=int(feature_dim))\n    head_state = {name: tensor for name, tensor in state.items()\n                  if not name.startswith(\'backbone.\')}\n    head.load_state_dict(head_state, strict=True)\n    head.eval().to(device)\n    del ck, state, head_state\n    gc.collect()\n    return head\n\ndef _eval_centers(mask, D, k):\n    valid = np.where(mask > 0)[0]\n    if len(valid) < 3:\n        valid = np.arange(min(3, D))\n    lo, hi = (int(valid.min()), int(valid.max()))\n    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]\n    if not cs:\n        cs = [max(1, min((lo + hi) // 2, D - 2))]\n    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)\n    return [cs[i] for i in idx]\n\ndef eval_windows(vol, mask, k, res, norm=NORM):\n    D = vol.shape[0]\n    cs = _eval_centers(mask, D, k)\n    wins = np.empty((len(cs), 3, res, res), np.float32)\n    for j, c in enumerate(cs):\n        c = max(1, min(c, D - 2))\n        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0\n        t = torch.from_numpy(tri)\n        if t.shape[-1] != res:\n            t = F.interpolate(t[None], size=(res, res), mode=\'bilinear\', align_corners=False)[0]\n        wins[j] = t.numpy()\n    x = torch.from_numpy(wins)\n    if norm == \'imagenet\':\n        x = (x - _MEAN) / _STD\n    return x\n\n@torch.no_grad()\ndef infer_probs(model, xwins, device):\n    x = xwins.unsqueeze(0).to(device)\n    use_cuda = device != \'cpu\' and str(device).startswith(\'cuda\')\n    if use_cuda:\n        try:\n            with torch.autocast(\'cuda\', dtype=torch.float16):\n                o = torch.sigmoid(model(x).float())\n            return o[0].cpu().numpy()\n        except RuntimeError:\n            torch.cuda.empty_cache()\n            o = torch.sigmoid(model(x).float())\n            return o[0].cpu().numpy()\n    o = torch.sigmoid(model(x).float())\n    return o[0].cpu().numpy()\n\n@torch.no_grad()\ndef infer_probs_two_heads(model, refit_head, xwins, device):\n    x = xwins.unsqueeze(0).to(device)\n    use_cuda = device != \'cpu\' and str(device).startswith(\'cuda\')\n    def forward_heads():\n        features = model.encode(x)\n        original = torch.sigmoid(model.head(features).float())\n        refitted = torch.sigmoid(refit_head.head(features).float())\n        return (original[0].cpu().numpy(), refitted[0].cpu().numpy())\n    if use_cuda:\n        try:\n            with torch.autocast(\'cuda\', dtype=torch.float16):\n                return forward_heads()\n        except RuntimeError:\n            torch.cuda.empty_cache()\n            return forward_heads()\n    return forward_heads()\n\ndef rankpct(x):\n    order = x.argsort(0).argsort(0).astype(np.float64)\n    return order / max(1, x.shape[0] - 1)\n\ndef _make_reader():\n    import pydicom, cv2\n    from pydicom.pixel_data_handlers.util import apply_modality_lut\n\n    def order_and_meta(sdir):\n        fs = glob.glob(sdir + \'/*.dcm\')\n        recs = []\n        ps_list = []\n        for f in fs:\n            try:\n                h = pydicom.dcmread(f, stop_before_pixels=True)\n                iop = getattr(h, \'ImageOrientationPatient\', None)\n                ipp = getattr(h, \'ImagePositionPatient\', None)\n                if iop is not None and ipp is not None and (len(iop) == 6):\n                    r = np.array(iop[:3], float)\n                    c = np.array(iop[3:], float)\n                    n = np.cross(r, c)\n                    pos = float(np.dot(np.array(ipp, float), n))\n                else:\n                    pos = float(getattr(h, \'InstanceNumber\', 0) or 0)\n                ps = getattr(h, \'PixelSpacing\', None)\n                ps = float(ps[0]) if ps is not None else 0.5\n                ps_list.append(ps)\n                recs.append((pos, f, ps))\n            except Exception:\n                recs.append((0.0, f, 0.5))\n        recs.sort(key=lambda x: x[0])\n        med_ps = float(np.median(ps_list)) if ps_list else 0.5\n        return ([(f, ps) for _, f, ps in recs], med_ps)\n\n    def read_px(f):\n        d = pydicom.dcmread(f)\n        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)\n        if str(getattr(d, \'PhotometricInterpretation\', \'\')) == \'MONOCHROME1\':\n            a = a.max() - a\n        return a\n\n    def mm_crop_resize(a, ps):\n        h, w = a.shape\n        cpx = int(round(CROP_MM / max(ps, 0.001)))\n        cpx = min(cpx, min(h, w))\n        y0 = (h - cpx) // 2\n        x0 = (w - cpx) // 2\n        a = a[y0:y0 + cpx, x0:x0 + cpx]\n        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)\n    return (order_and_meta, read_px, mm_crop_resize)\n\ndef _pick_series_for_slot(rows, plane, fluid, used):\n    cands = [r for r in rows if r[\'Anatomical_Plane\'] == plane and r[\'SeriesInstanceUID\'] not in used]\n    if fluid in (0, 1):\n        pref = [r for r in cands if int(r.get(\'Fluid_Sensitive\', 0) or 0) == fluid]\n        if pref:\n            return pref[0]\n    return cands[0] if cands else None\n\ndef build_study(sid, ser_records, tsdir, reader):\n    order_and_meta, read_px, mm_crop_resize = reader\n    rows = ser_records.get(sid, [])\n    vol = np.zeros((MAXS, IMG, IMG), np.uint8)\n    idx = 0\n    used = set()\n    for plane, fluid, k in SLOTS:\n        r = _pick_series_for_slot(rows, plane, fluid, used)\n        if r is None:\n            idx += k\n            continue\n        used.add(r[\'SeriesInstanceUID\'])\n        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r[\'SeriesInstanceUID\']}")\n        if not files:\n            idx += k\n            continue\n        n = len(files)\n        lo, hi = (int(n * SPAN_LO), int(n * SPAN_HI) - 1)\n        hi = max(hi, lo)\n        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k\n        arrs = []\n        pss = []\n        for p in picks:\n            fp, ps = files[min(p, n - 1)]\n            try:\n                arrs.append(read_px(fp))\n                pss.append(ps)\n            except Exception:\n                arrs.append(None)\n                pss.append(med_ps)\n        valid = [a for a in arrs if a is not None]\n        if valid:\n            allpx = np.concatenate([a.ravel() for a in valid])\n            loq, hiq = np.percentile(allpx, [2.0, 98.0])\n        else:\n            loq, hiq = (0.0, 1.0)\n        for a, ps in zip(arrs, pss):\n            if idx >= MAXS:\n                break\n            if a is None:\n                idx += 1\n                continue\n            aw = np.clip((a - loq) / (hiq - loq + 1e-06), 0, 1)\n            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)\n            vol[idx] = (aw * 255).astype(np.uint8)\n            idx += 1\n        if idx >= MAXS:\n            break\n    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)\n    return (vol, mask)\n\ndef find_test_root():\n    cands = [\'/kaggle/input/competitions/rsna-knee-abnormality-detection\', \'/kaggle/input/rsna-knee-abnormality-detection\']\n    for b in cands:\n        if os.path.exists(b + \'/test.csv\'):\n            return b\n    for d, _, f in os.walk(\'/kaggle/input\'):\n        if \'test.csv\' in f and (os.path.isdir(d + \'/test_series\') or os.path.isdir(d + \'/test_images\')):\n            return d\n    for d, _, f in os.walk(\'/kaggle/input\'):\n        if \'test.csv\' in f:\n            return d\n    raise RuntimeError(\'no test root under /kaggle/input\')\n\ndef find_weight_file(fname):\n    direct = [f\'/kaggle/input/raptor-knee-maxspan/{fname}\', f\'/kaggle/input/raptor-knee-native384dense/{fname}\', f\'/kaggle/input/raptor-knee-native384/{fname}\', f\'/kaggle/input/raptor-knee-arms/{fname}\', f\'/kaggle/input/raptor-knee-arms/1/{fname}\', f\'/kaggle/input/raptor-cnn336/{fname}\']\n    for p in direct:\n        if os.path.exists(p):\n            return p\n    for d in sorted(glob.glob(\'/kaggle/input/*/\')):\n        if \'competition\' in d.lower():\n            continue\n        hits = glob.glob(os.path.join(d, \'**\', fname), recursive=True)\n        if hits:\n            return hits[0]\n    raise RuntimeError(f\'{fname} not found under /kaggle/input\')\n\ndef find_optional_verified_weight(fname, expected_sha256, root=\'/kaggle/input\'):\n    hits = []\n    for directory in sorted(glob.glob(os.path.join(root, \'*/\'))):\n        if \'competition\' in directory.lower():\n            continue\n        hits.extend(glob.glob(os.path.join(directory, \'**\', fname), recursive=True))\n    for path in sorted(set(hits)):\n        digest = hashlib.sha256()\n        with open(path, \'rb\') as stream:\n            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b\'\'):\n                digest.update(chunk)\n        if digest.hexdigest() == expected_sha256:\n            return path\n        print(f\'[head-refit] ignored hash-mismatched optional checkpoint: {path}\', flush=True)\n    return None\n\ndef main():\n    import pandas as pd\n    t0 = time.time()\n    dev = "cuda" if torch.cuda.is_available() else "cpu"\n    print(f"device {dev} | gpus {torch.cuda.device_count()} | torch {torch.__version__}", flush=True)\n    root = find_test_root()\n    tsdir = root + "/test_series"\n    if not os.path.isdir(tsdir):\n        tsdir = root + "/test_images"\n    print("test root:", root, "| series dir:", tsdir, flush=True)\n    test = pd.read_csv(root + "/test.csv")\n    test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)\n    test_ids = test["StudyInstanceUID"].tolist()\n    tser = pd.read_csv(root + "/test_series.csv")\n    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)\n    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)\n    series = {key: frame.to_dict("records") for key, frame in tser.groupby("StudyInstanceUID")}\n    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)\n    sub_cols = ["StudyInstanceUID"] + LAB\n    sample = os.path.join(root, "sample_submission.csv")\n    if os.path.exists(sample):\n        sub_cols = list(pd.read_csv(sample, nrows=1).columns)\n    reader = _make_reader()\n    n_study, n_arm = len(test_ids), len(ARMS)\n    arm_probs = [np.full((n_study, len(LAB)), 0.5, np.float32) for _ in range(n_arm)]\n    for arm_index, arm in enumerate(ARMS):\n        globals()["IMG"] = int(arm["img"])\n        globals()["SLOTS"] = list(arm["slots"])\n        globals()["MAXS"] = sum(slot[2] for slot in SLOTS)\n        globals()["SPAN_LO"], globals()["SPAN_HI"] = map(float, arm["span"])\n        globals()["K_EVAL"] = int(arm["k_eval"])\n        weight_path = find_weight_file(arm["file"])\n        model, resolution = load_model(weight_path, arm["arch"], arm["res"], dev)\n        print(f"[arm {arm_index}] {arm[\'name\']} | img {IMG} | slices {MAXS} | "\n              f"span {SPAN_LO:.2f}-{SPAN_HI:.2f} | windows {K_EVAL} | "\n              f"res {resolution} | {time.time() - t0:.0f}s", flush=True)\n        for study_index, study_uid in enumerate(test_ids):\n            try:\n                volume, mask = build_study(study_uid, series, tsdir, reader)\n                windows = eval_windows(volume, mask, k=K_EVAL, res=resolution, norm=NORM)\n                if bool(arm.get("reverse", False)):\n                    windows = windows.flip(1).contiguous()\n                arm_probs[arm_index][study_index] = infer_probs(model, windows, dev)\n                del volume, mask, windows\n            except Exception as error:\n                print(f"  [arm {arm_index}] study {study_index} {study_uid[:16]} FALLBACK "\n                      f"({type(error).__name__}: {error})", flush=True)\n            if (study_index + 1) % 100 == 0 or study_index + 1 == n_study:\n                print(f"  [arm {arm_index}] {study_index + 1}/{n_study} | "\n                      f"{time.time() - t0:.0f}s", flush=True)\n        del model\n        gc.collect()\n        if str(dev).startswith("cuda"):\n            torch.cuda.empty_cache()\n        print(f"[arm {arm_index}] done + freed | {time.time() - t0:.0f}s", flush=True)\n    weights = np.array([float(arm.get("w", 1.0)) for arm in ARMS], dtype=np.float64)\n    weights /= weights.sum()\n    print(f"[blend] global probability mean w="\n          f"{dict(zip([arm[\'name\'] for arm in ARMS], weights.round(4)))}", flush=True)\n    probability_blend = np.tensordot(\n        weights, np.stack([np.clip(values, 0, 1) for values in arm_probs]), axes=(0, 0))\n    ranks = rankpct(probability_blend)\n    if not np.isfinite(ranks).all():\n        ranks[~np.isfinite(ranks)] = 0.5\n    submission = pd.DataFrame(ranks.astype(np.float32), columns=LAB)\n    submission.insert(0, "StudyInstanceUID", test_ids)\n    submission = submission[sub_cols]\n    assert submission["StudyInstanceUID"].tolist() == test_ids\n    assert np.isfinite(submission[LAB].values).all()\n    out = "/kaggle/working/_raptor.csv"\n    submission.to_csv(out, index=False)\n    print("wrote", out, "|", len(submission), "rows x", len(submission.columns), "cols", flush=True)\n    print(submission.head().to_string(index=False), flush=True)\n    print(f"DONE {time.time() - t0:.0f}s", flush=True)\n# ==== END VERBATIM _KE_SRC ====\n\n\ndef _apply_raptor_env_overrides():\n    """Apply the documented RAPTOR_* env-var overrides in place, without\n    editing the verbatim body above."""\n    arm_names = os.environ.get("RAPTOR_ARM_NAMES")\n    if arm_names:\n        wanted = {n.strip() for n in arm_names.split(",") if n.strip()}\n        filtered = [a for a in ARMS if a["name"] in wanted]\n        if not filtered:\n            raise ValueError(f"RAPTOR_ARM_NAMES matched no arm in {wanted}")\n        globals()["ARMS"] = filtered\n\n    extra_dirs = os.environ.get("RAPTOR_CKPT_DIRS")\n    if extra_dirs:\n        extra = [d for d in extra_dirs.split(os.pathsep) if d]\n        _orig_find_weight_file = find_weight_file\n\n        def _find_weight_file_with_extra_dirs(fname, _extra=extra,\n                                              _orig=_orig_find_weight_file):\n            for d in _extra:\n                p = os.path.join(d, fname)\n                if os.path.exists(p):\n                    return p\n            return _orig(fname)\n\n        globals()["find_weight_file"] = _find_weight_file_with_extra_dirs\n\n\ndef run():\n    """Entry point: apply env overrides, run main(), relocate output if asked."""\n    _apply_raptor_env_overrides()\n    main()\n    out_override = os.environ.get("RAPTOR_OUTPUT_CSV")\n    if out_override and out_override != "/kaggle/working/_raptor.csv":\n        os.replace("/kaggle/working/_raptor.csv", out_override)\n\n\nif __name__ == "__main__":\n    run()\n'
# END BUNDLED RAPTOR

# BEGIN BUNDLED RESGATED
_RESGATED_SRC = '# notebooks/fusion/resgated_runner.py\n#\n# Standalone extraction of the subprocess launch that `_coat_substitute()`\n# makes in the public kernel "the-bee-s-knees-final-rsna-push" (Kaggle\n# kernel prvsiyan/the-bee-s-knees-final-rsna-push, pulled to\n# C:/kaggle/staging/competitors/the-bee-s-knees-final-rsna-push/ on\n# 2026-09-09; source authenticated by the pulled notebook\'s own SHA-256\n# 6b03919bbca5fe61e37c5ea6546609a771cd7db6503121f3e2dacdd8c9b6b052).\n#\n# Original residual-gated CoAtNet checkpoints and runtime module:\n# Mattia Angeli, dataset `mattiaangeli/rsna-knee-coat-resgated-ep10-top3`\n# (`coatnet_resgated_ep10_top3_inference.py` + manifest +\n# `coat_resgated_gold_e0{4,6,8}.pt`), CC0-1.0. Offline opencv wheel:\n# `mattiaangeli/opencv-python-headless-4120088-x86`\n# (opencv-python-headless 4.12.0.88, Apache-2.0). Assembled/reproduced (per\n# the source kernel\'s licence table) from Mattia Angeli\'s V27 notebook,\n# Apache-2.0.\n#\n# `_coat_substitute()` in the source cell runs the released module as a\n# subprocess and then rank-blends its output into `/kaggle/working/_raptor.csv`\n# at private_alpha=0.40. This file reproduces only the subprocess launch\n# (manifest/wheel SHA-256 verification kept, exactly as the source does it)\n# and writes the resgated arm\'s own per-study probabilities straight to\n# `_resgated.csv`, in whatever column order the released module\'s own\n# `run_submission()` emits (that module is the one that owns\n# `sample_submission` column ordering; this runner does not reindex or\n# validate it against `_raptor.csv` -- that identity check, and all\n# cross-arm fusion, happens in `fusion_submit.py`, a later task).\n#\n# Inputs are located by marker file (manifest / wheel filename) walked\n# recursively under /kaggle/input, which resolves both the flat\n# /kaggle/input/<name>/... mount layout and the nested\n# /kaggle/input/datasets/<owner>/<name>/... layout, matching the pattern\n# `src/preprocess.py:find_by_marker` uses elsewhere in this repo.\n#\n# Argument/env contract (all optional; defaults reproduce the source\n# kernel\'s hardcoded behaviour):\n#   RESGATED_ARTIFACT_ROOT       directory holding\n#                                 coat_resgated_ep10_top3_manifest.json and\n#                                 coatnet_resgated_ep10_top3_inference.py\n#                                 (default: discovered under /kaggle/input by\n#                                 manifest SHA-256)\n#   RESGATED_OUTPUT_CSV          output path (default:\n#                                 /kaggle/working/_resgated.csv)\n#   RESGATED_GPU_BATCH_STUDIES   passed through to run_submission()\n#                                 (default: 2, the source kernel\'s value)\n#   RESGATED_BACKBONE_MICRO_IMAGES  passed through to run_submission()\n#                                 (default: 8, the source kernel\'s value)\n# No other parameterisation is added: the manifest/wheel SHA-256 checks and\n# the epoch 4/6/8 checkpoint contract are exactly as in the source kernel\n# and are not overridable.\n#\n# Run standalone: `python resgated_runner.py` (writes _resgated.csv and\n# exits; manifest/wheel absent locally -> RuntimeError naming the missing\n# dataset).\n# Import-safe: `import resgated_runner` defines everything below without\n# running anything.\n\nimport hashlib\nimport json\nimport os\nimport subprocess\nimport sys\nfrom pathlib import Path\n\nMANIFEST_NAME = "coat_resgated_ep10_top3_manifest.json"\nMANIFEST_SHA256 = "98511a8fdeb9da0e6e70c78d013ff636e1476f31c80b5dc134d294b18c3f284e"\nWHEEL_GLOB = "opencv_python_headless-4.12.0.88-*.whl"\nWHEEL_SHA256 = "236c8df54a90f4d02076e6f9c1cc763d794542e886c576a6fee46ec8ff75a7a9"\n# the two competition mount layouts, exactly as src/preprocess.py:find_by_marker\n# prunes them. "train_series"/"test_series" are NOT top-level under\n# /kaggle/input - they live inside the competition dir - so listing them\n# here pruned nothing and left the flat layout\'s ~570 GB DICOM tree in\n# the rglob below.\nSKIP_TOP_DIRS = ("competitions", "rsna-knee-abnormality-detection")\nDEFAULT_OUTPUT_CSV = "/kaggle/working/_resgated.csv"\nDEFAULT_GPU_BATCH_STUDIES = 2\nDEFAULT_BACKBONE_MICRO_IMAGES = 8\n\n\ndef _sha256(path):\n    digest = hashlib.sha256()\n    with open(path, "rb") as fh:\n        for chunk in iter(lambda: fh.read(8 << 20), b""):\n            digest.update(chunk)\n    return digest.hexdigest()\n\n\ndef _find_by_marker_and_hash(pattern, expected_sha256, root="/kaggle/input"):\n    """Locate a file matching `pattern` under `root`, whatever the mount\n    layout, and verify its SHA-256. Returns the first verified match found\n    in a deterministic (sorted) walk, or None."""\n    root_path = Path(root)\n    if not root_path.is_dir():\n        return None\n    for base in sorted(root_path.iterdir()):\n        if base.name in SKIP_TOP_DIRS:\n            continue\n        for candidate in sorted(base.rglob(pattern)):\n            if candidate.is_file() and _sha256(candidate) == expected_sha256:\n                return candidate\n    return None\n\n\ndef find_artifact_root():\n    """Directory holding the manifest (and the released inference module\n    next to it), verified by SHA-256."""\n    override = os.environ.get("RESGATED_ARTIFACT_ROOT")\n    if override:\n        manifest = Path(override) / MANIFEST_NAME\n        if not manifest.is_file() or _sha256(manifest) != MANIFEST_SHA256:\n            raise FileNotFoundError(\n                f"RESGATED_ARTIFACT_ROOT={override!r} does not contain a "\n                f"hash-verified {MANIFEST_NAME}")\n        return Path(override)\n    manifest = _find_by_marker_and_hash(MANIFEST_NAME, MANIFEST_SHA256)\n    if manifest is None:\n        raise FileNotFoundError(\n            f"{MANIFEST_NAME} absent or hash mismatch under /kaggle/input "\n            "(expected dataset mattiaangeli/rsna-knee-coat-resgated-ep10-top3)")\n    return manifest.parent\n\n\ndef find_opencv_wheel():\n    wheel = _find_by_marker_and_hash(WHEEL_GLOB, WHEEL_SHA256)\n    if wheel is None:\n        raise FileNotFoundError(\n            f"{WHEEL_GLOB} absent or hash mismatch under /kaggle/input "\n            "(expected dataset mattiaangeli/opencv-python-headless-4120088-x86)")\n    return wheel\n\n\ndef install_opencv_wheel(wheel_path, target_dir):\n    subprocess.run(\n        [sys.executable, "-m", "pip", "install", "--no-deps", "--quiet",\n         "--target", str(target_dir), str(wheel_path)],\n        check=True)\n\n\ndef run(output_csv=None, artifact_root=None):\n    """Locate the released resgated module, install its pinned opencv wheel\n    offline, run it in a subprocess, and write its per-study probabilities\n    to `output_csv` (sample_submission column order). No blending."""\n    out = Path(output_csv or os.environ.get("RESGATED_OUTPUT_CSV", DEFAULT_OUTPUT_CSV))\n    art = Path(artifact_root) if artifact_root else find_artifact_root()\n    wheel = find_opencv_wheel()\n\n    env_dir = Path("/kaggle/working/_resgated_env")\n    install_opencv_wheel(wheel, env_dir)\n\n    gpu_batch_studies = int(os.environ.get(\n        "RESGATED_GPU_BATCH_STUDIES", DEFAULT_GPU_BATCH_STUDIES))\n    backbone_micro_images = int(os.environ.get(\n        "RESGATED_BACKBONE_MICRO_IMAGES", DEFAULT_BACKBONE_MICRO_IMAGES))\n    receipt_path = out.with_name(out.stem + "_receipt.json")\n\n    child = (\n        "import sys, json\\n"\n        f"sys.path.insert(0, {str(env_dir)!r})\\n"\n        f"sys.path.insert(0, {str(art)!r})\\n"\n        "import cv2; assert cv2.__version__ == \'4.12.0\', cv2.__version__\\n"\n        "import torch; assert torch.cuda.device_count() == 2\\n"\n        "import coatnet_resgated_ep10_top3_inference as rt\\n"\n        "assert rt.base.cv2.__version__ == \'4.12.0\'\\n"\n        "from pathlib import Path\\n"\n        "r = rt.run_submission(competition_root=rt.base.find_competition_root(),\\n"\n        f"    artifact_root=Path({str(art)!r}), output_path=Path({str(out)!r}),\\n"\n        f"    gpu_batch_studies={gpu_batch_studies}, "\n        f"backbone_micro_images={backbone_micro_images})\\n"\n        "assert r[\'status\'] == rt.SUBMISSION_STATUS\\n"\n        "assert r[\'models\'] == 3\\n"\n        "assert [i[\'epoch\'] for i in r[\'checkpoints\']] == [4, 6, 8]\\n"\n        "assert r[\'fallback_studies\'] == 0, r[\'failures\']\\n"\n        f"Path({str(receipt_path)!r}).write_text(json.dumps(r, indent=2))\\n")\n    env = dict(os.environ)\n    env["PYTHONPATH"] = f"{env_dir}{os.pathsep}{art}{os.pathsep}" + env.get("PYTHONPATH", "")\n    proc = subprocess.run([sys.executable, "-c", child], env=env,\n                          capture_output=True, text=True)\n    if proc.returncode != 0:\n        raise RuntimeError(f"resgated child failed: {proc.stderr[-700:]}")\n    return out\n\n\nif __name__ == "__main__":\n    written = run()\n    print(f"wrote {written}")\n'
# END BUNDLED RESGATED

# BEGIN BUNDLED INFER_SUBMIT
_INFER_SRC = '# RSNA Knee: offline inference -> submission.csv. THE submission kernel: internet OFF, GPU on, <= 9 h.\n# Attach: competition data, rsna-67knee-src, training fold-weights -- see docs/code-notes.md "inference pipeline overview".\n\nimport csv\nimport json\nimport os\nimport sys\nimport time\nfrom collections import deque\nfrom concurrent.futures import ProcessPoolExecutor\nfrom concurrent.futures.process import BrokenProcessPool\nimport numpy as np\nimport torch\n\n_SRC_CANDIDATES = ("/kaggle/input/rsna-67knee-src",\n                   "/kaggle/input/datasets/patimakornsrinamen/rsna-67knee-src",\n                   # pre-rename slug, kept so the kernel runs against either name\n                   "/kaggle/input/rsna-knee-src",\n                   "/kaggle/input/datasets/patimakornsrinamen/rsna-knee-src")\nfor _src in _SRC_CANDIDATES:\n    if os.path.exists(_src):\n        sys.path.insert(0, _src)\n        break\nelse:\n    raise FileNotFoundError(f"rsna-67knee-src not mounted; tried {_SRC_CANDIDATES}")\nfrom preprocess import (LABELS, build_study_tensor, find_by_marker, find_input,\n                        read_series_table, recipe_kwargs)\nfrom model import Model, POOLERS, predict_study\nfrom blend_eval import rank_blend\n\nCOMP = find_input(\n    "/kaggle/input/rsna-knee-abnormality-detection",\n    "/kaggle/input/competitions/rsna-knee-abnormality-detection",\n)\nOUT = "/kaggle/working/submission.csv"\nDECODE_WORKERS = 4\n# weights come from a pinned dataset slug, never a kernel\'s latest output -- see docs/code-notes.md "pinned weights and version check"\nEXPECTED_LABEL_FILE = "consensus_v1.csv"\nEXPECTED_RECIPE_VERSION = {"dinov2s": "v4", "coatnet2": "v4", "convnexts": "v4"}\nWEIGHTS_CANDIDATES = (\n    "/kaggle/input/rsna-67knee-weights-v4b",\n    "/kaggle/input/datasets/patimakornsrinamen/rsna-67knee-weights-v4b",\n)\n# dinov2s reads the WEIGHTS_CANDIDATES global so a monkeypatch still works -- see docs/code-notes.md "per-family weights candidates"\nFAMILY_WEIGHTS = {\n    "dinov2s": WEIGHTS_CANDIDATES,\n    "coatnet2": (\n        "/kaggle/input/rsna-67knee-weights-v6c",\n        "/kaggle/input/datasets/patimakornsrinamen/rsna-67knee-weights-v6c",\n    ),\n    # round 10 pure-CNN arm (ConvNeXt-Small), trained locally; dataset pinned when the run lands\n    "convnexts": (\n        "/kaggle/input/rsna-67knee-weights-convnexts",\n        "/kaggle/input/datasets/patimakornsrinamen/rsna-67knee-weights-convnexts",\n    ),\n}\n\n# MAIN/EFFICIENCY presets trade fold+slice coverage for speed; see PRESETS below -- see docs/code-notes.md "MAIN and EFFICIENCY presets"\nPRESET = "coatnet"  # "main" | "main_attn" | "efficiency" | "coatnet" | "fast" | "blend" - coatnet (CoAtNet-2 v6c alone, 0.912 LB) is the shipping default since round 6; main_attn (DINOv2-S v4b) scored 0.892 and the weighted blend 0.909\n\n# TIMING_STUDIES>0 times training studies instead of scoring the test set -- see docs/code-notes.md "TIMING_STUDIES timing mode"\nTIMING_STUDIES = 0\n\n# TIMING_GOLD_SLICES sweeps slice budgets against gold58, ignoring TIMING_STUDIES -- see docs/code-notes.md "TIMING_GOLD_SLICES sweep mode"\nTIMING_GOLD_SLICES = ""\n\n# FAST_FOLDS/FAST_SLICES let this round\'s operating point move in one line -- see docs/code-notes.md "FAST preset constants"\nFAST_FOLDS = 1\nFAST_SLICES = None\n\n# Optional per-preset key "window_stride" (absent/None = the training-time\n# non-overlapping window grouping): an int makes model.study_windows score\n# dense overlapping windows every `window_stride` slices at inference\n# (stride 1 = +0.0071 paired OOF on v6c, ~3x the backbone work per study;\n# see docs/methodology.md). Only meaningful for window-trained families.\nPRESETS = {\n    "main": {"families": ("dinov2s",), "use_tta": None, "max_folds": None,\n             "max_slices": None, "pool": None},\n    # MAIN_ATTN: the main preset with the tuned window_pool ignored - the\n    # diagnostic that separates the pooling table from the pixels on the LB.\n    "main_attn": {"families": ("dinov2s",), "use_tta": None,\n                  "max_folds": None, "max_slices": None, "pool": "attn"},\n    "efficiency": {"families": ("dinov2s",), "use_tta": False,\n                   "max_folds": 1, "max_slices": 6, "pool": "attn"},\n    # COATNET: arm 2 alone, same recipe and pooling stance as main_attn.\n    "coatnet": {"families": ("coatnet2",), "use_tta": None,\n                "max_folds": None, "max_slices": None, "pool": "attn"},\n    # CONVNEXT: the round-10 ConvNeXt-Small arm alone, same stance as coatnet.\n    "convnext": {"families": ("convnexts",), "use_tta": None,\n                 "max_folds": None, "max_slices": None, "pool": "attn"},\n    # FAST: CoAtNet-2 alone, no TTA, forced all-attn pooling -- see docs/code-notes.md "FAST preset definition"\n    "fast": {"families": ("coatnet2",), "use_tta": False,\n             "max_folds": FAST_FOLDS, "max_slices": FAST_SLICES,\n             "pool": "attn"},\n    # BLEND: rank-blend of both arms; allow_partial/allow_equal_weights stay False -- see docs/code-notes.md "BLEND preset definition"\n    "blend": {"families": ("dinov2s", "coatnet2"), "use_tta": None,\n              "max_folds": None, "max_slices": None, "pool": "attn",\n              "weights_file": "blend_weights.json", "allow_partial": False,\n              "allow_equal_weights": False},\n}\n\n# MAX_INFER_HOURS stops a multi-family run before it burns the whole session -- see docs/code-notes.md "MAX_INFER_HOURS budget"\nMAX_INFER_HOURS = 8.5\n\n# PRIOR_FALLBACK backs run_configs (round-2/3) that carry no prior of their own -- see docs/code-notes.md "PRIOR_FALLBACK prevalence"\nPRIOR_FALLBACK = [0.2412, 0.1784, 0.4441, 0.2153, 0.3875, 0.3004,\n                  0.462, 0.5571, 0.2665, 0.3159, 0.2467, 0.1619]\n\n# decode-worker state, set by _init_worker in every process (pool or inline)\n_WORKER_RECIPE = {}\n_WORKER_MAX_SLICES = None\n_WORKER_SLICE_ALIGN = 1\n_WORKER_SERIES_DIR = "test_series"\n\n\ndef decode_study(args):\n    """Worker: (uid, series_rows) -> (uid, volume, filled_slots); volume is\n    None when decoding raised. filled_slots == 0 means nothing decoded."""\n    uid, series_rows = args\n    assert _WORKER_RECIPE, "decode worker used before _init_worker"\n    try:\n        vol, filled = build_study_tensor(\n            os.path.join(COMP, _WORKER_SERIES_DIR, uid), series_rows,\n            max_slices=_WORKER_MAX_SLICES, slice_align=_WORKER_SLICE_ALIGN,\n            **_WORKER_RECIPE)\n        return uid, vol, filled\n    except Exception as e:\n        print(f"WARN study {uid} decode failed ({type(e).__name__})",\n              flush=True)\n        return uid, None, 0\n\n\ndef _init_worker(recipe, max_slices, slice_align=1, series_dir="test_series"):\n    global _WORKER_MAX_SLICES, _WORKER_SLICE_ALIGN, _WORKER_SERIES_DIR\n    _WORKER_RECIPE.update(recipe)\n    _WORKER_MAX_SLICES = max_slices\n    _WORKER_SLICE_ALIGN = slice_align\n    _WORKER_SERIES_DIR = series_dir\n\n\ndef pool_windows(probs, window_pool, p_attn):\n    """probs: (n_models, n_windows, 12) NaN-padded per-window sigmoid probs\n    (model.predict_study\'s win_probs). window_pool: {label: mode}. p_attn:\n    (12,) from predict_study - the ensemble\'s own attention pooling. Returns\n    (12,): per label, the pooled value for window_pool[label] (model.POOLERS:\n    mean/max/top2, per model across windows then averaged across models -\n    the same order tune_pooling.py uses per fold); an "attn" mode, or a label\n    missing from window_pool, keeps p_attn unchanged. Pure function so it can\n    be unit-tested against tune_pooling.py\'s pooling."""\n    out = np.array(p_attn, dtype=np.float32).copy()\n    if not window_pool:\n        return out\n    for c, name in enumerate(LABELS):\n        mode = window_pool.get(name, "attn")\n        if mode not in POOLERS:  # "attn", or an unknown mode: leave p_attn\n            continue\n        col = probs[:, :, c]  # (n_models, n_windows)\n        per_model = POOLERS[mode](col, axis=1)\n        out[c] = float(np.mean(per_model))\n    return out\n\n\ndef decode_all(work_items, recipe, max_slices, workers=None, slice_align=1,\n              series_dir="test_series"):\n    """Yield (uid, vol, filled) for every work item, in input order.\n\n    Runs a process pool with a bounded number of studies in flight (so a\n    fast decoder cannot pile gigabytes of volumes ahead of the GPU) and, if a\n    worker dies hard (segfault in a C decoder, OOM kill), finishes the rest\n    inline instead of losing every prediction made so far. workers <= 1\n    decodes inline from the start (local dry-runs). series_dir is the\n    subdirectory of the competition mount holding each uid\'s DICOMs -\n    "test_series" for a normal run, "train_series" for TIMING_STUDIES."""\n    workers = DECODE_WORKERS if workers is None else workers\n    if workers <= 1:\n        _init_worker(recipe, max_slices, slice_align, series_dir)\n        for w in work_items:\n            yield decode_study(w)\n        return\n    ex = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,\n                             initargs=(recipe, max_slices, slice_align,\n                                      series_dir))\n    pending = deque()\n    submitted = done = 0\n    try:\n        while done < len(work_items):\n            while submitted < len(work_items) and len(pending) < 4 * workers:\n                pending.append(ex.submit(decode_study, work_items[submitted]))\n                submitted += 1\n            yield pending.popleft().result()\n            done += 1\n    except BrokenProcessPool:\n        print(f"WARN decode pool died after {done} studies; finishing the "\n              f"remaining {len(work_items) - done} inline", flush=True)\n        _init_worker(recipe, max_slices, slice_align, series_dir)\n        for w in work_items[done:]:\n            yield decode_study(w)\n    finally:\n        ex.shutdown(wait=False, cancel_futures=True)\n\n\ndef _expected_recipe_version(family):\n    """Resolve EXPECTED_RECIPE_VERSION for one family. A bare string (or\n    None) applies to every family - back-compat with a global override, e.g.\n    a dry-run driver\'s `I.EXPECTED_RECIPE_VERSION = None`. A dict (the\n    shipping default) is a per-family table keyed by run_config["family"];\n    a family absent from it gets no version check."""\n    if EXPECTED_RECIPE_VERSION is None or isinstance(EXPECTED_RECIPE_VERSION, str):\n        return EXPECTED_RECIPE_VERSION\n    return EXPECTED_RECIPE_VERSION.get(family)\n\n\ndef load_family(family, dev):\n    """Resolve a family\'s weights dir, validate its run_config against this\n    family, and load its fold models. Returns (models, cfg, weights_dir).\n\n    dinov2s\'s candidates are read from the current WEIGHTS_CANDIDATES global\n    (not FAMILY_WEIGHTS["dinov2s"]) so a dry-run driver that monkeypatches\n    WEIGHTS_CANDIDATES after import still takes effect."""\n    candidates = (WEIGHTS_CANDIDATES if family == "dinov2s"\n                 else FAMILY_WEIGHTS[family])\n    try:\n        weights_dir = find_input(*candidates)\n    except FileNotFoundError:\n        # fallback for local dry-runs; on Kaggle the dataset must be attached\n        weights_dir = find_by_marker(f"{family}_fold0.pt")\n    cfg_path = os.path.join(weights_dir, "run_config.json")\n    with open(cfg_path) as fh:\n        cfg = json.load(fh)\n    print(f"WEIGHTS_DIR[{family}]: {weights_dir} | device: {dev} | cfg: {cfg}",\n          flush=True)\n    assert cfg["label_file"] == EXPECTED_LABEL_FILE, (\n        f"{family} weights were trained on {cfg[\'label_file\']}, expected "\n        f"{EXPECTED_LABEL_FILE} - wrong weights attached")\n    expected_version = _expected_recipe_version(family)\n    if expected_version is not None:\n        assert cfg.get("version", "v2") == expected_version, (\n            f"{family} weights carry recipe {cfg.get(\'version\', \'v2\')}, "\n            f"expected {expected_version} - wrong weights dataset attached")\n    assert cfg.get("family", "dinov2s") == family, (\n        f"weights at {weights_dir} carry family "\n        f"{cfg.get(\'family\', \'dinov2s\')!r}, expected {family!r}")\n    assert not cfg.get("smoke"), (\n        f"weights at {weights_dir} are a smoke run (smoke={cfg.get(\'smoke\')})")\n    n_folds_expected = int(cfg.get("n_folds", 3))\n\n    preset = PRESETS[PRESET]\n    max_folds = preset["max_folds"] or n_folds_expected\n    # "_last.pt" is the SWA companion file and must not count as a fold weight -- see docs/code-notes.md "fold weight file filter"\n    weight_files = sorted(f for f in os.listdir(weights_dir)\n                          if f.startswith(f"{family}_fold") and f.endswith(".pt")\n                          and not f.endswith("_last.pt"))\n    assert len(weight_files) == n_folds_expected, (\n        f"expected {n_folds_expected} {family} fold weights in {weights_dir}, "\n        f"found {len(weight_files)}")\n    models = []\n    for f in weight_files[:max_folds]:\n        m = Model(pretrained=False, img_size=cfg["size"], arch=cfg.get("arch"),\n                  n_slots=cfg.get("n_slots")).to(dev).eval()\n        m.load_state_dict(torch.load(os.path.join(weights_dir, f),\n                                     map_location=dev))\n        models.append(m)\n    print(f"{len(models)} {family} fold models loaded", flush=True)\n    return models, cfg, weights_dir\n\n\ndef _resolve_use_tta(preset, cfg):\n    """preset["use_tta"] overrides; None trusts the training run\'s own\n    measurement (same rule for every family - see the PRESETS comment)."""\n    use_tta = preset["use_tta"]\n    if use_tta is None:\n        if "use_tta" in cfg:  # decided in train on the unrounded AUCs\n            use_tta = bool(cfg["use_tta"])\n        else:  # round-2/3 run_config: only the rounded AUCs are recorded\n            use_tta = (cfg.get("tta_gold58_auc") is not None\n                       and cfg.get("gold58_auc") is not None\n                       and cfg["tta_gold58_auc"] >= cfg["gold58_auc"])\n    return use_tta\n\n\ndef score_study(models, dev, use_amp, use_tta, window, window_pool, prior,\n                uid, vol, filled, stride=None):\n    """One family\'s score for one study. Returns (probs (12,), failed: bool);\n    failed marks a prior fallback (decode failure, non-finite prediction with\n    no usable attention score, or an inference exception)."""\n    if vol is None or filled == 0:\n        # decode raised, or no slot decoded: a black image would get\n        # an arbitrary constant score - use the prior instead\n        return prior, True\n    try:\n        # one backbone pass yields both the attention score and per-window probs -- see docs/code-notes.md "one backbone pass per study"\n        p_attn, wprobs, _ = predict_study(models, vol, dev, use_amp,\n                                          tta=use_tta, window=window,\n                                          stride=stride)\n        non_finite = (not np.all(np.isfinite(p_attn))\n                     or (wprobs is not None\n                         and not np.all(np.isfinite(wprobs))))\n        if non_finite:\n            # never write nan into the CSV: fall back to the\n            # attention-only score if it\'s usable, else the prior\n            if np.all(np.isfinite(p_attn)):\n                print(f"WARN study {uid} non-finite prediction; "\n                      "using attention-only score", flush=True)\n                return p_attn, False\n            else:\n                print(f"WARN study {uid} non-finite prediction; "\n                      "using prior", flush=True)\n                return prior, True\n        elif window is None:\n            return p_attn, False\n        elif window_pool and any(m != "attn" for m in window_pool.values()):\n            return pool_windows(wprobs, window_pool, p_attn), False\n        else:\n            return p_attn, False\n    except Exception as e:\n        print(f"WARN study {uid} inference failed "\n              f"({type(e).__name__}); using prior", flush=True)\n        return prior, True\n\n\ndef projected_hours(elapsed_s, done, total):\n    """Pure: elapsed_s spent on `done` (of `total`) studies -> the projected\n    total wall-clock hours at that same per-study rate. done must be > 0."""\n    return elapsed_s / done * total / 3600.0\n\n\ndef check_projection(elapsed_s, done, total, n_families):\n    """Print the projected total runtime and, for a multi-family preset\n    (a blend, which pays for every family\'s forward pass), hard-fail if it\n    would exceed MAX_INFER_HOURS - a single-family preset never raises here,\n    same as before this check existed. MAX_INFER_HOURS is read as a module\n    global so a driver that patches it after import still takes effect."""\n    proj = projected_hours(elapsed_s, done, total)\n    print(f"projected total: {proj:.2f} h", flush=True)\n    if n_families > 1 and proj > MAX_INFER_HOURS:\n        raise RuntimeError(\n            f"projected {proj:.2f} h exceeds MAX_INFER_HOURS={MAX_INFER_HOURS} h")\n    return proj\n\n\ndef _write_csv(path, uids, prob_rows):\n    with open(path, "w", newline="") as fh:\n        w = csv.writer(fh)\n        w.writerow(["StudyInstanceUID"] + LABELS)\n        w.writerows([[u] + [f"{p:.6f}" for p in row]\n                    for u, row in zip(uids, prob_rows)])\n\n\ndef read_gold58(path):\n    """gold58.csv -> (uids, y) with y (len(uids), 12) in LABELS order. Fails\n    loudly, naming the missing column, if the uid column or any of the 12\n    LABELS columns is absent."""\n    with open(path, newline="", encoding="utf-8") as fh:\n        reader = csv.DictReader(fh)\n        fields = reader.fieldnames or []\n        assert "StudyInstanceUID" in fields, (\n            f"gold58.csv missing column \'StudyInstanceUID\' (found {fields})")\n        for name in LABELS:\n            assert name in fields, (\n                f"gold58.csv missing column {name!r} (found {fields})")\n        uids, rows = [], []\n        for row in reader:\n            uids.append(row["StudyInstanceUID"])\n            rows.append([float(row[name]) for name in LABELS])\n    return uids, np.asarray(rows, dtype=np.float64)\n\n\ndef main():\n    dev = "cuda" if torch.cuda.is_available() else "cpu"\n    use_amp = dev == "cuda"\n    preset = PRESETS[PRESET]\n    families = preset["families"]\n\n    # a family load failure is a hard stop unless the preset sets allow_partial -- see docs/code-notes.md "family load failure handling"\n    loaded = {}  # family -> (models, cfg, weights_dir, use_tta)\n    for family in families:\n        try:\n            models, cfg, weights_dir = load_family(family, dev)\n        except Exception as exc:\n            if not preset.get("allow_partial"):\n                raise\n            print(f"WARN family {family} failed: {exc!r}", flush=True)\n            continue\n        loaded[family] = (models, cfg, weights_dir, _resolve_use_tta(preset, cfg))\n    if not loaded:\n        raise RuntimeError(\n            f"no family loaded for preset {PRESET!r} (tried {families})")\n    fam_names = list(loaded.keys())\n\n    # blend_weights.json failures are hard errors; allow_equal_weights only WARNs -- see docs/code-notes.md "blend_weights.json contract"\n    weights_file = preset.get("weights_file")\n    blend_weights, blend_weights_matrix = None, None\n    if weights_file:\n        hits = [os.path.join(loaded[fam][2], weights_file) for fam in fam_names\n               if os.path.isfile(os.path.join(loaded[fam][2], weights_file))]\n        if not hits:\n            if not preset.get("allow_equal_weights"):\n                raise RuntimeError(\n                    f"{weights_file} not found in any of "\n                    f"{[loaded[fam][2] for fam in fam_names]}")\n            print(f"WARN {weights_file} not found; using equal weights",\n                  flush=True)\n        elif len(hits) > 1:\n            raise RuntimeError(\n                f"{weights_file} found in more than one weights dir: {hits}")\n        else:\n            with open(hits[0]) as fh:\n                bw_json = json.load(fh)\n            json_families, want_families = set(bw_json["families"]), set(fam_names)\n            if json_families != want_families:\n                raise RuntimeError(\n                    f"{weights_file} families {json_families} != loaded "\n                    f"families {want_families}")\n            try:\n                blend_weights_matrix = np.array(\n                    [[bw_json["weights"][label][fam] for label in LABELS]\n                     for fam in fam_names])\n            except KeyError as e:\n                raise RuntimeError(\n                    f"{weights_file} missing weight entry {e} (labels="\n                    f"{LABELS}, families={fam_names})") from e\n            blend_weights = {"path": hits[0], "families": fam_names,\n                             "matrix": blend_weights_matrix.tolist()}\n\n    # families sharing a (recipe, window) pair are grouped so decode_all runs once -- see docs/code-notes.md "recipe/window family grouping"\n    recipe_by_family = {fam: recipe_kwargs(loaded[fam][1]) for fam in fam_names}\n    window_by_family = {fam: loaded[fam][1].get("window") for fam in fam_names}\n    groups = []  # [((recipe_items, window), [fam, ...])]\n    group_index = {}\n    for fam in fam_names:\n        key = (tuple(sorted(recipe_by_family[fam].items())), window_by_family[fam])\n        if key not in group_index:\n            group_index[key] = len(groups)\n            groups.append((key, []))\n        groups[group_index[key]][1].append(fam)\n\n    max_slices = preset["max_slices"]\n    window_stride = preset.get("window_stride")  # None = training-time grouping\n    slice_align_by_group = {}\n    for key, fams in groups:\n        _, group_window = key\n        if max_slices is not None and group_window is not None:\n            assert max_slices % group_window == 0, (\n                f"preset max_slices={max_slices} is not a multiple of "\n                f"{fams}\'s window={group_window} - the decoded subset would "\n                "not land on window-group boundaries")\n        # slice_align keeps a truncated decode on window-group boundaries -- see docs/code-notes.md "slice_align boundary rule"\n        slice_align_by_group[key] = group_window or 1\n\n    # per-family window_pool / prior: each family\'s own tuned table / prior,\n    # not shared, even across families that do share a recipe/window.\n    window_pool_by_family, prior_by_family = {}, {}\n    for fam in fam_names:\n        cfg = loaded[fam][1]\n        if preset["pool"] == "attn":\n            # efficiency/coatnet/blend/main_attn force all-attn pooling -- see docs/code-notes.md "forced all-attn pooling presets"\n            window_pool_by_family[fam] = {name: "attn" for name in LABELS}\n        else:  # preset["pool"] is None (main): trust the run_config\'s table\n            window_pool_by_family[fam] = cfg.get("window_pool")\n        prior_by_family[fam] = cfg.get("prior", PRIOR_FALLBACK)\n\n    infer_cfg = {\n        "preset": PRESET, "families": fam_names, "max_slices": max_slices,\n        "recipe_groups": [\n            {"families": fams, "recipe": dict(key[0]), "window": key[1],\n             "slice_align": slice_align_by_group[key]}\n            for key, fams in groups\n        ],\n        "per_family": {\n            fam: {"n_folds": len(loaded[fam][0]),\n                  "arch": loaded[fam][1].get("arch"),\n                  "use_tta": loaded[fam][3],\n                  "weights_dir": loaded[fam][2],\n                  "window_pool": window_pool_by_family[fam],\n                  "recipe": recipe_by_family[fam],\n                  "window": window_by_family[fam]}\n            for fam in fam_names\n        },\n        "blend_weights": blend_weights,\n        "projected_hours": None,  # filled in after the study loop\n    }\n    print(f"INFER_CFG: {infer_cfg}", flush=True)\n    with open(os.path.join(os.path.dirname(OUT), "infer_config.json"),\n              "w") as fh:\n        json.dump(infer_cfg, fh)  # the effective config, next to the CSV\n\n    # TIMING_GOLD_SLICES is checked before TIMING_STUDIES and ignores it entirely -- see docs/code-notes.md "TIMING_GOLD_SLICES precedence"\n    if TIMING_GOLD_SLICES:\n        # tune_pooling imported here so a src mount without it can\'t break other paths -- see docs/code-notes.md "deferred tune_pooling import"\n        from tune_pooling import auc_per_label\n        gold_slices = [int(s) for s in TIMING_GOLD_SLICES.split(",") if s.strip()]\n        for budget in gold_slices:\n            for key, fams in groups:\n                group_window = key[1]\n                assert group_window is None or budget % group_window == 0, (\n                    f"TIMING_GOLD_SLICES budget={budget} is not a multiple "\n                    f"of {fams}\'s window={group_window} - the decoded "\n                    "subset would not land on window-group boundaries")\n\n        src_dir = find_input(*_SRC_CANDIDATES)\n        gold_uids, gold_y = read_gold58(os.path.join(src_dir, "gold58.csv"))\n        by_study = read_series_table(os.path.join(COMP, "train_series.csv"))\n        work_items = [(u, by_study.get(u, [])) for u in gold_uids]\n        n_studies = len(gold_uids)\n        n_folds = sum(len(loaded[fam][0]) for fam in fam_names)\n        print(f"{n_studies} gold studies (from gold58.csv)", flush=True)\n\n        records = []\n        for budget in gold_slices:\n            decode_s = 0.0\n            gpu_s = 0.0\n            wt0 = time.time()\n            probs_by_family = {fam: [None] * n_studies for fam in fam_names}\n            for key, fams in groups:\n                group_recipe, group_window = key[0], key[1]\n                group_recipe = dict(group_recipe)\n                group_slice_align = slice_align_by_group[key]\n                decode_iter = iter(decode_all(\n                    work_items, group_recipe, budget,\n                    slice_align=group_slice_align, series_dir="train_series"))\n                k = 0\n                while True:\n                    dt0 = time.time()\n                    try:\n                        uid, vol, filled = next(decode_iter)\n                    except StopIteration:\n                        break\n                    decode_s += time.time() - dt0\n                    assert uid == gold_uids[k], (\n                        f"decode order drifted at index {k} ({uid!r} != "\n                        f"{gold_uids[k]!r})")\n                    for fam in fams:\n                        models, _, _, use_tta = loaded[fam]\n                        gt_start = time.time()\n                        probs, _ = score_study(\n                            models, dev, use_amp, use_tta, group_window,\n                            window_pool_by_family[fam], prior_by_family[fam],\n                            uid, vol, filled, stride=window_stride)\n                        gpu_s += time.time() - gt_start\n                        probs_by_family[fam][k] = np.asarray(probs, dtype=np.float64)\n                    k += 1\n            wall_s = time.time() - wt0\n            if len(fam_names) == 1:\n                final_probs = probs_by_family[fam_names[0]]\n            else:\n                mats = [np.stack(probs_by_family[fam]) for fam in fam_names]\n                final_probs = list(rank_blend(mats, weights=blend_weights_matrix))\n            scores = np.stack(final_probs)\n            per_label = auc_per_label(gold_y, scores)\n            # None (not NaN) when every label is single-class in this gold\n            # set: json.dump would otherwise write a bare NaN, invalid JSON.\n            macro_auc = (float(np.mean(list(per_label.values())))\n                        if per_label else None)\n            records.append({\n                "max_slices": budget, "auc": macro_auc,\n                "auc_per_label": per_label, "decode_s": decode_s,\n                "gpu_s": gpu_s, "wall_s": wall_s, "n_studies": n_studies,\n                "n_folds": n_folds,\n            })\n\n        print("GOLD_SWEEP results:", flush=True)\n        print(f"{\'max_slices\':>10} {\'auc\':>8} {\'decode_s\':>9} {\'gpu_s\':>8} "\n              f"{\'wall_s\':>8}", flush=True)\n        for r in records:\n            auc_str = f"{r[\'auc\']:.4f}" if r["auc"] is not None else "n/a"\n            print(f"{r[\'max_slices\']:>10} {auc_str:>8} "\n                  f"{r[\'decode_s\']:>9.2f} {r[\'gpu_s\']:>8.2f} "\n                  f"{r[\'wall_s\']:>8.2f}", flush=True)\n        with open(os.path.join(os.path.dirname(OUT), "timing_gold.json"),\n                  "w") as fh:\n            json.dump(records, fh)\n        return\n\n    # TIMING_STUDIES times training studies since the visible test mount is tiny -- see docs/code-notes.md "TIMING_STUDIES uses training tree"\n    timing = TIMING_STUDIES > 0\n    if timing:\n        by_study = read_series_table(os.path.join(COMP, "train_series.csv"))\n        test_uids = sorted(by_study.keys())[:TIMING_STUDIES]\n        series_dir = "train_series"\n        print(f"{len(test_uids)} timing studies (from train_series.csv)",\n              flush=True)\n    else:\n        by_study = read_series_table(os.path.join(COMP, "test_series.csv"))\n        with open(os.path.join(COMP, "test.csv"), newline="",\n                  encoding="utf-8") as fh:\n            test_uids = [r["StudyInstanceUID"] for r in csv.DictReader(fh)]\n        series_dir = "test_series"\n        print(f"{len(test_uids)} test studies", flush=True)\n\n    # per-study (uid, rows) tuples: the series table is NOT captured in the\n    # callable, so nothing large is re-pickled per map call\n    work_items = [(u, by_study.get(u, [])) for u in test_uids]\n    out_uids = list(test_uids)\n    probs_by_family = {fam: [None] * len(test_uids) for fam in fam_names}\n    n_failed_by_family = {fam: 0 for fam in fam_names}\n    proj_hours_by_group = {}\n    decode_seconds = 0.0\n    gpu_seconds = 0.0\n    t0 = time.time()\n\n    def _timed_decode(gen):\n        # decode_seconds separates decode wait time from GPU time for the projection -- see docs/code-notes.md "timed decode bookkeeping"\n        nonlocal decode_seconds\n        it = iter(gen)\n        while True:\n            dt0 = time.time()\n            try:\n                item = next(it)\n            except StopIteration:\n                return\n            decode_seconds += time.time() - dt0\n            yield item\n\n    for key, fams in groups:\n        group_recipe, group_window = key[0], key[1]\n        group_recipe = dict(group_recipe)\n        group_slice_align = slice_align_by_group[key]\n        gt0 = time.time()\n        decoded = decode_all(work_items, group_recipe, max_slices,\n                             slice_align=group_slice_align,\n                             series_dir=series_dir)\n        if timing:\n            decoded = _timed_decode(decoded)\n        for k, (uid, vol, filled) in enumerate(decoded):\n            assert uid == test_uids[k], (\n                f"decode order drifted at index {k} ({uid!r} != "\n                f"{test_uids[k]!r})")\n            for fam in fams:\n                models, _, _, use_tta = loaded[fam]\n                gt_start = time.time() if timing else None\n                probs, failed = score_study(\n                    models, dev, use_amp, use_tta, group_window,\n                    window_pool_by_family[fam], prior_by_family[fam],\n                    uid, vol, filled, stride=window_stride)\n                if timing:\n                    gpu_seconds += time.time() - gt_start\n                if failed:\n                    n_failed_by_family[fam] += 1\n                probs_by_family[fam][k] = np.asarray(probs, dtype=np.float64)\n            if (k + 1) % 100 == 0:\n                rate = (k + 1) / (time.time() - gt0)\n                print(f"[{\',\'.join(fams)}] {k + 1}/{len(test_uids)} "\n                      f"({rate:.2f} studies/s)", flush=True)\n            if k + 1 == 200:\n                proj_hours_by_group[key] = check_projection(\n                    time.time() - gt0, k + 1, len(test_uids), len(fams))\n\n    if timing:\n        total_wall_s = time.time() - t0\n        n_studies = len(test_uids)\n        studies_per_s = n_studies / total_wall_s if total_wall_s > 0 else 0.0\n        projected_hours_1300 = projected_hours(total_wall_s, n_studies, 1300)\n        timing_block = {\n            "preset": PRESET,\n            "families": fam_names,\n            "n_folds": {fam: len(loaded[fam][0]) for fam in fam_names},\n            "max_slices": max_slices,\n            "n_studies": n_studies,\n            "total_wall_s": total_wall_s,\n            "decode_s": decode_seconds,\n            "gpu_s": gpu_seconds,\n            "studies_per_s": studies_per_s,\n            "projected_hours_1300": projected_hours_1300,\n        }\n        print(f"TIMING: preset={PRESET} families={fam_names} "\n              f"n_folds={timing_block[\'n_folds\']} max_slices={max_slices} "\n              f"studies={n_studies} total_wall_s={total_wall_s:.2f} "\n              f"decode_s={decode_seconds:.2f} gpu_s={gpu_seconds:.2f} "\n              f"studies/s={studies_per_s:.3f} "\n              f"projected_hours_1300={projected_hours_1300:.2f}", flush=True)\n        with open(os.path.join(os.path.dirname(OUT), "timing.json"),\n                  "w") as fh:\n            json.dump(timing_block, fh)\n        return\n\n    infer_cfg["projected_hours"] = {\n        ",".join(fams): proj_hours_by_group.get(key) for key, fams in groups\n    }\n    with open(os.path.join(os.path.dirname(OUT), "infer_config.json"),\n              "w") as fh:\n        json.dump(infer_cfg, fh)  # re-written now that projected_hours is known\n\n    out_dir = os.path.dirname(OUT)\n    for fam in fam_names:\n        _write_csv(os.path.join(out_dir, f"probs_{fam}.csv"), out_uids,\n                  probs_by_family[fam])\n\n    if len(families) == 1:\n        # a single-family preset (main/main_attn/efficiency/coatnet): write\n        # the family\'s own scores unchanged, byte-identical to today\'s output.\n        fam = fam_names[0]\n        final_probs = probs_by_family[fam]\n        n_failed = n_failed_by_family[fam]\n    else:\n        # a blend preset always rank_blends, even with one surviving family -- see docs/code-notes.md "blend rank_blend passthrough"\n        mats = [np.stack(probs_by_family[fam]) for fam in fam_names]\n        final_probs = list(rank_blend(mats, weights=blend_weights_matrix))\n        n_failed = sum(n_failed_by_family.values())\n\n    _write_csv(OUT, out_uids, final_probs)\n    print(f"submission.csv written ({len(out_uids)} rows, {n_failed} "\n          f"fallback, {(time.time()-t0)/60:.1f} min)", flush=True)\n\n\nif __name__ == "__main__":\n    main()\n'
# END BUNDLED INFER_SUBMIT

# Arm registry: every arm we load and score ourselves (through
# infer_submit's own load_family/decode/score path), keyed by arm id rather
# than family, since two arms may share a family (v6c and pseudo1 are both
# coatnet2, trained on different label files -- see _run_our_arms). Dataset
# mounts at either /kaggle/input/<dataset> or
# /kaggle/input/datasets/patimakornsrinamen/<dataset>.
OUR_ARMS = {
    "v4b":       {"family": "dinov2s",   "label_file": "consensus_v1.csv", "dataset": "rsna-67knee-weights-v4b"},
    "v6c":       {"family": "coatnet2",  "label_file": "consensus_v1.csv", "dataset": "rsna-67knee-weights-v6c"},
    "pseudo1":   {"family": "coatnet2",  "label_file": "pseudo_v1.csv",    "dataset": "rsna-67knee-weights-pseudo-v1"},
    "convnexts": {"family": "convnexts", "label_file": "consensus_v1.csv", "dataset": "rsna-67knee-weights-convnexts"},
    # rs: v6c recipe retrained with random-start windows + train seed 7 (coatnet2_rs, round 10)
    "rs":        {"family": "coatnet2",  "label_file": "consensus_v1.csv", "dataset": "rsna-67knee-weights-rs"},
    "rs2":       {"family": "coatnet2",  "label_file": "consensus_v1.csv", "dataset": "rsna-67knee-weights-rs2"},
    # t4: rs recipe trained on consensus_v1_teach4.csv (Effusion / Synovitis / Contusion /
    # Medial OA <- rs dense OOF, w = 1); fold 0 dense OOF +0.0065 over rs (2026-09-30)
    "t4":        {"family": "coatnet2",  "label_file": "consensus_v1_teach4.csv", "dataset": "rsna-67knee-weights-t4"},
    # t42: t4 retrained with train seed 11 (coatnet2_rs2 arm); 5-fold dense OOF 0.8799 (2026-10-01)
    "t42":       {"family": "coatnet2",  "label_file": "consensus_v1_teach4.csv", "dataset": "rsna-67knee-weights-t42"},
    # t5: rs recipe trained on consensus_v1_t5.csv (teach4 columns + Baker's + Lateral OA <- rs
    # dense OOF, w = 1); 5-fold dense OOF +0.0034 over t4, t4+t5 rank pair 0.8866 (2026-10-02)
    "t5":        {"family": "coatnet2",  "label_file": "consensus_v1_t5.csv", "dataset": "rsna-67knee-weights-t5"},
    # t52: t5 retrained with train seed 11 (coatnet2_rs2 arm); 5-fold dense OOF 0.8837. O1b = O1 with t5 -> t52, a
    # seed-only swap that measures how far one arm swap moves the public score on noise alone (2026-10-21)
    "t52":       {"family": "coatnet2",  "label_file": "consensus_v1_t5.csv", "dataset": "rsna-67knee-weights-t52"},
    # t5f: t5's labels and recipe trained on every non-gold study (coatnet2_rsfull), five train seeds merged as five
    # "folds" (merge_full.py, use_tta pinned off); no OOF, read = O5 = O1 with t5 -> t5f (2026-10-21)
    "t5f":       {"family": "coatnet2",  "label_file": "consensus_v1_t5.csv", "dataset": "rsna-67knee-weights-t5f"},
    # t4f: t4's labels (teach4) the same way, full data x five train seeds; O6 = rs + t4f + t5f, pushed only if O5
    # reads full data as a gain (2026-10-21)
    "t4f":       {"family": "coatnet2",  "label_file": "consensus_v1_teach4.csv", "dataset": "rsna-67knee-weights-t4f"},
    # rsf: rs's labels (consensus_v1) the same way; O7 = rsf + t4f + t5f, trained only if O5 reads full data as a
    # gain (2026-10-21)
    "rsf":       {"family": "coatnet2",  "label_file": "consensus_v1.csv", "dataset": "rsna-67knee-weights-rsf"},
    # l7f: the rsfull recipe on the 7th-place public labels (songqizhou/rsna-knee-llm-labels 8_final_label.csv,
    # CC-BY-NC-SA-4.0; gold rows filled from consensus_v1), x5 seeds; late-submission round (2026-10-27)
    "l7f":       {"family": "coatnet2",  "label_file": "llm7_final.csv", "dataset": "rsna-67knee-weights-l7f"},
    # c7f: the same labels on ConvNeXt-Small (convnexts_rsfull), x5 seeds; the second model class for L3 = l7f + c7f
    "c7f":       {"family": "convnexts", "label_file": "llm7_final.csv", "dataset": "rsna-67knee-weights-c7f"},
    # t11: rs recipe trained on consensus_v1_t11.csv (every finding except Fracture <- rs dense OOF, w = 1);
    # 5-fold dense OOF 0.8910 vs t5 0.8852, a fold-leaky read (distilled labels); first LB test = O2 (2026-10-21)
    "t11":       {"family": "coatnet2",  "label_file": "consensus_v1_t11.csv", "dataset": "rsna-67knee-weights-t11"},
    # t112: t11 retrained with train seed 11 (coatnet2_rs2 arm), local 5090 (2026-10-21); same labels and recipe,
    # so it shares t11's decode group. Used only if O2 reads the t11 labels as a help.
    "t112":      {"family": "coatnet2",  "label_file": "consensus_v1_t11.csv", "dataset": "rsna-67knee-weights-t112"},
}
# self-contained public arms (raptor_infer.py, resgated_runner.py subprocess
# CSVs) -- not in OUR_ARMS, but valid arm ids in BLEND's weight dicts.
PUBLIC_ARM_IDS = ("raptor", "resgated")

LATERAL_MENISCUS = "Lateral Meniscus"
PRESETS = ("full", "public_only", "coat_only")

# BLEND collapses every weight/preset choice into one dict, so a staged copy
# of this file flips ONE dict per LB submission (see the module docstring).
#   preset: "full" (legacy "ours": CoAt branch + transformer branch) |
#     "public_only" | "coat_only" (CoAt branch alone, no transformer).
#   coat: CoAt-branch weights -- any arm id (incl. "raptor"/"resgated"),
#     renormalised per label by rank_blend; used by "full" and "coat_only".
#   transformer: single arm id, used by "full" only.
#   w_coat / w_coat_lateral_meniscus: used by "full" only.
#   public_coat: the submission_public_only.csv weights, unchanged.
#   window_stride: None (training-time non-overlapping windows, the legacy
#     path) | int: our arms score dense overlapping windows every that many
#     slices (infer_submit's "window_stride"; 1 = +0.0071 paired OOF on v6c,
#     ~3x our arms' GPU time). Applies to every OUR_ARMS arm in the run.
BLEND = {  # repo default = the round-7 shipped recipe; staged copies flip this dict per submission
    "preset": "full",
    "coat": {"raptor": 0.375, "v6c": 0.375, "resgated": 0.25},
    "transformer": "v4b",
    "w_coat": 0.60,
    "w_coat_lateral_meniscus": 1.00,
    "public_coat": {"raptor": 0.60, "resgated": 0.40},
    "window_stride": None,
}

# hard runtime budget, mirroring infer_submit.MAX_INFER_HOURS: raptor +
# resgated (subprocess wall time) + the projected total of our own arms'
# decode/score pass at study 200. A host notebook (F1) may lower it with
# RSNA_MAX_FUSION_HOURS and add RSNA_DEADLINE_EPOCH (unix seconds): every scored
# study then checks the wall clock and raises once the deadline has passed, so an
# overrun ends as an exception the host sees instead of the platform's 9 h kill
# (F1 v2, 2026-10-07, timed out with no score).
MAX_FUSION_HOURS = float(os.environ.get("RSNA_MAX_FUSION_HOURS", "8.5"))
DEADLINE_EPOCH = float(os.environ.get("RSNA_DEADLINE_EPOCH", "inf"))

# "ours-only" mode (environment, default off): skip raptor / resgated and the
# fusion, run only the OUR_ARMS ids named in BLEND["coat"], write
# probs_<arm>.csv and OURS_LEG_OUT = their per-label rank blend (BLEND["coat"]
# weights renormalised over our arms, pct-ranked). A host notebook
# (notebooks/f1/build_f1.py) blends that leg into a public stack's
# submission.csv at one global rank weight -- the lineage's 0.943 -> 0.946
# step (docs/competitors-code/d4-blend.md).
#   RSNA_OURS_ONLY=1          enable the mode
#   RSNA_PRIOR_ELAPSED_S=<s>  the host's wall clock so far, folded into the
#                             study-200 projection (MAX_FUSION_HOURS covers the
#                             whole run, not just this script)
#   RSNA_WINDOW_STRIDE=1|3|none  overrides BLEND["window_stride"]
#   RSNA_MAX_FUSION_HOURS / RSNA_DEADLINE_EPOCH  see MAX_FUSION_HOURS above
OURS_ONLY = os.environ.get("RSNA_OURS_ONLY") == "1"
OURS_LEG_OUT = "/kaggle/working/ours_leg.csv"

# "own submission" switch (default off; flipped only in a staged copy, together with
# OURS_ONLY = True): the ours-only leg is also written as submission.csv, so a run of our
# own arms alone is gradable (O1, 2026-10-21: rs + t4 + t5, no public checkpoint). It
# refuses to start if BLEND["coat"] names a public arm. Off = the F1 path, byte-identical.
OURS_SUBMIT = False


def _apply_env_overrides(blend):
    """RSNA_WINDOW_STRIDE -> blend["window_stride"] (in place); returns blend."""
    stride = os.environ.get("RSNA_WINDOW_STRIDE")
    if stride is not None:
        blend["window_stride"] = None if stride.lower() == "none" else int(stride)
    return blend


def _label_matrix(df):
    """(StudyInstanceUID, LABELS...) DataFrame -> (n, 12) float64, LABELS order."""
    return np.stack([df[label].to_numpy(dtype=np.float64) for label in LABELS], axis=1)


def _to_df(uids, values):
    out = pd.DataFrame(np.asarray(values, dtype=np.float64), columns=LABELS)
    out.insert(0, "StudyInstanceUID", list(uids))
    return out


def _pct_rank_matrix(mat):
    """Per-column pct_rank of an (n, k) matrix -> (n, k)."""
    return np.stack([pct_rank(mat[:, c]) for c in range(mat.shape[1])], axis=1)


def _coat_branch(arms, weights):
    """Weighted rank_blend of the named arms (see blend_eval.rank_blend):
    each arm contributes its own per-label percentile rank, weighted and
    renormalised per label by `weights`."""
    names = list(weights)
    mats = [_label_matrix(arms[name]) for name in names]
    w = np.array([[weights[name]] * len(LABELS) for name in names], dtype=np.float64)
    return rank_blend(mats, weights=w)


def _validate_blend(blend):
    """Every arm id BLEND references must be a known arm -- one of OUR_ARMS
    or the two public arms -- checked here (fuse() calls this first thing)
    and again at the top of main(), before any GPU work, so a typo in a
    staged copy fails fast instead of burning GPU hours."""
    preset = blend["preset"]
    assert preset in PRESETS, (
        f"BLEND['preset']: unknown preset {preset!r}, expected one of {PRESETS}")
    known = set(OUR_ARMS) | set(PUBLIC_ARM_IDS)
    for weights_name in ("coat", "public_coat"):
        for arm_id in blend[weights_name]:
            assert arm_id in known, (
                f"BLEND[{weights_name!r}]: unknown arm id {arm_id!r}, "
                f"expected one of {sorted(known)}")
    if preset == "full":
        assert blend["transformer"] in known, (
            f"BLEND['transformer']: unknown arm id {blend['transformer']!r}, "
            f"expected one of {sorted(known)}")
    stride = blend.get("window_stride")
    assert stride is None or (isinstance(stride, int) and stride >= 1), (
        f"BLEND['window_stride']: expected None or a positive int, got {stride!r}")


def fuse(arms, blend):
    """Pure blend math -- no I/O, no Kaggle mounts.

    arms: dict of arm name -> DataFrame (a "StudyInstanceUID" column plus
    one column per LABELS entry), one row per study; every arm must share
    the same StudyInstanceUID order. preset "full" needs "raptor",
    "resgated", and every arm named in blend["coat"]/blend["transformer"];
    "coat_only" needs "raptor", "resgated", and every arm named in
    blend["coat"]; "public_only" needs only "raptor" and "resgated" -- the
    "full"/"coat_only" blends are only computed for their own preset, so a
    "public_only" run never requires our own arms to be present.

    blend: a BLEND-shaped dict (see BLEND's own comment above) -- selects,
    via blend["preset"], which variant is computed as submission_df; the
    public-only variant is always computed (it is always written to
    submission_public_only.csv too -- see the module docstring).

    Returns (submission_df, public_only_df)."""
    _validate_blend(blend)
    preset = blend["preset"]
    uids = arms["raptor"]["StudyInstanceUID"].tolist()
    for name, df in arms.items():
        if df["StudyInstanceUID"].tolist() != uids:
            raise ValueError(f"fuse: {name}'s StudyInstanceUID order != raptor's")

    coat_public = _coat_branch(arms, blend["public_coat"])
    public_df = _to_df(uids, _pct_rank_matrix(coat_public))

    if preset == "public_only":
        return public_df, public_df

    coat = _coat_branch(arms, blend["coat"])

    if preset == "coat_only":
        final_df = _to_df(uids, _pct_rank_matrix(coat))
        return final_df, public_df

    transformer_rank = _pct_rank_matrix(_label_matrix(arms[blend["transformer"]]))
    w = np.array([blend["w_coat_lateral_meniscus"] if label == LATERAL_MENISCUS
                  else blend["w_coat"] for label in LABELS])
    mix = (1.0 - w) * transformer_rank + w * coat
    full_df = _to_df(uids, _pct_rank_matrix(mix))
    return full_df, public_df


def ours_leg(arms, weights):
    """Per-label rank blend of our own arms only: `weights` restricted to the
    OUR_ARMS ids present in `arms` (renormalised per label by rank_blend),
    pct-ranked once more so the leg is itself a submission-shaped matrix in
    [0, 1]. Pure -- no I/O."""
    ours = {name: w for name, w in weights.items() if name in arms and name in OUR_ARMS}
    if not ours:
        raise ValueError(f"ours_leg: no OUR_ARMS id among {sorted(weights)}")
    uids = arms[next(iter(ours))]["StudyInstanceUID"].tolist()
    for name in ours:
        if arms[name]["StudyInstanceUID"].tolist() != uids:
            raise ValueError(f"ours_leg: {name}'s StudyInstanceUID order differs")
    return _to_df(uids, _pct_rank_matrix(_coat_branch(arms, ours)))


def _load_arm_csv(csv_path, canonical_uids):
    """Load one arm's raw CSV, reindexed to canonical_uids order and
    reordered to LABELS columns by name (the resgated runner does not
    reindex or validate against raptor's order or column names -- see its
    own header -- so that identity check happens here)."""
    df = pd.read_csv(csv_path, dtype={"StudyInstanceUID": str})
    missing_labels = [label for label in LABELS if label not in df.columns]
    if missing_labels:
        raise RuntimeError(f"{csv_path}: missing label column(s) {missing_labels}")
    df = df.set_index("StudyInstanceUID").reindex(canonical_uids)
    if df[LABELS].isna().any().any():
        bad = df.index[df[LABELS].isna().any(axis=1)].tolist()
        raise RuntimeError(f"{csv_path}: missing/NaN row(s) for studies {bad[:5]}"
                           f"{'...' if len(bad) > 5 else ''}")
    return df.reset_index()[["StudyInstanceUID"] + LABELS]


def _run_subprocess_arm(name, script_path, src_text, env_key, out_csv,
                        extra_env=None):
    with open(script_path, "w") as fh:
        fh.write(src_text)
    env = dict(os.environ)
    env[env_key] = out_csv
    env.update(extra_env or {})
    t0 = time.time()
    proc = subprocess.run([sys.executable, script_path], env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} ({script_path}) failed, exit code {proc.returncode}")
    return time.time() - t0


def _required_our_arm_ids(blend):
    """Which OUR_ARMS ids the active preset's weights actually reference --
    e.g. a "coat_only" preset with no "v4b" in blend["coat"] never loads
    DINOv2, and "public_only" never loads any of our own arms."""
    preset = blend["preset"]
    names = set()
    if preset in ("full", "coat_only"):
        names |= set(blend["coat"])
    if preset == "full":
        names.add(blend["transformer"])
    return sorted(name for name in names if name in OUR_ARMS)


def _run_our_arms(dev, prior_elapsed_s, arm_ids):
    """Run the given OUR_ARMS ids (any mix of families -- e.g. v6c and
    pseudo1 are both coatnet2) through infer_submit's own
    load_family/decode/score path (v4 recipe), grouped by (recipe, window)
    exactly as infer_submit.main() does, forcing all-attn window pooling
    (the main_attn shipping stance -- see infer_submit.py's PRESETS comment;
    the tuned window_pool table does not transfer here).

    Arms are loaded one at a time: I.FAMILY_WEIGHTS[family] and
    I.EXPECTED_LABEL_FILE are module globals on the embedded infer_submit,
    so two coatnet2 arms with different label files must not be loaded
    concurrently. dinov2s's load_family reads I.WEIGHTS_CANDIDATES instead
    of I.FAMILY_WEIGHTS["dinov2s"] (see infer_submit.py's own comment), so
    that global is set too when the arm's family is dinov2s.

    prior_elapsed_s: seconds already spent on raptor + resgated, folded into
    the study-200 projection so the fail-fast (MAX_FUSION_HOURS) reflects
    the whole run's wall clock, not just this function's own share -- and,
    critically, raises WHILE still at study 200 rather than after finishing
    every study, the same as infer_submit.check_projection.

    Returns (test_uids, {arm_id: [prob_row, ...]}, projected_total_hours,
    elapsed_s). projected_total_hours is None if fewer than 200 test
    studies exist (no checkpoint is ever reached), which is always true when
    arm_ids is empty (a "public_only" run never decodes a single study for
    our own arms)."""
    import importlib
    if WORKING not in sys.path:
        sys.path.insert(0, WORKING)
    I = importlib.import_module("infer_submit")

    loaded = {}
    for arm_id in arm_ids:
        spec = OUR_ARMS[arm_id]
        family = spec["family"]
        weights_dirs = (
            f"/kaggle/input/{spec['dataset']}",
            f"/kaggle/input/datasets/patimakornsrinamen/{spec['dataset']}",
        )
        I.FAMILY_WEIGHTS[family] = weights_dirs
        if family == "dinov2s":
            I.WEIGHTS_CANDIDATES = weights_dirs
        I.EXPECTED_LABEL_FILE = spec["label_file"]
        models, cfg, weights_dir = I.load_family(family, dev)
        use_tta = I._resolve_use_tta({"use_tta": None}, cfg)
        loaded[arm_id] = (models, cfg, weights_dir, use_tta)

    use_amp = dev == "cuda"
    recipe_by_arm = {a: I.recipe_kwargs(loaded[a][1]) for a in arm_ids}
    window_by_arm = {a: loaded[a][1].get("window") for a in arm_ids}
    groups, group_index = [], {}
    for a in arm_ids:
        key = (tuple(sorted(recipe_by_arm[a].items())), window_by_arm[a])
        if key not in group_index:
            group_index[key] = len(groups)
            groups.append((key, []))
        groups[group_index[key]][1].append(a)

    by_study = I.read_series_table(os.path.join(I.COMP, "test_series.csv"))
    test_df = pd.read_csv(os.path.join(I.COMP, "test.csv"),
                          dtype={"StudyInstanceUID": str})
    test_uids = test_df["StudyInstanceUID"].tolist()
    work_items = [(u, by_study.get(u, [])) for u in test_uids]

    probs_by_arm = {a: [None] * len(test_uids) for a in arm_ids}
    window_pool_attn = {name: "attn" for name in LABELS}
    t_start = time.time()
    projected_total_hours = None
    for key, arms_in_group in groups:
        group_recipe, group_window = dict(key[0]), key[1]
        slice_align = group_window or 1
        gt0 = time.time()
        decoded = I.decode_all(work_items, group_recipe, max_slices=None,
                               slice_align=slice_align)
        for k, (uid, vol, filled) in enumerate(decoded):
            assert uid == test_uids[k], (
                f"decode order drifted at index {k} ({uid!r} != {test_uids[k]!r})")
            for a in arms_in_group:
                models, cfg, _, use_tta = loaded[a]
                prior = cfg.get("prior", I.PRIOR_FALLBACK)
                probs, _failed = I.score_study(
                    models, dev, use_amp, use_tta, group_window,
                    window_pool_attn, prior, uid, vol, filled,
                    stride=BLEND.get("window_stride"))
                probs_by_arm[a][k] = np.asarray(probs, dtype=np.float64)
            if time.time() > DEADLINE_EPOCH:
                raise RuntimeError(
                    f"RSNA_DEADLINE_EPOCH passed at study {k + 1}/{len(test_uids)}")
            if k + 1 == 200:
                checkpoint_hours = I.projected_hours(
                    time.time() - gt0, k + 1, len(test_uids))
                projected_total_hours = checkpoint_hours + prior_elapsed_s / 3600.0
                print(f"projected total: {projected_total_hours:.2f} h", flush=True)
                if projected_total_hours > MAX_FUSION_HOURS:
                    raise RuntimeError(
                        f"projected {projected_total_hours:.2f} h exceeds "
                        f"MAX_FUSION_HOURS={MAX_FUSION_HOURS} h")
    return test_uids, probs_by_arm, projected_total_hours, time.time() - t_start


def main():
    _apply_env_overrides(BLEND)
    _validate_blend(BLEND)  # before any GPU work -- fail fast on a bad arm id
    if OURS_SUBMIT:
        assert OURS_ONLY, "OURS_SUBMIT needs OURS_ONLY"
        assert BLEND["preset"] == "coat_only", f"OURS_SUBMIT needs preset coat_only, got {BLEND['preset']!r}"
        public = sorted(set(BLEND["coat"]) - set(OUR_ARMS))
        assert not public, f"OURS_SUBMIT: BLEND['coat'] names non-own arms {public}"
        print(f"OURS_SUBMIT_MODE arms={sorted(BLEND['coat'])} "
              f"window_stride={BLEND.get('window_stride')}", flush=True)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    os.makedirs(WORKING, exist_ok=True)
    with open(INFER_SUBMIT_SCRIPT, "w") as fh:
        fh.write(_INFER_SRC)

    runtimes = {}
    prior_elapsed_s = float(os.environ.get("RSNA_PRIOR_ELAPSED_S", "0"))
    if not OURS_ONLY:
        runtimes["raptor"] = _run_subprocess_arm(
            "raptor_infer.py", RAPTOR_SCRIPT, _RAPTOR_SRC,
            "RAPTOR_OUTPUT_CSV", RAPTOR_CSV,
            extra_env={"RAPTOR_ARM_NAMES": RAPTOR_ARM_NAMES})
        runtimes["resgated"] = _run_subprocess_arm(
            "resgated_runner.py", RESGATED_SCRIPT, _RESGATED_SRC,
            "RESGATED_OUTPUT_CSV", RESGATED_CSV)
        prior_elapsed_s += runtimes["raptor"] + runtimes["resgated"]

    our_arm_ids = _required_our_arm_ids(BLEND)
    test_uids, our_probs, projected_total_hours, our_elapsed_s = _run_our_arms(
        dev, prior_elapsed_s, our_arm_ids)
    for arm_id in our_arm_ids:
        runtimes[arm_id] = our_elapsed_s  # one shared decode/score pass

    arms = {}
    if not OURS_ONLY:
        arms["raptor"] = _load_arm_csv(RAPTOR_CSV, test_uids)
        arms["resgated"] = _load_arm_csv(RESGATED_CSV, test_uids)
    for arm_id in our_arm_ids:
        arms[arm_id] = _to_df(test_uids, np.stack(our_probs[arm_id]))

    out_dir = os.path.dirname(OUT)
    for name, df in arms.items():
        df.to_csv(os.path.join(out_dir, f"probs_{name}.csv"), index=False)

    if OURS_ONLY:
        leg_df = ours_leg(arms, BLEND["coat"])
        leg_df.to_csv(OURS_LEG_OUT, index=False)
        if OURS_SUBMIT:
            leg_df.to_csv(OUT, index=False)
        summary = (f"ours_leg.csv written ({len(leg_df)} rows, arms={our_arm_ids}, "
                   f"window_stride={BLEND.get('window_stride')}); "
                   + ("submission.csv = ours_leg.csv (OURS_SUBMIT)" if OURS_SUBMIT
                      else "submission.csv untouched"))
    else:
        submission_df, public_df = fuse(arms, BLEND)
        submission_df.to_csv(OUT, index=False)
        public_df.to_csv(PUBLIC_ONLY_OUT, index=False)
        summary = (f"submission.csv written ({len(submission_df)} rows, "
                   f"preset={BLEND['preset']}); submission_public_only.csv written "
                   f"({len(public_df)} rows)")

    # the fail-fast itself already happened inside _run_our_arms, at study
    # 200 -- projected_total_hours here is just the value to record.
    fusion_cfg = {
        "blend": BLEND,
        "ours_only": OURS_ONLY,
        "arms_run": ([] if OURS_ONLY else list(PUBLIC_ARM_IDS)) + our_arm_ids,
        "runtimes_s": runtimes,
        "projected_total_hours_at_study_200": projected_total_hours,
        "total_elapsed_s": time.time() - t0,
    }
    if OURS_SUBMIT:
        fusion_cfg["ours_submit"] = True
    with open(os.path.join(out_dir, "fusion_config.json"), "w") as fh:
        json.dump(fusion_cfg, fh, indent=2)

    print(summary, flush=True)


if __name__ == "__main__":
    main()
