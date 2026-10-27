# notebooks/fusion/raptor_infer.py
#
# Standalone extraction of the `_KE_SRC` cell from the public kernel
# "the-bee-s-knees-final-rsna-push" (Kaggle kernel
# prvsiyan/the-bee-s-knees-final-rsna-push, pulled to
# C:/kaggle/staging/competitors/the-bee-s-knees-final-rsna-push/ on 2026-09-09).
# Source authenticated by the pulled notebook's own SHA-256
# 6b03919bbca5fe61e37c5ea6546609a771cd7db6503121f3e2dacdd8c9b6b052.
#
# Original Raptor CoAtNet inference recipe and weights: dreaddevelopment
# (Johnathan Wagner), datasets `raptor-knee-maxspan`, `raptor-knee-native384`,
# `raptor-knee-native384dense`, `raptor-knee-arms` -- all four CC0-1.0.
# Assembled/reproduced (per the source kernel's licence table) from Mattia
# Angeli's V27 notebook, Apache-2.0. This file keeps that licence for the
# reproduced code; the CC0-1.0 terms above cover the checkpoint weights only.
#
# Everything between the BEGIN/END sentinels below is byte-identical to the
# `_KE_SRC` string literal in the source notebook's cell 32 (verified by
# diffing this block against `ast.literal_eval()` of that assignment) -- do
# not edit it. It defines Raptor's own
# 4-sub-arm (maxspan-v5 / native384dense-v10 / maxspan-v5-reverse /
# native384-v8) rank blend and writes `/kaggle/working/_raptor.csv` in
# `sample_submission` column order; no cross-arm fusion (Raptor vs. our arms
# vs. resgated) happens here -- that lives in `fusion_submit.py`.
#
# `find_test_root`/`find_weight_file` inside the verbatim body already walk
# all of /kaggle/input recursively, so they resolve both the flat
# /kaggle/input/<name>/... mount layout and the nested
# /kaggle/input/datasets/<owner>/<name>/... layout without needing any
# changes here.
#
# Argument/env contract (all optional; defaults reproduce the source kernel's
# hardcoded behaviour exactly):
#   RAPTOR_ARM_NAMES   comma-separated subset of the ARMS[*]["name"] values to
#                      run, e.g. "maxspan-v5,native384-v8" (default: all four)
#   RAPTOR_CKPT_DIRS   os.pathsep-separated extra directories searched for a
#                      checkpoint file before the built-in
#                      /kaggle/input/raptor-knee-* search list
#   RAPTOR_OUTPUT_CSV  output path (default: /kaggle/working/_raptor.csv, the
#                      path the source kernel hardcodes)
# No other parameterisation is added: the slot layout, window sampling, and
# per-arm blend weights are exactly as in the source cell.
#
# Run standalone: `python raptor_infer.py` (writes _raptor.csv and exits;
# weights absent -> RuntimeError naming the missing checkpoint file).
# Import-safe: `import raptor_infer` defines everything below without
# running main().

# ==== BEGIN VERBATIM _KE_SRC (source notebook cell 32) ====
import os, glob, time, gc, hashlib
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
IMG = 336
CROP_MM = 140.0
SPAN_LO, SPAN_HI = 0.02, 0.98
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14),
         ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(slot[2] for slot in SLOTS)
K_EVAL = 62
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_SLOTS64 = [("Sagittal", 1, 18), ("Sagittal", 0, 14),
            ("Coronal", 1, 12), ("Coronal", 0, 8), ("Axial", -1, 12)]
_SLOTS44 = [("Sagittal", 1, 12), ("Sagittal", 0, 10),
            ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]
ARMS = [
    {"name": "maxspan-v5", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.60},
    {"name": "native384dense-v10", "file": "raptor_ft_coatnet_v10_full.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": False, "w": 0.10},
    {"name": "maxspan-v5-reverse", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 336, "slots": _SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "reverse": True, "w": 0.10},
    {"name": "native384-v8", "file": "raptor_ft_coatnet_v8_full_swa.pt",
     "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384,
     "img": 384, "slots": _SLOTS44, "span": (0.06, 0.94), "k_eval": 42,
     "reverse": False, "w": 0.20},
]

def build_backbone(arch, pretrained=False):
    hybrid = arch.startswith(('maxvit', 'maxxvit', 'coatnet', 'coat_', 'convnext'))
    is_vit = not hybrid and any((k in arch for k in ('vit', 'deit', 'dinov2', 'eva', 'beit')))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool='token', dynamic_img_size=True)
    else:
        kw.update(global_pool='avg')
    return timm.create_model(arch, **kw)

class RaptorClassifier(nn.Module):

    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum('bkn,bkf->bnf', a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))

def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location='cpu', weights_only=False)
    arch = ck.get('arch', arch_default)
    ck_res = int(ck.get('res', res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck['model'], strict=True)
    model.eval().to(device)
    del ck
    gc.collect()
    return (model, ck_res)

def load_refit_head(pt_path, feature_dim, device):
    ck = torch.load(pt_path, map_location='cpu', weights_only=False)
    state = ck.get('model', ck)
    head = RaptorClassifier(nn.Identity(), F_dim=int(feature_dim))
    head_state = {name: tensor for name, tensor in state.items()
                  if not name.startswith('backbone.')}
    head.load_state_dict(head_state, strict=True)
    head.eval().to(device)
    del ck, state, head_state
    gc.collect()
    return head

def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = (int(valid.min()), int(valid.max()))
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]

def eval_windows(vol, mask, k, res, norm=NORM):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode='bilinear', align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == 'imagenet':
        x = (x - _MEAN) / _STD
    return x

@torch.no_grad()
def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != 'cpu' and str(device).startswith('cuda')
    if use_cuda:
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
        except RuntimeError:
            torch.cuda.empty_cache()
            o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
    o = torch.sigmoid(model(x).float())
    return o[0].cpu().numpy()

@torch.no_grad()
def infer_probs_two_heads(model, refit_head, xwins, device):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != 'cpu' and str(device).startswith('cuda')
    def forward_heads():
        features = model.encode(x)
        original = torch.sigmoid(model.head(features).float())
        refitted = torch.sigmoid(refit_head.head(features).float())
        return (original[0].cpu().numpy(), refitted[0].cpu().numpy())
    if use_cuda:
        try:
            with torch.autocast('cuda', dtype=torch.float16):
                return forward_heads()
        except RuntimeError:
            torch.cuda.empty_cache()
            return forward_heads()
    return forward_heads()

def rankpct(x):
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, x.shape[0] - 1)

def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + '/*.dcm')
        recs = []
        ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and (len(iop) == 6):
                    r = np.array(iop[:3], float)
                    c = np.array(iop[3:], float)
                    n = np.cross(r, c)
                    pos = float(np.dot(np.array(ipp, float), n))
                else:
                    pos = float(getattr(h, 'InstanceNumber', 0) or 0)
                ps = getattr(h, 'PixelSpacing', None)
                ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps)
                recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return ([(f, ps) for _, f, ps in recs], med_ps)

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape
        cpx = int(round(CROP_MM / max(ps, 0.001)))
        cpx = min(cpx, min(h, w))
        y0 = (h - cpx) // 2
        x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return (order_and_meta, read_px, mm_crop_resize)

def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None

def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx = 0
    used = set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = (int(n * SPAN_LO), int(n * SPAN_HI) - 1)
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []
        pss = []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp))
                pss.append(ps)
            except Exception:
                arrs.append(None)
                pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = (0.0, 1.0)
        for a, ps in zip(arrs, pss):
            if idx >= MAXS:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-06), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8)
            idx += 1
        if idx >= MAXS:
            break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return (vol, mask)

def find_test_root():
    cands = ['/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection']
    for b in cands:
        if os.path.exists(b + '/test.csv'):
            return b
    for d, _, f in os.walk('/kaggle/input'):
        if 'test.csv' in f and (os.path.isdir(d + '/test_series') or os.path.isdir(d + '/test_images')):
            return d
    for d, _, f in os.walk('/kaggle/input'):
        if 'test.csv' in f:
            return d
    raise RuntimeError('no test root under /kaggle/input')

def find_weight_file(fname):
    direct = [f'/kaggle/input/raptor-knee-maxspan/{fname}', f'/kaggle/input/raptor-knee-native384dense/{fname}', f'/kaggle/input/raptor-knee-native384/{fname}', f'/kaggle/input/raptor-knee-arms/{fname}', f'/kaggle/input/raptor-knee-arms/1/{fname}', f'/kaggle/input/raptor-cnn336/{fname}']
    for p in direct:
        if os.path.exists(p):
            return p
    for d in sorted(glob.glob('/kaggle/input/*/')):
        if 'competition' in d.lower():
            continue
        hits = glob.glob(os.path.join(d, '**', fname), recursive=True)
        if hits:
            return hits[0]
    raise RuntimeError(f'{fname} not found under /kaggle/input')

def find_optional_verified_weight(fname, expected_sha256, root='/kaggle/input'):
    hits = []
    for directory in sorted(glob.glob(os.path.join(root, '*/'))):
        if 'competition' in directory.lower():
            continue
        hits.extend(glob.glob(os.path.join(directory, '**', fname), recursive=True))
    for path in sorted(set(hits)):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() == expected_sha256:
            return path
        print(f'[head-refit] ignored hash-mismatched optional checkpoint: {path}', flush=True)
    return None

def main():
    import pandas as pd
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev} | gpus {torch.cuda.device_count()} | torch {torch.__version__}", flush=True)
    root = find_test_root()
    tsdir = root + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = root + "/test_images"
    print("test root:", root, "| series dir:", tsdir, flush=True)
    test = pd.read_csv(root + "/test.csv")
    test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = pd.read_csv(root + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    series = {key: frame.to_dict("records") for key, frame in tser.groupby("StudyInstanceUID")}
    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)
    sub_cols = ["StudyInstanceUID"] + LAB
    sample = os.path.join(root, "sample_submission.csv")
    if os.path.exists(sample):
        sub_cols = list(pd.read_csv(sample, nrows=1).columns)
    reader = _make_reader()
    n_study, n_arm = len(test_ids), len(ARMS)
    arm_probs = [np.full((n_study, len(LAB)), 0.5, np.float32) for _ in range(n_arm)]
    for arm_index, arm in enumerate(ARMS):
        globals()["IMG"] = int(arm["img"])
        globals()["SLOTS"] = list(arm["slots"])
        globals()["MAXS"] = sum(slot[2] for slot in SLOTS)
        globals()["SPAN_LO"], globals()["SPAN_HI"] = map(float, arm["span"])
        globals()["K_EVAL"] = int(arm["k_eval"])
        weight_path = find_weight_file(arm["file"])
        model, resolution = load_model(weight_path, arm["arch"], arm["res"], dev)
        print(f"[arm {arm_index}] {arm['name']} | img {IMG} | slices {MAXS} | "
              f"span {SPAN_LO:.2f}-{SPAN_HI:.2f} | windows {K_EVAL} | "
              f"res {resolution} | {time.time() - t0:.0f}s", flush=True)
        for study_index, study_uid in enumerate(test_ids):
            try:
                volume, mask = build_study(study_uid, series, tsdir, reader)
                windows = eval_windows(volume, mask, k=K_EVAL, res=resolution, norm=NORM)
                if bool(arm.get("reverse", False)):
                    windows = windows.flip(1).contiguous()
                arm_probs[arm_index][study_index] = infer_probs(model, windows, dev)
                del volume, mask, windows
            except Exception as error:
                print(f"  [arm {arm_index}] study {study_index} {study_uid[:16]} FALLBACK "
                      f"({type(error).__name__}: {error})", flush=True)
            if (study_index + 1) % 100 == 0 or study_index + 1 == n_study:
                print(f"  [arm {arm_index}] {study_index + 1}/{n_study} | "
                      f"{time.time() - t0:.0f}s", flush=True)
        del model
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[arm {arm_index}] done + freed | {time.time() - t0:.0f}s", flush=True)
    weights = np.array([float(arm.get("w", 1.0)) for arm in ARMS], dtype=np.float64)
    weights /= weights.sum()
    print(f"[blend] global probability mean w="
          f"{dict(zip([arm['name'] for arm in ARMS], weights.round(4)))}", flush=True)
    probability_blend = np.tensordot(
        weights, np.stack([np.clip(values, 0, 1) for values in arm_probs]), axes=(0, 0))
    ranks = rankpct(probability_blend)
    if not np.isfinite(ranks).all():
        ranks[~np.isfinite(ranks)] = 0.5
    submission = pd.DataFrame(ranks.astype(np.float32), columns=LAB)
    submission.insert(0, "StudyInstanceUID", test_ids)
    submission = submission[sub_cols]
    assert submission["StudyInstanceUID"].tolist() == test_ids
    assert np.isfinite(submission[LAB].values).all()
    out = "/kaggle/working/_raptor.csv"
    submission.to_csv(out, index=False)
    print("wrote", out, "|", len(submission), "rows x", len(submission.columns), "cols", flush=True)
    print(submission.head().to_string(index=False), flush=True)
    print(f"DONE {time.time() - t0:.0f}s", flush=True)
# ==== END VERBATIM _KE_SRC ====


def _apply_raptor_env_overrides():
    """Apply the documented RAPTOR_* env-var overrides in place, without
    editing the verbatim body above."""
    arm_names = os.environ.get("RAPTOR_ARM_NAMES")
    if arm_names:
        wanted = {n.strip() for n in arm_names.split(",") if n.strip()}
        filtered = [a for a in ARMS if a["name"] in wanted]
        if not filtered:
            raise ValueError(f"RAPTOR_ARM_NAMES matched no arm in {wanted}")
        globals()["ARMS"] = filtered

    extra_dirs = os.environ.get("RAPTOR_CKPT_DIRS")
    if extra_dirs:
        extra = [d for d in extra_dirs.split(os.pathsep) if d]
        _orig_find_weight_file = find_weight_file

        def _find_weight_file_with_extra_dirs(fname, _extra=extra,
                                              _orig=_orig_find_weight_file):
            for d in _extra:
                p = os.path.join(d, fname)
                if os.path.exists(p):
                    return p
            return _orig(fname)

        globals()["find_weight_file"] = _find_weight_file_with_extra_dirs


def run():
    """Entry point: apply env overrides, run main(), relocate output if asked."""
    _apply_raptor_env_overrides()
    main()
    out_override = os.environ.get("RAPTOR_OUTPUT_CSV")
    if out_override and out_override != "/kaggle/working/_raptor.csv":
        os.replace("/kaggle/working/_raptor.csv", out_override)


if __name__ == "__main__":
    run()
