"""CPU unit test for Model(spatial_pool="label_attn") vs the pre-change model.

Plain script (no pytest on this PC): `py src/test_spatial_pool.py`.
The reference half loads `git show HEAD:rsna-67knee/src/model.py`, so it is
only meaningful while HEAD predates the spatial_pool change."""
import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
import model  # noqa: E402

KW = dict(pretrained=False, img_size=280, arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k",
          grad_ckpt=False, unfreeze="stem_stage0", drop_path=0.1, n_slots=6)


def load_head_model():
    try:
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=SRC, text=True).strip()
        src = subprocess.check_output(["git", "show", "HEAD:rsna-67knee/src/model.py"], cwd=top)
    except Exception as e:  # noqa: BLE001
        print(f"WARN git show HEAD:model.py failed ({e}); skipping the pre-change reference half")
        return None
    path = os.path.join(tempfile.mkdtemp(), "model_head.py")
    with open(path, "wb") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("model_head", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(mod, **extra):
    torch.manual_seed(0)
    return mod.Model(**KW, **extra).eval()


model_head = load_head_model()
old = build(model_head) if model_head is not None else None
off = build(model)
on = build(model, spatial_pool="label_attn")

if old is not None:
    assert old.state_dict().keys() == off.state_dict().keys()
    for k, v in old.state_dict().items():
        assert torch.equal(v, off.state_dict()[k]), k
assert set(on.state_dict()) - set(off.state_dict()) == {"spatial_query"}

torch.manual_seed(1)
x = torch.rand(5, 3, 280, 280)
slots = torch.tensor([0, 1, 2, 3, 0])
owner = torch.tensor([0, 0, 0, 1, 1])
n = 2
with torch.no_grad():
    if old is not None:
        assert torch.equal(old(x, slots, owner, n), off(x, slots, owner, n))
        assert torch.equal(old(x[:, 0], slots, owner, n), off(x[:, 0], slots, owner, n))
    h = on.embed(x, slots)
    assert h.shape == (5, 12, 256), h.shape
    L = on.head(h, owner, 2)
    assert L.shape == (2, 12) and torch.isfinite(L).all()
    assert on.head(h, torch.arange(5), 5).shape == (5, 12)
    ref = off(x, slots, owner, n)
    assert torch.allclose(L, ref, atol=1e-5), (L - ref).abs().max()
    assert on.embed(x[:, 0], slots).shape == (5, 12, 256)

on.train()
L = on(x, slots, owner, n)
L.sum().backward()
assert on.spatial_query.grad is not None and on.spatial_query.grad.abs().max() > 0

with torch.no_grad():
    on.spatial_query.normal_(0, 0.5)
on.eval()
with torch.no_grad():
    assert (on(x, slots, owner, n) - off(x, slots, owner, n)).abs().max() > 1e-4

    sd = on.state_dict()
    on2 = build(model, spatial_pool="label_attn")
    on2.load_state_dict(sd, strict=True)
    assert torch.equal(on2(x, slots, owner, n), on(x, slots, owner, n))

for bad in (lambda: model.Model(**KW, spatial_pool="bogus"),
            lambda: model.Model(pretrained=False, img_size=224, spatial_pool="label_attn")):
    try:
        bad()
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

print("PASS test_spatial_pool")
