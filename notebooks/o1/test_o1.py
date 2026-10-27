"""Checks for build_o1.py: the staged copy differs from fusion_submit.py in exactly the five switch lines, imports
with OURS_ONLY / OURS_SUBMIT on and a BLEND that names only our arms, and kernel-metadata.json mounts only our own
datasets plus the competition.

  py notebooks/o1/test_o1.py
"""
import difflib
import importlib.util
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import build_o1  # noqa: E402


def _diff(a, b):
    return [l for l in difflib.unified_diff(a.split("\n"), b.split("\n"), lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]


if __name__ == "__main__":
    base = open(build_o1.FUSION, encoding="utf-8").read()
    with tempfile.TemporaryDirectory() as tmp:
        path = build_o1.build(tmp)
        staged = open(path, encoding="utf-8").read()
        meta = json.load(open(os.path.join(tmp, "kernel-metadata.json")))
        spec = importlib.util.spec_from_file_location("o1_fusion", path)
        fs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fs)
    d = _diff(base, staged)
    added = [l[1:] for l in d if l[0] == "+"]
    assert len(d) == 10 and len(added) == 5, d
    assert fs.OURS_ONLY is True and fs.OURS_SUBMIT is True
    assert fs.BLEND["preset"] == "coat_only" and fs.BLEND["window_stride"] == 1, fs.BLEND
    assert fs.BLEND["coat"] == {"rs": 1.0, "t4": 1.0, "t5": 1.0}, fs.BLEND["coat"]
    assert not set(fs.BLEND["coat"]) & set(fs.PUBLIC_ARM_IDS)
    fs._validate_blend(fs.BLEND)
    assert fs._required_our_arm_ids(fs.BLEND) == ["rs", "t4", "t5"]
    spec = importlib.util.spec_from_file_location("base_fusion", build_o1.FUSION)
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    assert bs.OURS_SUBMIT is False and bs.BLEND["preset"] == "full"
    assert all(getattr(fs, v) == getattr(bs, v) for v in ("_INFER_SRC", "_RAPTOR_SRC", "_RESGATED_SRC", "OUR_ARMS"))
    assert all(s.startswith("patimakornsrinamen/") for s in meta["dataset_sources"]), meta["dataset_sources"]
    assert meta["dataset_sources"] == ["patimakornsrinamen/rsna-knee-src", "patimakornsrinamen/rsna-67knee-weights-rs",
                                       "patimakornsrinamen/rsna-67knee-weights-t4", "patimakornsrinamen/rsna-67knee-weights-t5"]
    assert meta["competition_sources"] == ["rsna-knee-abnormality-detection"] and meta["kernel_sources"] == []
    assert meta["id"] == "patimakornsrinamen/rsna-67knee-o1" and meta["is_private"] == "true"
    assert meta["docker_image"] == build_o1.DOCKER_IMAGE
    print("O1 == fusion_submit.py except 5 switch lines:\n  " + "\n  ".join(added))
    print("O1 datasets: " + ", ".join(meta["dataset_sources"]))
    with tempfile.TemporaryDirectory() as tmp:   # O2: O1 with rs swapped for t11
        path2 = build_o1.build(tmp, "patimakornsrinamen/rsna-67knee-o2", ("t4", "t5", "t11"))
        staged2 = open(path2, encoding="utf-8").read()
        meta2 = json.load(open(os.path.join(tmp, "kernel-metadata.json")))
        spec = importlib.util.spec_from_file_location("o2_fusion", path2)
        fs2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fs2)
    d = _diff(staged, staged2)
    assert len(d) == 2 and all('"coat": ' in l for l in d), d
    assert fs2.BLEND["coat"] == {"t4": 1.0, "t5": 1.0, "t11": 1.0} and fs2.OURS_SUBMIT is True
    assert fs2._required_our_arm_ids(fs2.BLEND) == ["t11", "t4", "t5"]
    assert fs2.OUR_ARMS["t11"]["label_file"] == "consensus_v1_t11.csv"
    assert meta2["dataset_sources"] == ["patimakornsrinamen/rsna-knee-src", "patimakornsrinamen/rsna-67knee-weights-t4",
                                        "patimakornsrinamen/rsna-67knee-weights-t5", "patimakornsrinamen/rsna-67knee-weights-t11"]
    assert meta2["id"] == "patimakornsrinamen/rsna-67knee-o2"
    print("O2 == O1 except the coat line: " + d[1][1:].strip())
    with tempfile.TemporaryDirectory() as tmp:   # O3: O2 with t4 swapped for t112
        path3 = build_o1.build(tmp, "patimakornsrinamen/rsna-67knee-o3", ("t5", "t11", "t112"))
        staged3 = open(path3, encoding="utf-8").read()
        meta3 = json.load(open(os.path.join(tmp, "kernel-metadata.json")))
        spec = importlib.util.spec_from_file_location("o3_fusion", path3)
        fs3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fs3)
    d = _diff(staged2, staged3)
    assert len(d) == 2 and all('"coat": ' in l for l in d), d
    assert fs3.BLEND["coat"] == {"t5": 1.0, "t11": 1.0, "t112": 1.0} and fs3.OURS_SUBMIT is True
    fs3._validate_blend(fs3.BLEND)
    assert fs3._required_our_arm_ids(fs3.BLEND) == ["t11", "t112", "t5"]
    assert fs3.OUR_ARMS["t112"] == dict(fs3.OUR_ARMS["t11"], dataset="rsna-67knee-weights-t112")
    assert meta3["dataset_sources"] == ["patimakornsrinamen/rsna-knee-src", "patimakornsrinamen/rsna-67knee-weights-t5",
                                        "patimakornsrinamen/rsna-67knee-weights-t11",
                                        "patimakornsrinamen/rsna-67knee-weights-t112"]
    assert meta3["id"] == "patimakornsrinamen/rsna-67knee-o3"
    print("O3 == O2 except the coat line: " + d[1][1:].strip())
    with tempfile.TemporaryDirectory() as tmp:   # O4 (tie / t11-hurts branch): five seed arms, no t11
        path4 = build_o1.build(tmp, "patimakornsrinamen/rsna-67knee-o4", ("rs", "rs2", "t4", "t42", "t5"))
        meta4 = json.load(open(os.path.join(tmp, "kernel-metadata.json")))
        spec = importlib.util.spec_from_file_location("o4_fusion", path4)
        fs4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fs4)
    fs4._validate_blend(fs4.BLEND)
    assert fs4._required_our_arm_ids(fs4.BLEND) == ["rs", "rs2", "t4", "t42", "t5"]
    assert fs4.OUR_ARMS["t42"] == dict(fs4.OUR_ARMS["t4"], dataset="rsna-67knee-weights-t42")
    assert meta4["dataset_sources"][1:] == [f"patimakornsrinamen/rsna-67knee-weights-{a}"
                                            for a in ("rs", "rs2", "t4", "t42", "t5")]
    print("O4 datasets: " + ", ".join(meta4["dataset_sources"]))
    for name, swap in (("o1b", "t52"), ("o5", "t5f")):   # O1 with t5 swapped for its seed / full-data replicate
        with tempfile.TemporaryDirectory() as tmp:
            p = build_o1.build(tmp, f"patimakornsrinamen/rsna-67knee-{name}", ("rs", "t4", swap))
            st = open(p, encoding="utf-8").read()
            m = json.load(open(os.path.join(tmp, "kernel-metadata.json")))
            spec = importlib.util.spec_from_file_location(f"{name}_fusion", p)
            f = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(f)
        d = _diff(staged, st)
        assert len(d) == 2 and all('"coat": ' in l for l in d), d
        f._validate_blend(f.BLEND)
        assert f._required_our_arm_ids(f.BLEND) == ["rs", "t4", swap]
        assert f.OUR_ARMS[swap] == dict(f.OUR_ARMS["t5"], dataset=f"rsna-67knee-weights-{swap}")
        assert m["dataset_sources"] == ["patimakornsrinamen/rsna-knee-src", "patimakornsrinamen/rsna-67knee-weights-rs",
                                        "patimakornsrinamen/rsna-67knee-weights-t4",
                                        f"patimakornsrinamen/rsna-67knee-weights-{swap}"]
        assert m["id"] == f"patimakornsrinamen/rsna-67knee-{name}"
        print(f"{name.upper()} == O1 except the coat line: " + d[1][1:].strip())
        if name == "o5":
            staged5 = st
    with tempfile.TemporaryDirectory() as tmp:   # O6: O5 with t4 swapped for t4f
        p = build_o1.build(tmp, "patimakornsrinamen/rsna-67knee-o6", ("rs", "t4f", "t5f"))
        st6 = open(p, encoding="utf-8").read()
        m6 = json.load(open(os.path.join(tmp, "kernel-metadata.json")))
        spec = importlib.util.spec_from_file_location("o6_fusion", p)
        f6 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(f6)
    d = _diff(staged5, st6)
    assert len(d) == 2 and all('"coat": ' in l for l in d), d
    f6._validate_blend(f6.BLEND)
    assert f6._required_our_arm_ids(f6.BLEND) == ["rs", "t4f", "t5f"]
    assert f6.OUR_ARMS["t4f"] == dict(f6.OUR_ARMS["t4"], dataset="rsna-67knee-weights-t4f")
    assert m6["dataset_sources"][1:] == [f"patimakornsrinamen/rsna-67knee-weights-{a}" for a in ("rs", "t4f", "t5f")]
    assert m6["id"] == "patimakornsrinamen/rsna-67knee-o6"
    print("O6 == O5 except the coat line: " + d[1][1:].strip())
    with tempfile.TemporaryDirectory() as tmp:   # O7: O6 with rs swapped for rsf
        p = build_o1.build(tmp, "patimakornsrinamen/rsna-67knee-o7", ("rsf", "t4f", "t5f"))
        st7 = open(p, encoding="utf-8").read()
        m7 = json.load(open(os.path.join(tmp, "kernel-metadata.json")))
        spec = importlib.util.spec_from_file_location("o7_fusion", p)
        f7 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(f7)
    d = _diff(st6, st7)
    assert len(d) == 2 and all('"coat": ' in l for l in d), d
    f7._validate_blend(f7.BLEND)
    assert f7._required_our_arm_ids(f7.BLEND) == ["rsf", "t4f", "t5f"]
    assert f7.OUR_ARMS["rsf"] == dict(f7.OUR_ARMS["rs"], dataset="rsna-67knee-weights-rsf")
    assert m7["dataset_sources"][1:] == [f"patimakornsrinamen/rsna-67knee-weights-{a}" for a in ("rsf", "t4f", "t5f")]
    assert m7["id"] == "patimakornsrinamen/rsna-67knee-o7"
    print("O7 == O6 except the coat line: " + d[1][1:].strip())
    with tempfile.TemporaryDirectory() as tmp:   # O7b: O5 with rs swapped for rsf (O7 if O6 reads t4f as a loss)
        p = build_o1.build(tmp, "patimakornsrinamen/rsna-67knee-o7", ("rsf", "t4", "t5f"))
        st7b = open(p, encoding="utf-8").read()
        m7b = json.load(open(os.path.join(tmp, "kernel-metadata.json")))
        spec = importlib.util.spec_from_file_location("o7b_fusion", p)
        f7b = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(f7b)
    d = _diff(staged5, st7b)
    assert len(d) == 2 and all('"coat": ' in l for l in d), d
    f7b._validate_blend(f7b.BLEND)
    assert f7b._required_our_arm_ids(f7b.BLEND) == ["rsf", "t4", "t5f"]
    assert m7b["dataset_sources"][1:] == [f"patimakornsrinamen/rsna-67knee-weights-{a}" for a in ("rsf", "t4", "t5f")]
    print("O7b == O5 except the coat line: " + d[1][1:].strip())
    with tempfile.TemporaryDirectory() as tmp:
        st7 = open(build_o1.build(tmp, "patimakornsrinamen/rsna-67knee-o7", ("rsf", "t4f", "t5f")), encoding="utf-8").read()
    for name, extra in (("o8a", "rs"), ("o8b", "t5")):   # O8: O7 plus one fold-trained arm (width probe)
        with tempfile.TemporaryDirectory() as tmp:
            p = build_o1.build(tmp, f"patimakornsrinamen/rsna-67knee-{name}", ("rsf", "t4f", "t5f", extra))
            st = open(p, encoding="utf-8").read()
            m = json.load(open(os.path.join(tmp, "kernel-metadata.json")))
            spec = importlib.util.spec_from_file_location(f"{name}_fusion", p)
            f = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(f)
        d = _diff(st7, st)
        assert len(d) == 2 and all('"coat": ' in l for l in d), d
        f._validate_blend(f.BLEND)
        assert f._required_our_arm_ids(f.BLEND) == sorted(["rsf", "t4f", "t5f", extra])
        assert f.BLEND["coat"] == {"rsf": 1.0, "t4f": 1.0, "t5f": 1.0, extra: 1.0}
        assert m["dataset_sources"][1:] == [f"patimakornsrinamen/rsna-67knee-weights-{a}"
                                            for a in ("rsf", "t4f", "t5f", extra)]
        assert m["id"] == f"patimakornsrinamen/rsna-67knee-{name}"
        print(f"{name.upper()} == O7 except the coat line: " + d[1][1:].strip())
