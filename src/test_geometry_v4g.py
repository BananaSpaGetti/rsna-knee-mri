"""Synthetic tests for the v4g geometry switches in preprocess.py
(docs/specs/geometry-v4g.md, Task 10b). Runs with `py src/test_geometry_v4g.py`
(prints PASS per case) or `py -m pytest src/test_geometry_v4g.py -q`."""

import os
import sys
import tempfile

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess as P  # noqa: E402

IOP_SAG_NEG = [0, 1, 0, 0, 0, -1]   # r x c = -x (raw normal points to patient right)
IOP_SAG_POS = [0, 1, 0, 0, 0, 1]    # r x c = +x
IOP_COR = [1, 0, 0, 0, 0, -1]       # r x c = +y
IOP_AX = [1, 0, 0, 0, 1, 0]         # r x c = +z


def _pixels(k, frames=1):
    r, c = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    base = (c * 20 + r + 50 * k).astype(np.uint16)
    if frames == 1:
        return base
    return np.stack([base + 7 * f for f in range(frames)]).astype(np.uint16)


def _write(path, arr, *, ipp=None, iop=None, inst=None, photometric="MONOCHROME2",
           lat=None, slope=None, intercept=None, spacing=(1.0, 1.0),
           frame_ipps=None, shared_iop=None):
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=fm, preamble=b"\0" * 128)
    ds.SOPClassUID = fm.MediaStorageSOPClassUID
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = 16, 16
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = 0
    ds.PixelSpacing = list(spacing)
    if inst is not None:
        ds.InstanceNumber = inst
    if ipp is not None:
        ds.ImagePositionPatient = list(ipp)
    if iop is not None:
        ds.ImageOrientationPatient = list(iop)
    if lat is not None:
        ds.Laterality = lat
    if slope is not None:
        ds.RescaleSlope = str(slope)
    if intercept is not None:
        ds.RescaleIntercept = str(intercept)
    if frame_ipps is not None:
        ds.NumberOfFrames = len(frame_ipps)
        shared = Dataset()
        if shared_iop is not None:
            po = Dataset()
            po.ImageOrientationPatient = list(shared_iop)
            shared.PlaneOrientationSequence = Sequence([po])
        ds.SharedFunctionalGroupsSequence = Sequence([shared])
        pf = []
        for fi in frame_ipps:
            g = Dataset()
            pp = Dataset()
            pp.ImagePositionPatient = list(fi)
            g.PlanePositionSequence = Sequence([pp])
            pf.append(g)
        ds.PerFrameFunctionalGroupsSequence = Sequence(pf)
    ds.PixelData = np.ascontiguousarray(arr).tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


def _series(root, name, plane_iop, ipps, *, lat=None, shuffle=True, **kw):
    d = os.path.join(root, name)
    os.makedirs(d)
    files = []
    for k, ipp in enumerate(ipps):
        files.append(_write(os.path.join(d, f"s{k:03d}.dcm"), _pixels(k),
                            ipp=ipp, iop=plane_iop, inst=k + 1, lat=lat, **kw))
    if shuffle:
        files = files[::-1]
    return files


def _row(uid, plane, fluid=1, fat=0):
    return {"SeriesInstanceUID": uid, "Anatomical_Plane": plane,
            "Fluid_Sensitive": str(fluid), "Fat_Suppression": str(fat)}


def test_1_sagittal_negative_normal_reversed():
    with tempfile.TemporaryDirectory() as t:
        files = _series(t, "a", IOP_SAG_NEG, [(x, 0, 0) for x in (10, 20, 30, 40, 50)])
        v4 = P.sort_series_files(files, strict_geometry=True)
        refs, rc = P.order_slices_cardinal(P._read_slice_headers(files), "Sagittal")
        assert rc["flipped_normal"] is True, rc
        assert rc["method"] == "ipp_projection_cardinalized"
        assert [r[0] for r in refs] == v4[::-1]
        assert [r[1] for r in refs] == [None] * 5
        assert rc["ipp_x_median"] == 30.0


def test_2_sagittal_positive_normal_same():
    with tempfile.TemporaryDirectory() as t:
        files = _series(t, "a", IOP_SAG_POS, [(x, 0, 0) for x in (10, 20, 30, 40, 50)])
        v4 = P.sort_series_files(files, strict_geometry=True)
        refs, rc = P.order_slices_cardinal(P._read_slice_headers(files), "Sagittal")
        assert rc["flipped_normal"] is False, rc
        assert [r[0] for r in refs] == v4


def test_3_duplicate_positions_rejected():
    with tempfile.TemporaryDirectory() as t:
        files = _series(t, "a", IOP_AX, [(0, 0, 5), (0, 0, 5), (0, 0, 9)])
        refs, rc = P.order_slices_cardinal(P._read_slice_headers(files), "Axial")
        assert rc["method"] == "rejected_duplicate_position", rc
        assert refs == [] and rc["min_delta"] == 0.0


def test_4_no_ipp_falls_back_to_v4():
    with tempfile.TemporaryDirectory() as t:
        files = _series(t, "a", IOP_AX, [None] * 4)
        refs, rc = P.order_slices_cardinal(P._read_slice_headers(files), "Axial")
        assert rc["method"] == "v4_fallback_no_geometry", rc
        assert rc["n_positioned"] == 0 and rc["ipp_x_median"] is None
        assert refs == P.series_slice_refs(files, expand_frames=True,
                                           strict_geometry=True)


def test_5_decide_study_side():
    def rc(lat=(), x=None):
        return {"lat_values": list(lat), "ipp_x_median": x}
    assert P.decide_study_side({"a": ("Axial", rc(["L"]))})[:2] == ("L", "dicom_laterality_tag")
    side, src, tags, _ = P.decide_study_side({"a": ("Axial", rc(["L"])),
                                              "b": ("Coronal", rc(["R"]))})
    assert side is None and src == "conflicting_laterality_tags" and tags == ["L", "R"]
    side, src, _, med = P.decide_study_side({"a": ("Sagittal", rc(x=-12.0)),
                                             "b": ("Sagittal", rc(x=-9.0))})
    assert (side, src) == ("R", "sagittal_ipp_x") and med == -10.5
    assert P.decide_study_side({"a": ("Sagittal", rc(x=5.0))})[:2] == ("L", "sagittal_ipp_x")
    assert P.decide_study_side({"a": ("Axial", rc())})[:2] == (None, "unknown_no_flip")
    assert P.decide_study_side({})[:2] == (None, "unknown_no_flip")


def test_6_flip_for_plane():
    assert P.flip_for_plane("Coronal", "R") is True
    assert P.flip_for_plane("Sagittal", "R") is False
    assert P.flip_for_plane("Axial", "L") is False
    assert P.flip_for_plane("Axial", None) is False


def test_7_monochrome1_lut_exact():
    with tempfile.TemporaryDirectory() as t:
        arr = _pixels(3)
        p = _write(os.path.join(t, "m1.dcm"), arr, ipp=(0, 0, 0), iop=IOP_AX,
                   inst=1, photometric="MONOCHROME1", slope=2, intercept=-100)
        x = P.load_crop_float(p, crop_mm=150.0)
        want = arr.astype(np.float32) * 2.0 + -100.0
        want = (float(want.min()) + float(want.max())) - want
        assert x.dtype == np.float32 and x.shape == (16, 16)
        assert np.array_equal(x, want)
        got = P.window_series_crops([x], 16, False)[0]
        s = want[::4, ::4].ravel()
        lo, hi = np.percentile(s, [1, 99])
        y = np.clip((want - lo) / max(float(hi - lo), 1e-6), 0, 1)
        exp = (np.clip(y.astype(np.float32), 0, 1) * 255).astype(np.uint8)
        assert got.dtype == np.uint8 and np.array_equal(got, exp)
        assert np.array_equal(P.window_series_crops([x], 16, True)[0], exp[:, ::-1])
        # MONOCHROME2, slope 1 / intercept 0: v4 path vs crop_window path (branch is live)
        p2 = _write(os.path.join(t, "m2.dcm"), arr, ipp=(0, 0, 0), iop=IOP_AX, inst=1)
        a = P.load_slice(p2, size=32, crop_mm=150.0, isotropic=True, strict_geometry=True)
        b = P.window_series_crops([P.load_crop_float(p2, 150.0)], 32, False)[0]
        print("    v4 vs crop_window max abs diff:",
              int(np.abs(a.astype(int) - b.astype(int)).max()))


def _three_series_study(root, lat="R"):
    sag = _series(root, "sag", IOP_SAG_NEG, [(x, 0, 0) for x in (10, 20, 30, 40, 50)], lat=lat)
    cor = _series(root, "cor", IOP_COR, [(0, y, 0) for y in (5, 15, 25, 35, 45)], lat=lat)
    ax = _series(root, "ax", IOP_AX, [(0, 0, z) for z in (1, 11, 21, 31, 41)], lat=lat)
    del sag, cor, ax
    return [_row("sag", "Sagittal"), _row("cor", "Coronal"), _row("ax", "Axial")]


def test_8_build_study_tensor_side_canonicalisation():
    with tempfile.TemporaryDirectory() as t:
        rows = _three_series_study(t, lat="R")
        kw = {**P.recipe_kwargs(P.RECIPE_V4G), "size": 32}
        rc = {}
        a, fa = P.build_study_tensor(t, rows, receipts=rc, **kw)
        b, fb = P.build_study_tensor(t, rows, **{**kw, "canonical_side": False})
        assert fa == fb == 3, (fa, fb)
        assert rc["side"] == "R" and rc["side_source"] == "dicom_laterality_tag"
        assert rc["series"]["sag"]["reversed"] is True and rc["series"]["sag"]["flip"] is False
        assert rc["series"]["cor"]["flip"] is True and rc["series"]["ax"]["flip"] is True
        assert rc["series"]["sag"]["flipped_normal"] is True
        # coronal slot 2 (fluid): mirrored columns of the unflipped build
        assert a[2].any() and np.array_equal(a[2], b[2][:, :, ::-1])
        assert np.array_equal(a[4], b[4][:, :, ::-1])
        # sagittal slot 0: same rows reversed over the filled range, nothing else
        assert a[0, :5].any() and np.array_equal(a[0, :5], b[0, :5][::-1])
        assert not a[0, 5:].any() and not b[0, 5:].any()
    # left knee: no flip, no reversal
    with tempfile.TemporaryDirectory() as t:
        rows = _three_series_study(t, lat="L")
        rc = {}
        a, _ = P.build_study_tensor(t, rows, receipts=rc,
                                    **{**P.recipe_kwargs(P.RECIPE_V4G), "size": 32})
        b, _ = P.build_study_tensor(t, rows, **{**P.recipe_kwargs(P.RECIPE_V4G),
                                                "size": 32, "canonical_side": False})
        assert rc["side"] == "L" and np.array_equal(a, b)


def test_8b_rejected_series_and_rig_like_study():
    with tempfile.TemporaryDirectory() as t:
        _series(t, "dup", IOP_AX, [(0, 0, 0)] * 3)
        rc = {}
        out, filled = P.build_study_tensor(
            t, [_row("dup", "Axial")], receipts=rc,
            **{**P.recipe_kwargs(P.RECIPE_V4G), "size": 32})
        assert filled == 0 and not out.any()
        assert rc["series"]["dup"]["method"] == "rejected_duplicate_position"


def test_9_multiframe_cardinal_order():
    with tempfile.TemporaryDirectory() as t:
        p = _write(os.path.join(t, "mf.dcm"), _pixels(0, frames=4), inst=1,
                   frame_ipps=[(0, 0, 40), (0, 0, 30), (0, 0, 20), (0, 0, 10)],
                   shared_iop=IOP_AX)
        recs = P._read_slice_headers([p])
        assert recs[0]["n_frames"] == 4 and recs[0]["iop"] is not None
        refs, rc = P.order_slices_cardinal(recs, "Axial")
        assert rc["n_multiframe_files"] == 1 and rc["n_entries"] == 4
        assert rc["method"] == "ipp_projection_cardinalized"
        assert refs == [(p, 3), (p, 2), (p, 1), (p, 0)], refs


def _main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print("PASS", n)
    print(f"{len(tests)} tests PASS")


if __name__ == "__main__":
    _main()
