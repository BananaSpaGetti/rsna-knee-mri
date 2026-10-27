# Transfer-syntax census of the competition DICOMs, and a decode test per syntax (CPU only, no model).
#
# Question: do the public inference stacks (nartaa's Raptor, heliosli's R7/S6, goodpjw's reader) silently drop
# compressed slices? Raptor's fastread decodes uncompressed pixels itself and falls back to pydicom for anything
# else, turning any exception into "no slice". The submission image (same docker_image as H7/F3bR) ships Python 3.13
# and, as far as we know, no pylibjpeg/gdcm, and the reader dataset's decoder wheels are cp311/cp312 only.
#
# Output /kaggle/working/tscensus.json:
#   per split: series count by TransferSyntaxUID (first file of each series), studies with >= 1 compressed series,
#   within-series consistency on a sample, and per compressed syntax whether pixel_array decodes with the image's
#   own pydicom, then with pylibjpeg (+ libjpeg, openjpeg) and with python-gdcm (cp313 wheels from PyPI, saved to
#   /kaggle/working/wheels so they can become a dataset), with pixel equality between the two decoders.
import collections
import glob
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

T0 = time.time()
ROOT = next(p for p in ['/kaggle/input/competitions/rsna-knee-abnormality-detection',
                        '/kaggle/input/rsna-knee-abnormality-detection'] if os.path.isdir(p))
OUT = '/kaggle/working'
UNCOMPRESSED = {'1.2.840.10008.1.2', '1.2.840.10008.1.2.1', '1.2.840.10008.1.2.2'}
SAMPLES_PER_TS = 6
import pydicom

res = dict(python=sys.version, pydicom=pydicom.__version__, root=ROOT, top=sorted(os.listdir(ROOT)))
print(json.dumps(res), flush=True)


def first_hdr(sdir):
    fs = sorted(glob.glob(sdir + '/*.dcm'))
    if not fs:
        return sdir, 'EMPTY', 0, None
    try:
        h = pydicom.dcmread(fs[0], stop_before_pixels=True)
        return sdir, str(h.file_meta.TransferSyntaxUID), len(fs), fs[0]
    except Exception as e:
        return sdir, 'ERR ' + type(e).__name__, len(fs), fs[0]


def all_ts(sdir):
    out = collections.Counter()
    for f in sorted(glob.glob(sdir + '/*.dcm')):
        try:
            out[str(pydicom.dcmread(f, stop_before_pixels=True).file_meta.TransferSyntaxUID)] += 1
        except Exception as e:
            out['ERR ' + type(e).__name__] += 1
    return dict(out)


samples = collections.defaultdict(list)
for split in ('test_series', 'train_series'):
    base = os.path.join(ROOT, split)
    if not os.path.isdir(base):
        res[split] = 'missing'
        continue
    t = time.time()
    sdirs = [os.path.join(base, st, se) for st in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, st))
             for se in sorted(os.listdir(os.path.join(base, st))) if os.path.isdir(os.path.join(base, st, se))]
    with ThreadPoolExecutor(48) as ex:
        recs = list(ex.map(first_hdr, sdirs))
    by_ts = collections.Counter(r[1] for r in recs)
    files_by_ts = collections.Counter()
    study_comp = collections.defaultdict(lambda: [0, 0])
    for sdir, ts, n, f in recs:
        files_by_ts[ts] += n
        st = os.path.basename(os.path.dirname(sdir))
        study_comp[st][0] += 1
        if ts not in UNCOMPRESSED:
            study_comp[st][1] += 1
            if f is not None and len(samples[ts]) < SAMPLES_PER_TS:
                samples[ts].append(f)
    comp_series = [r[0] for r in recs if r[1] not in UNCOMPRESSED]
    rnd = random.Random(0)
    check = rnd.sample(sdirs, min(60, len(sdirs))) + rnd.sample(comp_series, min(40, len(comp_series)))
    with ThreadPoolExecutor(24) as ex:
        mixed = [(s, c) for s, c in zip(check, ex.map(all_ts, check)) if len(c) > 1]
    res[split] = dict(
        series=len(recs), studies=len(study_comp), series_by_ts=dict(by_ts), files_by_ts=dict(files_by_ts),
        studies_any_compressed=sum(1 for v in study_comp.values() if v[1] > 0),
        studies_all_compressed=sum(1 for v in study_comp.values() if v[1] == v[0]),
        consistency_checked=len(check), mixed_series=[[os.path.relpath(s, ROOT), c] for s, c in mixed][:20],
        seconds=round(time.time() - t, 1))
    print(split, json.dumps(res[split]), flush=True)

DECODE_PROBE = r'''
import json, sys, hashlib
import numpy as np
import pydicom
try:
    from pydicom.pixels import apply_modality_lut
except ImportError:
    from pydicom.pixel_data_handlers.util import apply_modality_lut
out = {}
for f in json.load(open(sys.argv[1])):
    try:
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        out[f] = dict(ok=True, shape=list(a.shape), md5=hashlib.md5(a.tobytes()).hexdigest(), mean=float(a.mean()))
    except Exception as e:
        out[f] = dict(ok=False, err=(type(e).__name__ + ': ' + str(e))[:300])
json.dump(out, open(sys.argv[2], 'w'))
'''
open('/tmp/decode_probe.py', 'w').write(DECODE_PROBE)
files = [f for fs in samples.values() for f in fs]
json.dump(files, open('/tmp/probe_files.json', 'w'))


def probe(tag, pythonpath=None):
    env = dict(os.environ)
    if pythonpath:
        env['PYTHONPATH'] = pythonpath + os.pathsep + env.get('PYTHONPATH', '')
    out = '/tmp/probe_%s.json' % tag
    p = subprocess.run([sys.executable, '/tmp/decode_probe.py', '/tmp/probe_files.json', out], env=env,
                       capture_output=True, text=True)
    if p.returncode != 0:
        return dict(error=p.stderr[-1500:])
    return json.load(open(out))


decode = dict(image=probe('image'))
os.makedirs(OUT + '/wheels', exist_ok=True)
pkgs = dict(pylibjpeg=['pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg'], gdcm=['python-gdcm'])
for tag, names in pkgs.items():
    dl = subprocess.run([sys.executable, '-m', 'pip', 'download', '-q', '--only-binary=:all:', '--no-deps',
                         '--dest', OUT + '/wheels'] + names, capture_output=True, text=True)
    tgt = '/tmp/dec_' + tag
    ins = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--target', tgt,
                          '--find-links', OUT + '/wheels'] + names, capture_output=True, text=True)
    decode[tag + '_install'] = dict(download_rc=dl.returncode, download_err=dl.stderr[-800:],
                                    install_rc=ins.returncode, install_err=ins.stderr[-800:])
    decode[tag] = probe(tag, tgt) if ins.returncode == 0 else 'not installed'
res['wheels'] = sorted(os.listdir(OUT + '/wheels'))

summary = {}
for ts, fs in samples.items():
    row = {}
    for tag in ('image', 'pylibjpeg', 'gdcm'):
        d = decode.get(tag)
        if isinstance(d, dict) and 'error' not in d:
            row[tag] = sum(1 for f in fs if d.get(f, {}).get('ok'))
        else:
            row[tag] = None
    pj, gd = decode.get('pylibjpeg'), decode.get('gdcm')
    if isinstance(pj, dict) and isinstance(gd, dict) and 'error' not in pj and 'error' not in gd:
        row['pylibjpeg_eq_gdcm'] = sum(1 for f in fs if pj.get(f, {}).get('ok') and gd.get(f, {}).get('ok')
                                       and pj[f]['md5'] == gd[f]['md5'])
    row['n'] = len(fs)
    summary[ts] = row
res['decode_summary'] = summary
res['decode'] = decode
res['seconds'] = round(time.time() - T0, 1)
json.dump(res, open(OUT + '/tscensus.json', 'w'), indent=1)
print('DECODE_SUMMARY ' + json.dumps(summary), flush=True)
print('TSCENSUS_COMPLETE seconds %.0f' % res['seconds'], flush=True)
