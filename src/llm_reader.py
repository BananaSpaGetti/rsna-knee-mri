"""Read every radiology report with an LLM and write a per-finding verdict table -
one more "reader" for the label consensus (round 10 track 1).

Usage:
  py src/llm_reader.py --provider gateway --env <path/to/.env> --name qwen38_gw [--workers 4] [--limit N]
  py src/llm_reader.py --provider ollama --model gemma4:31b --name gemma4_31b [--workers 1]
  py src/llm_reader.py --provider anthropic --model claude-opus-5 --name opus5 [--workers 4]

Providers:
  gateway   OpenAI-compatible /chat/completions. The .env holds PASSPORT_BASE_URL,
            PASSPORT_API_KEY, PASSPORT_MODEL (the AIproject convention); thinking is
            switched off through chat_template_kwargs (vLLM / Qwen convention).
  ollama    local http://127.0.0.1:11434/api/chat with think=false, format=json.
  anthropic official SDK (needs ANTHROPIC_API_KEY or ~/.config/rsna_llm_key), structured
            output through output_config.format so the JSON always parses.

Every raw answer is appended to data/labels-ours/<name>.jsonl (uid, verdicts, meta), so a
run can be resumed; the CSV data/labels-ours/<name>.csv is rebuilt from the jsonl at the
end: per finding `<F>` (YES 1, NO 0, UNK 0.5 - the stevenleehans "full" convention that
pseudo_labels.py already understands), `<F>__verdict`, `<F>__severity`. The prompt follows
pilkwang's api_labeler.py contract (verdict YES/NO/UNK + severity 0-3, "judge only what
the report states") with our own finding definitions, because that reader is the
strongest of the three public ones on gold-58.
"""

import argparse
import concurrent.futures as cf
import csv
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import LABELS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(ROOT, "data", "train.csv")
OUT_DIR = os.path.join(ROOT, "data", "labels-ours")

FINDING_SPEC = """Findings (use these exact keys):
- "ACL": anterior cruciate ligament tear - partial or complete, acute or chronic, or a torn/failed ACL graft. Mucoid degeneration or a sprain WITHOUT tear is NO.
- "MCL": medial collateral ligament injury - sprain (grade 1-3), partial or complete tear, or oedema of the MCL. Normal/intact MCL is NO.
- "Medial Meniscus": tear of the medial meniscus (any type: horizontal, radial, root, bucket-handle, complex, flap, degenerative tear, or a prior partial meniscectomy with re-tear). Intrameniscal degeneration/signal that does NOT reach an articular surface is NO.
- "Lateral Meniscus": tear of the lateral meniscus, same criteria as the medial one.
- "Medial OA": osteoarthritis / chondral loss of the MEDIAL tibiofemoral compartment - cartilage thinning, fissuring, full-thickness defects, subchondral sclerosis/cysts/oedema from wear, osteophytes, or the words arthrosis/gonarthrosis/degenerative change located medially.
- "Lateral OA": the same for the LATERAL tibiofemoral compartment.
- "PF OA": the same for the patellofemoral compartment (patella and trochlea) - chondromalacia counts.
- "Effusion": joint effusion / intra-articular fluid of any size (trace, small, mild all count as YES). Fluid described as physiologic/normal amount is NO. Synonyms: derrame articular, liquido articular / aumento de liquido, hidrartrose, epanchement, Gelenkerguss / Ergussbildung, versamento, gewrichtsvocht / hydrops, eklem efuzyonu / sivi, ugroi / υδράρθρο, излив, izljev.
- "Synovitis": synovitis, synovial thickening / proliferation / hypertrophy / enhancement; synonyms sinovitis, sinovite, synovite, Synovitis / Synovialitis, sinovit, υμενίτιδα, синовит; plica alone is NO.
- "Baker's": Baker's (popliteal) cyst, including a ruptured one; synonyms quiste de Baker / quiste popliteo, kyste poplite, Baker-Zyste, cisto de Baker, Baker kisti, κύστη Baker, киста Бейкера.
- "Contusion": bone marrow oedema / contusion / bone bruise / trabecular microfracture in any bone of the knee; synonyms edema oseo / edema de medula osea / contusion osea, oedeme osseux / contusion osseuse, Knochenmarkodem / bone bruise, edema osseo, kemik iligi odemi, οίδημα μυελού, костномозговой отек, edem koštane srži. Subchondral oedema that the report attributes ONLY to osteoarthritis wear is still YES here (it is bone marrow oedema).
- "Fracture": any fracture - cortical, avulsion (Segond, tibial spine), osteochondral, subchondral insufficiency or stress fracture; synonyms fractura, fratura, fracture, Fraktur, frattura, kirik, κάταγμα, перелом, prijelom. A bone bruise WITHOUT a fracture line is NO.
"""

SYSTEM = f"""You are an expert musculoskeletal radiologist reading a knee MRI report.
Reports may be written in any language (Spanish, Portuguese, Greek, Bulgarian, Turkish, Croatian, Dutch, German, French, English, ...). Read each one in its original language.

For each of the twelve findings, state what THE REPORT IN FRONT OF YOU says.

{FINDING_SPEC}
verdict:
  YES  the report asserts the finding is present
  NO   the report denies it, or describes that structure or compartment as normal, intact or preserved
  UNK  the report does not mention the structure at all, or is explicitly uncertain / equivocal

severity (0 whenever the verdict is not YES):
  1  minimal, trace, small, mild, slight, grade 1, low-grade, chondromalacia grade 1-2
  2  stated without qualification, moderate, grade 2
  3  marked, large, severe, extensive, complete, full-thickness, grade 3-4, displaced, bucket-handle

Rules:
- Judge only what the report states. Do not infer a finding from a related one (an effusion does not imply synovitis; medial OA does not imply lateral OA).
- A structure described as normal or intact is NO, not UNK. "Menisci normal" answers BOTH menisci; "ligaments intact" answers ACL and MCL.
- A finding the report never names is UNK, not NO. Do not treat silence as absence.
- Medial and lateral are distinct compartments. Patellofemoral is a third compartment.
- Answer ONLY with a JSON object whose keys are exactly the twelve finding names above, each mapping to {{"verdict": "YES"|"NO"|"UNK", "severity": 0|1|2|3}}. No prose."""

SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "object",
                       "properties": {"verdict": {"type": "string", "enum": ["YES", "NO", "UNK"]},
                                      "severity": {"type": "integer", "enum": [0, 1, 2, 3]}},
                       "required": ["verdict", "severity"], "additionalProperties": False}
                   for k in LABELS},
    "required": list(LABELS),
    "additionalProperties": False,
}

# Compact answer format for paid APIs: one short string per finding ("NO", "UNK",
# "YES1".."YES3" = YES + severity) - about a third of the output tokens of the
# object form, which is where most of the cost sits.
COMPACT_ENUM = ["NO", "UNK", "YES1", "YES2", "YES3"]
COMPACT_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string", "enum": COMPACT_ENUM} for k in LABELS},
    "required": list(LABELS),
    "additionalProperties": False,
}
COMPACT_RULE = ('\n\nOUTPUT FORMAT (compact): a JSON object whose keys are exactly the twelve '
                'finding names, each mapping to ONE string: "NO", "UNK", "YES1" (present, '
                'severity 1), "YES2" (present, severity 2) or "YES3" (present, severity 3).')

SCORE = {"YES": 1.0, "NO": 0.0, "UNK": 0.5}
# Graded score for the `<F>` column: a mild finding in the report is less often a
# radiologist positive than a marked one (pilkwang's score column does the same, and
# it is why its Effusion AUC on gold-58 is 0.83 where flat YES=1 scores 0.65).
GRADED = {("NO", 0): 0.0, ("UNK", 0): 0.5, ("YES", 1): 0.7, ("YES", 2): 0.85, ("YES", 3): 1.0}
_lock = threading.Lock()


def load_env(path):
    env = {}
    for line in open(path, encoding="utf-8"):
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.rstrip("\n").split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env


def _post_json(url, body, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_answer(text):
    """Strict parse: twelve keys, each verdict in {YES,NO,UNK}, severity 0-3."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    obj = json.loads(text)
    out = {}
    for k in LABELS:
        v = obj[k]
        if isinstance(v, str):  # compact form: "NO" | "UNK" | "YES1".."YES3"
            v = v.strip().upper()
            if v.startswith("YES"):
                v = {"verdict": "YES", "severity": int(v[3:] or 2)}
            else:
                v = {"verdict": v, "severity": 0}
        verdict = str(v["verdict"]).upper()
        if verdict not in SCORE:
            raise ValueError(f"bad verdict {verdict!r} for {k}")
        sev = int(v.get("severity", 0))
        if verdict != "YES":
            sev = 0
        out[k] = {"verdict": verdict, "severity": max(0, min(3, sev))}
    return out


class Gateway:
    def __init__(self, env, model=None, workers=4, think=False):
        self.base = env["PASSPORT_BASE_URL"].rstrip("/")
        self.model = model or env["PASSPORT_MODEL"]
        self.headers = {"Authorization": "Bearer " + env["PASSPORT_API_KEY"],
                        "Content-Type": "application/json", "User-Agent": "rsna-67knee-reader/0.1"}
        self.workers = workers
        self.think = think

    def ask(self, report):
        body = {"model": self.model, "temperature": 0,
                "max_tokens": 6000 if self.think else 900,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": bool(self.think)},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": "REPORT:\n" + report}]}
        d = _post_json(self.base + "/chat/completions", body, self.headers, 180)
        msg = d["choices"][0]["message"]
        return msg.get("content") or "", {"usage": d.get("usage"), "model": d.get("model")}


class Ollama:
    def __init__(self, model, workers=1, url="http://127.0.0.1:11434"):
        self.model, self.url, self.workers = model, url, workers

    def ask(self, report):
        body = {"model": self.model, "stream": False, "think": False, "format": "json",
                "keep_alive": "30m",
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": "REPORT:\n" + report}],
                "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 900}}
        d = _post_json(self.url + "/api/chat", body, {"Content-Type": "application/json"}, 600)
        return d["message"]["content"], {"eval_count": d.get("eval_count"),
                                         "eval_duration": d.get("eval_duration")}


class AnthropicReader:
    """Compact output, cached system prompt (Claude Opus 5 caches prefixes >= 512
    tokens; this one is ~700), effort low. `params()` is shared by the synchronous
    path and the Message Batches path (50% price)."""

    def __init__(self, model="claude-opus-5", workers=4):
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            p = os.path.join(os.path.expanduser("~"), ".config", "rsna_llm_key")
            if os.path.isfile(p):
                key = open(p).read().strip()
        # An organisation-level key must name the workspace on every request.
        ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        if not ws:
            p = os.path.join(os.path.expanduser("~"), ".config", "rsna_llm_workspace")
            if os.path.isfile(p):
                ws = open(p).read().strip()
        headers = {"anthropic-workspace-id": ws} if ws else None
        kw = {"default_headers": headers} if headers else {}
        self.client = anthropic.Anthropic(api_key=key, **kw) if key else anthropic.Anthropic(**kw)
        self.model, self.workers = model, workers

    def params(self, report):
        return dict(
            model=self.model, max_tokens=600,
            system=[{"type": "text", "text": SYSTEM + COMPACT_RULE,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": "REPORT:\n" + report}],
            output_config={"format": {"type": "json_schema", "schema": COMPACT_SCHEMA},
                           "effort": "low"})

    @staticmethod
    def unpack(resp):
        if resp.stop_reason == "refusal":
            raise RuntimeError("refusal")
        text = next(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        return text, {"usage": {"input": u.input_tokens, "output": u.output_tokens,
                                "cache_read": getattr(u, "cache_read_input_tokens", None),
                                "cache_write": getattr(u, "cache_creation_input_tokens", None)}}

    def ask(self, report):
        return self.unpack(self.client.messages.create(**self.params(report)))

    def run_batch(self, items, jsonl_path, state_path):
        """items: [(uid, report)]. Creates one Message Batch (or resumes the id saved
        in state_path), polls until it ends, appends every result to jsonl_path."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        if os.path.exists(state_path):
            batch_id = open(state_path).read().strip()
            print(f"resuming batch {batch_id}", flush=True)
        else:
            reqs = [Request(custom_id=uid, params=MessageCreateParamsNonStreaming(**self.params(rep)))
                    for uid, rep in items]
            batch = self.client.messages.batches.create(requests=reqs)
            batch_id = batch.id
            open(state_path, "w").write(batch_id)
            print(f"created batch {batch_id} with {len(reqs)} requests", flush=True)
        while True:
            b = self.client.messages.batches.retrieve(batch_id)
            c = b.request_counts
            print(f"  {b.processing_status}: processing {c.processing} succeeded {c.succeeded} "
                  f"errored {c.errored} expired {c.expired}", flush=True)
            if b.processing_status == "ended":
                break
            time.sleep(60)
        n_ok = n_err = 0
        with open(jsonl_path, "a", encoding="utf-8") as out:
            for res in self.client.messages.batches.results(batch_id):
                uid = res.custom_id
                if res.result.type == "succeeded":
                    try:
                        text, meta = self.unpack(res.result.message)
                        rec = {"uid": uid, "verdicts": parse_answer(text), "meta": meta}
                        n_ok += 1
                    except Exception as e:
                        rec = {"uid": uid, "error": f"parse: {e!r}"[:200]}
                        n_err += 1
                else:
                    rec = {"uid": uid, "error": f"batch {res.result.type}"}
                    n_err += 1
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"batch done: ok {n_ok} err {n_err}", flush=True)


def make_provider(args):
    if args.provider == "gateway":
        return Gateway(load_env(args.env), args.model, args.workers, think=args.think)
    if args.provider == "ollama":
        return Ollama(args.model or "gemma4:31b", args.workers)
    if args.provider == "anthropic":
        return AnthropicReader(args.model or "claude-opus-5", args.workers)
    raise SystemExit(f"unknown provider {args.provider}")


def read_one(provider, uid, report, retries=4):
    last = None
    for attempt in range(retries):
        try:
            text, meta = provider.ask(report)
            verdicts = parse_answer(text)
            return {"uid": uid, "verdicts": verdicts, "meta": meta, "attempt": attempt}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            time.sleep(15 * (attempt + 1) if e.code in (429, 500, 502, 503, 504) else 2)
        except (json.JSONDecodeError, KeyError, ValueError, RuntimeError) as e:
            last = f"parse: {e!r}"[:200]
            time.sleep(1)
        except Exception as e:  # network / timeout
            last = repr(e)[:200]
            time.sleep(10 * (attempt + 1))
    return {"uid": uid, "error": last}


def write_csv(jsonl_path, csv_path):
    rows = {}
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if "verdicts" in rec:
                rows[rec["uid"]] = rec["verdicts"]
    cols = ["StudyInstanceUID"]
    for k in LABELS:
        cols += [k, f"{k}__verdict", f"{k}__severity"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for uid in sorted(rows):
            r = [uid]
            for k in LABELS:
                v = rows[uid][k]
                sev = v["severity"] if v["verdict"] == "YES" else 0
                r += [GRADED.get((v["verdict"], sev), SCORE[v["verdict"]]), v["verdict"], sev]
            w.writerow(r)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["gateway", "ollama", "anthropic"])
    ap.add_argument("--name", required=True, help="reader name -> data/labels-ours/<name>.{jsonl,csv}")
    ap.add_argument("--env", default=None, help="gateway: .env with PASSPORT_* keys")
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N studies (after gold-58)")
    ap.add_argument("--gold-first", action="store_true", help="put the 58 gold studies first")
    ap.add_argument("--think", action="store_true", help="gateway: leave the model's thinking on (slower)")
    ap.add_argument("--batch", action="store_true", help="anthropic: one Message Batch (50%% price, <= 24 h)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    jsonl_path = os.path.join(OUT_DIR, f"{args.name}.jsonl")
    csv_path = os.path.join(OUT_DIR, f"{args.name}.csv")

    import pandas as pd
    df = pd.read_csv(TRAIN_CSV)
    uids = df["StudyInstanceUID"].tolist()
    reports = dict(zip(uids, df["Report"].astype(str)))
    if args.gold_first:
        gold = set(pd.read_csv(os.path.join(ROOT, "data", "gold58.csv"))["StudyInstanceUID"])
        uids = [u for u in uids if u in gold] + [u for u in uids if u not in gold]
    if args.limit:
        uids = uids[:args.limit]

    done = set()
    if os.path.exists(jsonl_path):
        for line in open(jsonl_path, encoding="utf-8"):
            rec = json.loads(line)
            if "verdicts" in rec:
                done.add(rec["uid"])
    todo = [u for u in uids if u not in done]
    print(f"{args.name}: {len(done)} done, {len(todo)} to read with {args.provider} x{args.workers}", flush=True)

    provider = make_provider(args)
    if args.batch:
        if args.provider != "anthropic":
            raise SystemExit("--batch is only for the anthropic provider")
        if todo:
            provider.run_batch([(u, reports[u]) for u in todo], jsonl_path,
                               os.path.join(OUT_DIR, f"{args.name}.batch_id"))
        n = write_csv(jsonl_path, csv_path)
        print(f"wrote {csv_path} ({n} studies)", flush=True)
        return
    t0 = time.time()
    n_ok = n_err = 0
    with open(jsonl_path, "a", encoding="utf-8") as out, cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(read_one, provider, u, reports[u]): u for u in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            rec = fut.result()
            with _lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
            if "error" in rec:
                n_err += 1
                print(f"  ERR {rec['uid'][-12:]}: {rec['error']}", flush=True)
            else:
                n_ok += 1
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} ok {n_ok} err {n_err} {el/60:.1f} min "
                      f"({i/el:.2f} studies/s, eta {(len(todo)-i)/max(i/el,1e-6)/60:.0f} min)", flush=True)
    n = write_csv(jsonl_path, csv_path)
    print(f"wrote {csv_path} ({n} studies)", flush=True)


if __name__ == "__main__":
    main()
