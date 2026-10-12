"""
51_run_saia_model.py — paced, header-aware, resumable SAIA runner
=================================================================
SAIA limits (per key, shared across ALL models):
    30 / minute,  200 / hour,  1000 / day,  3000 / month
The binding sustained rate is 200/hour, so we run SINGLE-THREADED and pace at a
fixed interval (default ~19s => ~190/hour) — concurrency is pointless here and
only risks tripping caps. Before each call we inspect the live rate-limit
headers and:
  - pause until reset if the hourly/daily budget is nearly exhausted,
  - STOP entirely if the monthly budget hits a safety floor,
so we never waste a request or hard-fail.

Reuses the exact messages from batch_input.jsonl, restricted to the subset ids,
swapping only the model. Fully resumable (skips completed custom_ids).

Usage:
    python3 51_run_saia_model.py <model_id> [--ids saia_bestcond_ids.json]
        [--interval 19] [--max-tokens 5] [--month-floor 60] [--limit N]

Output: saia_out_<safe_model>.jsonl  (custom_id, raw, label)
"""

import os, sys, json, time, argparse
from openai import OpenAI

API_KEY  = os.environ["SAIA_API_KEY"]
BASE_URL = "https://chat-ai.academiccloud.de/v1"

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--ids", default="saia_bestcond_ids.json")
ap.add_argument("--interval", type=float, default=19.0,
                help="seconds between requests (19 => ~190/hour, under the 200 cap)")
ap.add_argument("--max-tokens", type=int, default=5)
ap.add_argument("--reasoning", action="store_true")
ap.add_argument("--month-floor", type=int, default=60,
                help="stop if remaining-month drops to/below this (safety buffer)")
ap.add_argument("--hour-floor", type=int, default=10)
ap.add_argument("--day-floor", type=int, default=10)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--max-retries", type=int, default=6)
args = ap.parse_args()

safe = args.model.replace("/", "_").replace(".", "_")
out_path = f"saia_out_{safe}.jsonl"

want_ids = set(json.load(open(args.ids)))
messages_by_id = {}
for line in open("batch_input.jsonl"):
    item = json.loads(line)
    if item["custom_id"] in want_ids:
        messages_by_id[item["custom_id"]] = item["body"]["messages"]

# A custom_id counts as done ONLY if it has a non-null label; rows that errored
# (e.g. local connection drops, which don't cost quota) are retried on resume.
done = set()
if os.path.exists(out_path):
    for line in open(out_path):
        try:
            r = json.loads(line)
            if r.get("label") is not None:
                done.add(r["custom_id"])
        except Exception:
            pass
pending = [c for c in sorted(messages_by_id) if c not in done]
if args.limit:
    pending = pending[: args.limit]

print(f"model={args.model}\n  subset={len(messages_by_id)}  done={len(done)}  "
      f"pending={len(pending)}  interval={args.interval}s (~{3600/args.interval:.0f}/hr)")
if not pending:
    print("Nothing to do."); sys.exit(0)

# max_retries=0 is CRITICAL: the SDK's default silent retries hit the gateway
# again on every 429 and this academic gateway counts EACH attempt against the
# quota — a hidden 2x-3x spend. We do all ret/pacing ourselves below.
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=0)
fout = open(out_path, "a")


def parse_label(text):
    if not text:
        return None
    last = None
    for ch in str(text):
        if ch in ("0", "1"):
            last = int(ch)
    return last


def hdr_int(headers, key, default=None):
    try:
        return int(headers.get(key))
    except (TypeError, ValueError):
        return default


t0 = time.time()
n_done = n_fail = 0
for i, cid in enumerate(pending):
    msgs = messages_by_id[cid]
    raw, label, headers = None, None, {}
    for attempt in range(args.max_retries):
        try:
            resp = client.chat.completions.with_raw_response.create(
                model=args.model, messages=msgs,
                temperature=0, max_tokens=args.max_tokens)
            headers = resp.headers
            comp = resp.parse()
            m = comp.choices[0].message
            raw = m.content or ""
            if args.reasoning and not raw.strip():
                raw = getattr(m, "reasoning_content", "") or ""
            label = parse_label(raw)
            break
        except Exception as e:
            msg = str(e)
            # rate-limited: read the reset window from the response headers and
            # wait exactly that long (don't hammer — each attempt costs quota).
            if "429" in msg or "rate" in msg.lower():
                reset = None
                resp_obj = getattr(e, "response", None)
                if resp_obj is not None:
                    h = getattr(resp_obj, "headers", {}) or {}
                    reset = hdr_int(h, "ratelimit-reset") or hdr_int(h, "retry-after")
                time.sleep((reset + 2) if reset else min(60 * (attempt + 1), 300))
                continue
            if attempt == args.max_retries - 1:
                raw = f"__ERR__ {type(e).__name__}: {msg[:120]}"
            else:
                time.sleep(min(2 ** attempt, 30))
    fout.write(json.dumps({"custom_id": cid, "raw": raw, "label": label}) + "\n")
    fout.flush()
    n_done += 1
    if label is None:
        n_fail += 1

    rem_min   = hdr_int(headers, "x-ratelimit-remaining-minute")
    rem_hour  = hdr_int(headers, "x-ratelimit-remaining-hour")
    rem_day   = hdr_int(headers, "x-ratelimit-remaining-day")
    rem_month = hdr_int(headers, "x-ratelimit-remaining-month")

    if n_done % 20 == 0 or n_done == len(pending):
        rate = n_done / (time.time() - t0) * 3600
        print(f"  {n_done}/{len(pending)}  fails={n_fail}  ~{rate:.0f}/hr  "
              f"remaining[min/hr/day/mon]={rem_min}/{rem_hour}/{rem_day}/{rem_month}",
              flush=True)

    # hard safety stop on monthly budget
    if rem_month is not None and rem_month <= args.month_floor:
        print(f"\nSTOP: monthly budget floor reached (remaining={rem_month}). "
              f"Resume later — progress saved to {out_path}.")
        break

    # pause when hourly/daily nearly exhausted
    if rem_hour is not None and rem_hour <= args.hour_floor:
        print(f"  hourly budget low (rem={rem_hour}); sleeping 15 min...", flush=True)
        time.sleep(15 * 60)
    elif rem_day is not None and rem_day <= args.day_floor:
        print(f"  daily budget low (rem={rem_day}); sleeping 60 min...", flush=True)
        time.sleep(60 * 60)
    elif i < len(pending) - 1:
        time.sleep(args.interval)

fout.close()
print(f"\nDONE {args.model}: {n_done} responses this run ({n_fail} failures) -> {out_path}")
