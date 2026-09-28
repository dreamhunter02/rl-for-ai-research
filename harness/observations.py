"""JSON-safe observation budgets. Offsets always describe delivered text."""
import copy
import json
import re

OBS_CAP = 4000


def compact_hits(hits, queries=(), limit=5):
    terms = [w for q in queries for w in re.findall(r"\w+", q.lower()) if len(w) > 2]
    out = []
    for hit in hits[:limit]:
        text = hit.get("text", "")
        positions = [text.lower().find(t) for t in terms if t in text.lower()]
        start = max(0, min(positions, default=0) - 100)
        record = {k: hit[k] for k in ("document_id", "page", "passage_id", "table_id", "score", "units", "title") if k in hit}
        record.update(snippet=text[start:start + 300], snippet_start=start + hit.get("start", 0))
        out.append(record)
    return out


def bounded_observation(payload, cap=OBS_CAP):
    """Trim structured fields, never serialized JSON; report exact read continuation."""
    obj = copy.deepcopy(payload if isinstance(payload, dict) else {"content": str(payload)})
    dumps = lambda x: json.dumps(x, ensure_ascii=False, separators=(",", ":"))
    raw_chars = len(dumps(obj))
    obj["observation"] = {"raw_chars": raw_chars, "truncated": False}
    # Leave complete individual hits/receipts intact: remove tails until the list fits.
    while len(dumps(obj)) > cap:
        lists = [v for k, v in obj.items() if isinstance(v, list) and v]
        if lists:
            max(lists, key=lambda v: len(dumps(v))).pop()
            obj["observation"]["truncated"] = True
            continue
        fields = [(len(v), k) for k, v in obj.items() if isinstance(v, str) and k in {"text", "content", "neighbor_context", "error"} and v]
        if not fields:
            # Oversized metadata is rejected, rather than falsely claiming evidence visibility.
            return dumps({"error": "Observation metadata exceeds budget; narrow the request.", "observation": {"raw_chars": raw_chars, "truncated": True}})
        _, key = max(fields)
        excess = len(dumps(obj)) - cap
        obj[key] = obj[key][:max(0, len(obj[key]) - max(1, excess // 2))]
        if key == "text" and "start" in obj:
            obj["end"] = obj["start"] + len(obj["text"])
            obj["next_start"] = obj["end"]
            obj["has_more"] = obj["end"] < obj.get("total_chars", obj["end"] + 1)
            obj.pop("provenance", None)  # old offsets no longer describe delivered text
        obj["observation"]["truncated"] = True
    return dumps(obj)
