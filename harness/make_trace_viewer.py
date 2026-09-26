"""Convert a teacher-trace JSONL into an HTML viewer for quick inspection.

Usage:
    python3 make_trace_viewer.py --in results/teacher_traces/train_gpt56.jsonl \\
        --out results/teacher_traces/train_gpt56_viewer.html \\
        --label GPT 5.6
"""
import argparse, json, html
from pathlib import Path

def esc(s):
    return html.escape(str(s)) if s is not None else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="teacher")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    body = []
    body.append(f"<h1>{esc(args.label)} traces ({len(rows)} rows)</h1>")
    body.append("<style>body{font-family:monospace;font-size:13px;max-width:1100px;margin:20px auto;}"
                " .row{border:1px solid #ccc;border-radius:8px;padding:10px;margin:14px 0;background:#fafafa;}"
                " .ok{background:#e8f5e9;border-color:#4caf50;}"
                " .mid{background:#fff8e1;border-color:#ffc107;}"
                " .bad{background:#fdecea;border-color:#e57373;}"
                " .tool{background:#263238;color:#eee;border-radius:6px;padding:8px;margin:6px 0;}"
                " .arg{color:#80deea;}"
                " .obs{background:#37474f;color:#a5d6a7;border-radius:6px;padding:8px;margin:6px 0;white-space:pre-wrap;max-height:200px;overflow:auto;}"
                " .ans{background:#e3f2fd;border-radius:6px;padding:8px;margin:6px 0;}"
                " .gold{background:#fff3e0;border-radius:6px;padding:6px;margin:4px 0;}"
                " .meta{color:#555;font-size:12px;margin:4px 0;}"
                " summary{cursor:pointer;font-weight:bold;}</style>")

    for r in rows:
        rw = r.get("reward", 0.0)
        cls = "ok" if rw >= 0.9 else ("mid" if rw >= 0.5 else "bad")
        fid = r.get("financebench_id", "?")
        body.append(f'<details class="row {cls}">')
        body.append(f'<summary>[{rw:.2f}] {esc(fid)}</summary>')
        body.append(f'<div class="meta">Reward: {rw:.2f} | Tool calls: {len(r.get("tool_calls", []))} | Wall: {r.get("wall_s", "?")}s</div>')
        body.append(f'<div><b>Q:</b> {esc(r.get("question", ""))}</div>')
        body.append(f'<div class="gold" style="font-size:14px"><b>\u2b50 GOLD (ground truth):</b> {esc(r.get("gold", ""))}</div>')

        trace = r.get("trace", [])
        for i, t in enumerate(trace):
            role = t.get("role")
            if role == "assistant":
                tcs = t.get("tool_calls", [])
                txt = t.get("content", "").strip()
                if txt:
                    body.append(f'<div class="ans">{esc(txt)}</div>')
                for tc in tcs:
                    name = tc.get("name", "?")
                    args_s = json.dumps(tc.get("arguments", {}), ensure_ascii=False, indent=2)
                    body.append(f'<div class="tool"><b>→ {esc(name)}</b><br><span class="arg">{esc(args_s)}</span></div>')
            elif role == "tool":
                cid = t.get("call_id", "")
                content = t.get("content", "")
                body.append(f'<div class="obs">{esc(content[:3000])}{"…" if len(content)>3000 else ""}</div>')

        body.append(f'<div class="ans"><b>Final:</b> {esc(r.get("answer_text", ""))}</div>')
        body.append('</details>')

    out_path.write_text("<!doctype html><html><head><meta charset='utf-8'><title>"
                        f"{esc(args.label)} traces</title></head><body>" + "\n".join(body) + "</body></html>")
    print(json.dumps({"rows": len(rows), "out": str(out_path)}))

if __name__ == "__main__":
    main()
