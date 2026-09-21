#!/usr/bin/env python3
"""Human semantic adjudication of strict/lenient scorer disagreements —
blinded annotation package (paper/PREREG_HUMAN_ADJUDICATION.md).

Source: the 600-state prospective TextVQA log (all 13,200 generations).
Unit = unique (state, whitespace/case-normalized answer) pair. Items (300):
  * 200 disagreement items: strict == 0 and format-tolerant > 0 (245 in the
    population), stratified by the dominant action family of the item's
    rollouts (proportional allocation, seed 20260921)
  * 50 both-correct (strict >= .5 and lenient >= .5) + 50 both-wrong
    (strict == 0 and lenient == 0) controls, random (seed 20260921)
Blinding: the sheet shows item_id, image, question, gold references (with
counts) and the answer text only — never the arm, replay, resolution or
either scorer's verdict. Global shuffle. The private mapping stays closed.
Outputs:
  data/human_adjudication/items.jsonl          private mapping (do not open)
  paper/human_adjudication_sheet.html          self-contained blinded sheet
                                               (images embedded; C / I / A
                                               radio buttons; autosave;
                                               "Export TSV" button)
  paper/human_adjudication_sheet.tsv           same items, plain TSV
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from ppu.evaluator import normalize_answer  # noqa: E402
from strong_controller_common import lenient  # noqa: E402

SEED = 20260921
N_CTRL = 50
N_DIS = 200
LOG = "logs/prospective_textvqa.jsonl"


def main():
    items = {}
    for line in open(ROOT / LOG):
        r = json.loads(line)
        if r.get("type") != "rollout":
            continue
        s = float(r["score_after"]) >= 0.5
        le = lenient(r["answer_after"], r["gold"]) >= 0.5
        key = (r["sample_id"], normalize_answer(r["answer_after"] or ""))
        it = items.setdefault(key, {
            "sample_id": r["sample_id"], "answer_norm": key[1], "answer_raw": r["answer_after"] or "",
            "question": r["question"], "gold": r["gold"], "image_path": r["image_path"],
            "strict": bool(s), "lenient": bool(le), "s_raw": float(r["score_after"]),
            "l_raw": float(lenient(r["answer_after"], r["gold"])), "n_rollouts": 0, "fams": Counter(), "reps": Counter()})
        it["n_rollouts"] += 1; it["fams"][r["action"]["family"]] += 1; it["reps"][r["replay"]] += 1
    strata = defaultdict(list)
    for it in items.values():
        if it["s_raw"] == 0.0 and it["l_raw"] > 0.0:
            it["stratum"] = "disagree"
        elif it["s_raw"] >= 0.5 and it["l_raw"] >= 0.5:
            it["stratum"] = "both_correct"
        elif it["s_raw"] == 0.0 and it["l_raw"] == 0.0:
            it["stratum"] = "both_wrong"
        else:
            it["stratum"] = "other"
        it["dominant_family"] = max(sorted(it["fams"]), key=lambda f: it["fams"][f])
        strata[it["stratum"]].append(it)
    rng = random.Random(SEED)
    # disagreement: proportional allocation over dominant family (largest remainder)
    dis = sorted(strata["disagree"], key=lambda x: (x["sample_id"], x["answer_norm"]))
    by_fam = defaultdict(list)
    for it in dis:
        by_fam[it["dominant_family"]].append(it)
    quota = {f: N_DIS * len(v) / len(dis) for f, v in by_fam.items()}
    alloc = {f: int(q) for f, q in quota.items()}
    for f in sorted(quota, key=lambda f: quota[f] - alloc[f], reverse=True)[: N_DIS - sum(alloc.values())]:
        alloc[f] += 1
    chosen = []
    for f in sorted(by_fam):
        chosen += rng.sample(by_fam[f], alloc[f])
    print("disagreement allocation by dominant family:", alloc)
    for k in ("both_correct", "both_wrong"):
        pool = sorted(strata[k], key=lambda x: (x["sample_id"], x["answer_norm"]))
        chosen += rng.sample(pool, N_CTRL)
    for it in chosen:
        it["item_id"] = "h" + hashlib.sha256(f"{SEED}|{it['sample_id']}|{it['answer_norm']}".encode()).hexdigest()[:10]
    rng.shuffle(chosen)
    print(f"unique items {len(items)}; strata sizes { {k: len(v) for k, v in strata.items()} }; sheet {len(chosen)}")

    out = ROOT / "data/human_adjudication"; out.mkdir(exist_ok=True)
    with open(out / "items.jsonl", "w") as fh:
        for it in chosen:
            rec = dict(it); rec["fams"] = dict(it["fams"]); rec["reps"] = dict(it["reps"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out / "population_strata.json").write_text(json.dumps(
        {k: len(v) for k, v in strata.items()} | {"n_unique_items": len(items), "n_ctrl_per_stratum": N_CTRL, "n_dis": N_DIS,
         "disagree_definition": "strict == 0 and lenient > 0", "disagree_alloc_by_family": alloc, "seed": SEED}, indent=1))

    def gold_str(g):
        c = Counter(x.strip().lower() for x in g)
        return "; ".join(f"{a} ×{n}" for a, n in c.most_common())

    rows = ["item_id\tquestion\tgold_references\tanswer\tverdict\tnote"]
    cards = []
    thumbs = {}
    for it in chosen:
        p = it["image_path"]
        if p not in thumbs:
            im = Image.open(ROOT / p).convert("RGB"); im.thumbnail((720, 720))
            buf = io.BytesIO(); im.save(buf, format="JPEG", quality=80)
            thumbs[p] = base64.b64encode(buf.getvalue()).decode()
        rows.append("\t".join([it["item_id"], it["question"].replace("\t", " "), gold_str(it["gold"]),
                               it["answer_raw"].replace("\t", " ").replace("\n", " "), "", ""]))
        cards.append(f"""
<div class="card" id="{it['item_id']}">
 <div class="hdr"><span class="idx"></span> <code>{it['item_id']}</code></div>
 <img src="data:image/jpeg;base64,{thumbs[p]}" loading="lazy">
 <div class="q"><b>Question:</b> {html.escape(it['question'])}</div>
 <div class="g"><b>Gold references (annotator votes):</b> {html.escape(gold_str(it['gold']))}</div>
 <div class="a"><b>Model answer:</b> <span class="ans">{html.escape(it['answer_raw'])}</span></div>
 <div class="v">
  <label><input type="radio" name="v_{it['item_id']}" value="C"> Semantically correct</label>
  <label><input type="radio" name="v_{it['item_id']}" value="I"> Incorrect</label>
  <label><input type="radio" name="v_{it['item_id']}" value="A"> Ambiguous</label>
  <input type="text" class="note" name="n_{it['item_id']}" placeholder="note (optional)">
 </div>
</div>""")
    (ROOT / "paper/human_adjudication_sheet.tsv").write_text("\n".join(rows) + "\n")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Semantic adjudication sheet</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:20px auto;padding:0 16px;background:#fafafa}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px;margin:18px 0}}
.card img{{max-width:100%;max-height:520px;display:block;margin:8px 0;border:1px solid #eee}}
.hdr{{color:#666;font-size:13px}} .idx{{font-weight:bold;color:#333}}
.q,.g,.a{{margin:6px 0}} .ans{{background:#fff7d6;padding:2px 6px;border-radius:4px;font-size:1.05em}}
.v label{{margin-right:16px}} .note{{width:280px;margin-left:12px}}
.done{{border-color:#7c7}} #bar{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:10px 0;z-index:9}}
#bar input{{margin-right:10px}} button{{padding:6px 12px}}
</style></head><body>
<div id="bar"><b>Human semantic adjudication.</b> Judge whether the model prediction correctly answers the question in context.
 Mark <b>Correct</b> only when its overall meaning is consistent with the reference answers and does not introduce a contradiction or an incompatible alternative.
 A prediction that merely contains the reference token somewhere is not necessarily correct.
 Use <b>Ambiguous</b> when the available image / question / references do not support a confident judgment. Surface form (casing, punctuation, extra function words) does not matter; meaning does.
 <br>Annotator: <input id="who" placeholder="your name"> <button onclick="exportTSV()">Export TSV</button> <span id="prog"></span>
 <br><small>Answers autosave in this browser. Three annotators judge independently; do not discuss items before exporting.</small></div>
{''.join(cards)}
<script>
const ids={json.dumps([it['item_id'] for it in chosen])};
document.querySelectorAll('.card').forEach((c,i)=>c.querySelector('.idx').textContent=(i+1)+'/'+ids.length);
function key(){{return 'adj_'+(document.getElementById('who').value||'anon');}}
function load(){{const d=JSON.parse(localStorage.getItem(key())||'{{}}');for(const id of ids){{if(d[id]){{const r=document.querySelector('input[name=v_'+id+'][value='+d[id].v+']');if(r)r.checked=true;document.querySelector('input[name=n_'+id+']').value=d[id].n||'';}}}}mark();}}
function save(){{const d={{}};for(const id of ids){{const r=document.querySelector('input[name=v_'+id+']:checked');if(r)d[id]={{v:r.value,n:document.querySelector('input[name=n_'+id+']').value}};}}localStorage.setItem(key(),JSON.stringify(d));mark();}}
function mark(){{let n=0;for(const id of ids){{const r=document.querySelector('input[name=v_'+id+']:checked');document.getElementById(id).classList.toggle('done',!!r);if(r)n++;}}document.getElementById('prog').textContent=n+'/'+ids.length+' done';}}
document.addEventListener('change',save);document.addEventListener('input',e=>{{if(e.target.classList.contains('note'))save();}});
document.getElementById('who').addEventListener('change',load);
function exportTSV(){{const who=document.getElementById('who').value||'anon';let t='item_id\\tverdict\\tnote\\n';for(const id of ids){{const r=document.querySelector('input[name=v_'+id+']:checked');t+=id+'\\t'+(r?r.value:'')+'\\t'+document.querySelector('input[name=n_'+id+']').value.replace(/[\\t\\n]/g,' ')+'\\n';}}const b=new Blob([t],{{type:'text/tab-separated-values'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='verdicts_'+who+'.tsv';a.click();}}
load();
</script></body></html>"""
    (ROOT / "paper/human_adjudication_sheet.html").write_text(page)
    sha = hashlib.sha256((out / "items.jsonl").read_bytes()).hexdigest()[:16]
    print(f"sheet: {len(chosen)} items -> paper/human_adjudication_sheet.html "
          f"({(ROOT / 'paper/human_adjudication_sheet.html').stat().st_size / 1e6:.1f} MB); mapping sha256[:16] {sha}")


if __name__ == "__main__":
    main()
