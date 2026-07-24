#!/usr/bin/env python3
"""
behind_the_scenes.py — a "press tour" exporter for The Daily Chance
===================================================================
Writes a single self-contained HTML page (`behind_the_scenes.html`) with two
tabs:

  1. REAL HEADLINES  — every real headline harvested from the RSS feeds this
     run (the raw material that survives is_bad_raw_material), grouped by
     source, with a live link to each.

  2. RECONSTRUCTED    — the largest practical sample of minted crosses. It walks
     the subject×predicate pools the SAME way the browser does (mintCross),
     applying the exact same two screens — same-origin reject (subj.id ==
     pred.id) and crossIsDefamatory — then de-duplicates and caps the sample.

This reuses the real pipeline (harvest, splitter, build_parts, is_defamatory),
so what you see is what the instrument actually has to work with. It does NOT
call the Anthropic API — the split is deterministic spaCy — so it is cheap and
offline-safe once feeds are fetched.

Run:   python behind_the_scenes.py
Then:  open behind_the_scenes.html
Optional:  CAP=5000 python behind_the_scenes.py   # raise the corpus cap
"""

import os
import sys
import html
import json
import random
import datetime
import itertools

import generate  # the real pipeline lives here

CAP = int(os.environ.get("CAP", "2000"))   # max reconstructed headlines on the page
SEED = os.environ.get("SEED")              # set for reproducible sampling
if SEED:
    random.seed(int(SEED))


# ---------------------------------------------------------------------------
# Screens, mirrored from the browser (template.html). We re-implement the JS
# crossIsDefamatory surface-string check here rather than calling spaCy per
# cross (which would be far too slow over tens of thousands of pairs). The
# server already pre-computed subj["is_person"], so this matches the browser.
# ---------------------------------------------------------------------------
def cross_is_defamatory(subj, pred, accusation_verbs):
    if not subj.get("is_person"):
        return False
    hay = " ".join([
        pred.get("verb_sing", "") or "",
        pred.get("verb_plur", "") or "",
        pred.get("rest", "") or "",
        pred.get("modal", "") or "",
    ]).lower()
    for w in hay.replace("'", " ").split():
        w = "".join(ch for ch in w if ch.isalpha())
        if w and w in accusation_verbs:
            return True
    return False


def pick_verb(pred, subj_plural):
    if pred.get("modal"):
        return pred.get("verb_plur") or pred.get("verb_sing") or ""
    return (pred.get("verb_plur") or "") if subj_plural else (pred.get("verb_sing") or "")


def mint(subj, pred):
    """Assemble one cross exactly as the browser's mintCross does."""
    verb = pick_verb(pred, subj.get("plural", False))
    pieces = [subj["text"]]
    if pred.get("modal"):
        pieces.append(pred["modal"])
    if verb:
        pieces.append(verb)
    if pred.get("rest"):
        pieces.append(pred["rest"])
    headline = " ".join(p for p in pieces if p).strip()
    if headline:
        headline = headline[0].upper() + headline[1:]
    return headline


# ---------------------------------------------------------------------------
# Build the data: harvest -> split -> part pools -> exhaustive cross sample.
# ---------------------------------------------------------------------------
def build():
    print("Harvesting feeds...", file=sys.stderr)
    items = generate.harvest()
    print(f"  {len(items)} real headlines kept", file=sys.stderr)

    print("Splitting (deterministic spaCy)...", file=sys.stderr)
    splits = generate.splitter.split_items(items)

    subjects, predicates = generate.build_parts(splits)
    accusation_verbs = {v.lower() for v in generate.ACCUSATION_VERBS}
    print(f"  {len(subjects)} subjects, {len(predicates)} predicates",
          file=sys.stderr)

    # Full theoretical space = |subjects| x |predicates|, minus same-origin and
    # defamatory pairs, minus duplicate surface strings. For the page we sample
    # up to CAP distinct crosses. We iterate a shuffled product so the sample is
    # a fair spread across the whole space rather than the first N rows.
    total_space = len(subjects) * len(predicates)
    print(f"  theoretical cross space: {total_space:,}", file=sys.stderr)

    idx_pairs = list(itertools.product(range(len(subjects)), range(len(predicates))))
    random.shuffle(idx_pairs)

    seen = set()
    crosses = []
    screened_same_origin = 0
    screened_defam = 0
    for si, pi in idx_pairs:
        subj, pred = subjects[si], predicates[pi]
        if subj["id"] == pred["id"]:
            screened_same_origin += 1
            continue
        if cross_is_defamatory(subj, pred, accusation_verbs):
            screened_defam += 1
            continue
        headline = mint(subj, pred)
        if not headline:
            continue
        key = headline.lower()
        if key in seen:
            continue
        seen.add(key)
        crosses.append({
            "headline": headline,
            "subject_src": subj.get("src", ""),
            "subject_orig": subj.get("orig", ""),
            "subject_link": subj.get("link", ""),
            "predicate_src": pred.get("src", ""),
            "predicate_orig": pred.get("orig", ""),
            "predicate_link": pred.get("link", ""),
        })
        if len(crosses) >= CAP:
            break

    # PP-swap crosses: batch-wide indexed PP-swap, shown here as its own
    # demonstrated feature — independent sample from the verb-cross pool
    # above, not a mirror of what generate.py actually published. Uses the
    # same build_pp_index / combine_index_pp_swap machinery as production.
    prep_index = generate.build_pp_index(splits)
    pp_seen = set()
    pp_crosses = []
    pp_pool = list(splits)
    random.shuffle(pp_pool)
    for rec in pp_pool:
        for pc in generate.combine_index_pp_swap(rec, rec, prep_index):
            headline = pc.get("headline")
            if not headline:
                continue
            key = headline.lower()
            if key in pp_seen:
                continue
            pp_seen.add(key)
            pp_crosses.append(pc)
            if len(pp_crosses) >= CAP:
                break
        if len(pp_crosses) >= CAP:
            break
    print(f"  PP-swap crosses: {len(pp_crosses)} shown", file=sys.stderr)

    # Group real headlines by source for the first tab.
    by_source = {}
    for it in items:
        by_source.setdefault(it["source"], []).append(it)
    real_grouped = [
        {"source": src, "items": sorted(rows, key=lambda r: r["title"].lower())}
        for src, rows in sorted(by_source.items())
    ]

    return {
        "real_grouped": real_grouped,
        "real_total": len(items),
        "crosses": crosses,
        "pp_crosses": pp_crosses,
        "stats": {
            "subject_count": len(subjects),
            "predicate_count": len(predicates),
            "total_space": total_space,
            "screened_same_origin": screened_same_origin,
            "screened_defam": screened_defam,
            "shown": len(crosses),
            "cap": CAP,
            "pp_shown": len(pp_crosses),
            "generated_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
    }


# ---------------------------------------------------------------------------
# Render the self-contained HTML page. Data is embedded as JSON; the page is a
# single file you can open with `open behind_the_scenes.html` — no server.
# ---------------------------------------------------------------------------
def render(data, out_path="behind_the_scenes.html"):
    payload = json.dumps(data, ensure_ascii=False)
    page = PAGE_TEMPLATE.replace("/*__BTS_DATA__*/", payload)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    s = data["stats"]
    print(f"\nWrote {out_path}", file=sys.stderr)
    print(f"  real headlines: {data['real_total']}", file=sys.stderr)
    print(f"  reconstructed shown: {s['shown']:,} of "
          f"{s['total_space']:,} possible crosses", file=sys.stderr)


PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Daily Chance — Behind the Scenes</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Spectral:ital,wght@0,400;0,500;1,400&family=Spectral+SC:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --paper:#f3efe4; --ink:#1c1a17; --muted:#6b655b;
    --rule:#cdc6b6; --oxblood:#7c1f17; --chip:#e9e3d4;
  }
  *{box-sizing:border-box;}
  body{
    margin:0; background:var(--paper); color:var(--ink);
    font-family:"Spectral",Georgia,serif; line-height:1.5;
    -webkit-font-smoothing:antialiased;
  }
  .wrap{max-width:860px; margin:0 auto; padding:32px 22px 80px;}
  header{border-bottom:3px double var(--ink); padding-bottom:14px; margin-bottom:8px;}
  h1{
    font-family:"Playfair Display",Georgia,serif; font-weight:900;
    font-size:clamp(30px,6vw,52px); margin:0; letter-spacing:.01em; text-align:center;
  }
  .kicker{
    font-family:"Spectral SC",serif; text-transform:uppercase; letter-spacing:.18em;
    font-size:12px; color:var(--muted); text-align:center; margin:4px 0 0;
  }
  .stats{
    font-family:"Spectral SC",serif; font-size:12.5px; color:var(--muted);
    text-align:center; margin:12px 0 0; letter-spacing:.04em;
  }
  .stats b{color:var(--oxblood);}
  nav.tabs{
    display:flex; gap:0; margin:26px 0 24px; border-bottom:1px solid var(--rule);
  }
  .tab{
    appearance:none; background:none; border:none; cursor:pointer;
    font-family:"Spectral SC",serif; font-size:14px; letter-spacing:.08em;
    text-transform:uppercase; color:var(--muted); padding:10px 18px;
    border-bottom:3px solid transparent; margin-bottom:-1px;
  }
  .tab[aria-selected="true"]{color:var(--ink); border-bottom-color:var(--oxblood);}
  .tab:hover{color:var(--ink);}
  .panel{display:none;}
  .panel.active{display:block;}

  .toolbar{display:flex; gap:10px; align-items:center; margin-bottom:18px; flex-wrap:wrap;}
  .search{
    flex:1 1 240px; font-family:"Spectral",serif; font-size:15px;
    padding:9px 12px; border:1px solid var(--rule); background:#fff;
    border-radius:2px; color:var(--ink);
  }
  .count{font-family:"Spectral SC",serif; font-size:12px; color:var(--muted); letter-spacing:.05em;}

  /* Real-headlines tab */
  .source-group{margin-bottom:26px;}
  .source-name{
    font-family:"Spectral SC",serif; font-size:13px; letter-spacing:.1em;
    text-transform:uppercase; color:var(--oxblood); border-bottom:1px solid var(--rule);
    padding-bottom:5px; margin-bottom:10px;
  }
  .real-item{padding:5px 0; border-bottom:1px dotted var(--rule);}
  .real-item a{color:var(--ink); text-decoration:none;}
  .real-item a:hover{text-decoration:underline; text-decoration-color:var(--oxblood);}

  /* Reconstructed tab */
  .cross{
    padding:14px 0; border-bottom:1px solid var(--rule);
  }
  .cross-head{
    font-family:"Playfair Display",Georgia,serif; font-weight:700;
    font-size:clamp(18px,3vw,23px); line-height:1.25; margin:0 0 6px;
  }
  .prov{font-size:13px; color:var(--muted);}
  .prov .lbl{
    font-family:"Spectral SC",serif; text-transform:uppercase;
    letter-spacing:.06em; font-size:10.5px; color:var(--oxblood); margin-right:4px;
  }
  .prov a{color:var(--muted);}
  .prov .row{margin:2px 0;}
  .more{
    display:block; margin:24px auto 0; font-family:"Spectral SC",serif;
    letter-spacing:.08em; text-transform:uppercase; font-size:13px;
    background:var(--ink); color:var(--paper); border:none; cursor:pointer;
    padding:11px 22px; border-radius:2px;
  }
  .more:hover{background:var(--oxblood);}
  .empty{color:var(--muted); font-style:italic; padding:30px 0; text-align:center;}
  mark{background:var(--oxblood); color:var(--paper); padding:0 2px; border-radius:2px;}
  footer{
    margin-top:50px; padding-top:16px; border-top:3px double var(--ink);
    text-align:center; font-family:"Spectral SC",serif; font-size:11px;
    letter-spacing:.12em; color:var(--muted); text-transform:uppercase;
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>The Daily Chance</h1>
    <p class="kicker">Behind the Scenes — The Working Material</p>
  </header>
  <p class="stats" id="stats"></p>

  <nav class="tabs" role="tablist">
    <button class="tab" id="tab-real" role="tab" aria-selected="true" aria-controls="panel-real">Real Headlines</button>
    <button class="tab" id="tab-recon" role="tab" aria-selected="false" aria-controls="panel-recon">Reconstructed</button>
    <button class="tab" id="tab-ppswap" role="tab" aria-selected="false" aria-controls="panel-ppswap">PP-Swap</button>
  </nav>

  <section class="panel active" id="panel-real" role="tabpanel" aria-labelledby="tab-real">
    <div class="toolbar">
      <input class="search" id="search-real" type="search" placeholder="Filter real headlines…">
      <span class="count" id="count-real"></span>
    </div>
    <div id="real-list"></div>
  </section>

  <section class="panel" id="panel-recon" role="tabpanel" aria-labelledby="tab-recon">
    <div class="toolbar">
      <input class="search" id="search-recon" type="search" placeholder="Filter reconstructed headlines…">
      <span class="count" id="count-recon"></span>
    </div>
    <div id="recon-list"></div>
    <button class="more" id="more-btn">Show 100 More</button>
  </section>

  <section class="panel" id="panel-ppswap" role="tabpanel" aria-labelledby="tab-ppswap">
    <div class="toolbar">
      <input class="search" id="search-ppswap" type="search" placeholder="Filter PP-swap headlines…">
      <span class="count" id="count-ppswap"></span>
    </div>
    <div id="ppswap-list"></div>
    <button class="more" id="more-btn-pp">Show 100 More</button>
  </section>

  <footer>— 30 —</footer>
</div>

<script id="bts-data" type="application/json">/*__BTS_DATA__*/</script>
<script>
const DATA = JSON.parse(document.getElementById('bts-data').textContent);
const esc = s => (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const escapeRegExp = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
function wordMatch(text, term){
  if(!term) return true;
  return new RegExp('\\b'+escapeRegExp(term)+'\\b', 'i').test(text||'');
}
function highlight(text, term){
  const escaped = esc(text);
  if(!term) return escaped;
  const re = new RegExp('\\b('+escapeRegExp(term)+')\\b', 'gi');
  return escaped.replace(re, '<mark>$1</mark>');
}
const link = (text, url, term) => {
  const body = term ? highlight(text, term) : esc(text);
  const q = '\u201C'+body+'\u201D';
  return (url && /^https?:\/\//i.test(url))
    ? '<a href="'+esc(url)+'" target="_blank" rel="noopener">'+q+'</a>' : q;
};

// ── Stats line ─────────────────────────────────────
const s = DATA.stats;
document.getElementById('stats').innerHTML =
  '<b>'+DATA.real_total+'</b> real headlines &nbsp;·&nbsp; ' +
  '<b>'+s.subject_count+'</b> subjects × <b>'+s.predicate_count+'</b> predicates = ' +
  '<b>'+s.total_space.toLocaleString()+'</b> possible crosses &nbsp;·&nbsp; ' +
  'showing <b>'+s.shown.toLocaleString()+'</b>';

// ── Tabs ───────────────────────────────────────────
const tabs = {
  real:   {btn: document.getElementById('tab-real'),   panel: document.getElementById('panel-real')},
  recon:  {btn: document.getElementById('tab-recon'),  panel: document.getElementById('panel-recon')},
  ppswap: {btn: document.getElementById('tab-ppswap'), panel: document.getElementById('panel-ppswap')},
};
function select(name){
  for(const k in tabs){
    const on = k===name;
    tabs[k].btn.setAttribute('aria-selected', on?'true':'false');
    tabs[k].panel.classList.toggle('active', on);
  }
}
tabs.real.btn.onclick   = ()=>select('real');
tabs.recon.btn.onclick  = ()=>select('recon');
tabs.ppswap.btn.onclick = ()=>select('ppswap');

// ── Real headlines tab ─────────────────────────────
const realList = document.getElementById('real-list');
const realCount = document.getElementById('count-real');
function renderReal(filter){
  const f = (filter||'').trim();
  let shown = 0;
  const html = DATA.real_grouped.map(g => {
    const rows = g.items.filter(it => wordMatch(it.title, f));
    if(!rows.length) return '';
    shown += rows.length;
    return '<div class="source-group"><div class="source-name">'+esc(g.source)+
      ' · '+rows.length+'</div>' +
      rows.map(it => '<div class="real-item">'+link(it.title, it.link, f)+'</div>').join('') +
      '</div>';
  }).join('');
  realList.innerHTML = html || '<p class="empty">No matches.</p>';
  realCount.textContent = shown + ' shown';
}
document.getElementById('search-real').addEventListener('input', e => renderReal(e.target.value));

// ── Reconstructed tab ──────────────────────────────
const reconList = document.getElementById('recon-list');
const reconCount = document.getElementById('count-recon');
const moreBtn = document.getElementById('more-btn');
const PAGE = 100;
let reconFiltered = DATA.crosses;
let reconShown = 0;

let reconTerm = '';
function crossHtml(c, term){
  return '<div class="cross">' +
    '<p class="cross-head">'+highlight(c.headline, term)+'</p>' +
    '<div class="prov">' +
      '<div class="row"><span class="lbl">Subject</span>'+esc(c.subject_src)+' — '+link(c.subject_orig, c.subject_link, term)+'</div>' +
      '<div class="row"><span class="lbl">Predicate</span>'+esc(c.predicate_src)+' — '+link(c.predicate_orig, c.predicate_link, term)+'</div>' +
    '</div></div>';
}
function renderRecon(reset){
  if(reset){ reconList.innerHTML=''; reconShown=0; }
  const next = reconFiltered.slice(reconShown, reconShown+PAGE);
  reconList.insertAdjacentHTML('beforeend', next.map(c=>crossHtml(c, reconTerm)).join(''));
  reconShown += next.length;
  reconCount.textContent = reconShown.toLocaleString()+' of '+reconFiltered.length.toLocaleString()+' shown';
  moreBtn.style.display = (reconShown < reconFiltered.length) ? 'block' : 'none';
  if(!reconFiltered.length) reconList.innerHTML='<p class="empty">No matches.</p>';
}
moreBtn.onclick = ()=>renderRecon(false);
document.getElementById('search-recon').addEventListener('input', e => {
  reconTerm = e.target.value.trim();
  reconFiltered = reconTerm ? DATA.crosses.filter(c => wordMatch(c.headline, reconTerm)) : DATA.crosses;
  renderRecon(true);
});

// ── PP-Swap tab ─────────────────────────────────────
const ppList = document.getElementById('ppswap-list');
const ppCount = document.getElementById('count-ppswap');
const moreBtnPP = document.getElementById('more-btn-pp');
let ppFiltered = DATA.pp_crosses;
let ppShown = 0;

let ppTerm = '';
function crossHtmlPP(c, term){
  return '<div class="cross">' +
    '<p class="cross-head">'+highlight(c.headline, term)+'</p>' +
    '<div class="prov">' +
      '<div class="row"><span class="lbl">Subject</span>'+esc(c.subject_src)+' — '+link(c.subject_orig, c.subject_link, term)+'</div>' +
      '<div class="row"><span class="lbl">PP source</span>'+esc(c.pp_src)+' — '+link(c.pp_orig, c.pp_link, term)+'</div>' +
    '</div></div>';
}
function renderPP(reset){
  if(reset){ ppList.innerHTML=''; ppShown=0; }
  const next = ppFiltered.slice(ppShown, ppShown+PAGE);
  ppList.insertAdjacentHTML('beforeend', next.map(c=>crossHtmlPP(c, ppTerm)).join(''));
  ppShown += next.length;
  ppCount.textContent = ppShown.toLocaleString()+' of '+ppFiltered.length.toLocaleString()+' shown';
  moreBtnPP.style.display = (ppShown < ppFiltered.length) ? 'block' : 'none';
  if(!ppFiltered.length) ppList.innerHTML='<p class="empty">No matches.</p>';
}
moreBtnPP.onclick = ()=>renderPP(false);
document.getElementById('search-ppswap').addEventListener('input', e => {
  ppTerm = e.target.value.trim();
  ppFiltered = ppTerm ? DATA.pp_crosses.filter(c => wordMatch(c.headline, ppTerm)) : DATA.pp_crosses;
  renderPP(true);
});

// ── Init ───────────────────────────────────────────
renderReal('');
renderRecon(true);
renderPP(true);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    render(build())
