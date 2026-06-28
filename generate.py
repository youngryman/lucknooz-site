#!/usr/bin/env python3
"""
LuckNooz generator — LLM edition
================================
Fetches news headlines from RSS feeds, then uses Claude (Haiku) to perform the
recombination directly: split each headline before its main verb, optionally
detach a short trailing prepositional phrase, cross the pieces between
headlines, and re-conjugate ONLY the pivoting verb to agree with its new
subject. Nothing else is changed — every other word comes verbatim from the
source. Pure principled chance; no editing, no invention, surreal preserved.

Output: writes `lucknooz.html` by injecting the recombined headlines into the
template. The browser just displays them (it no longer assembles anything).

Run locally:   python generate.py
Requires the ANTHROPIC_API_KEY environment variable to be set.
"""

import os
import sys
import html
import json
import random
import datetime

import re
import feedparser
import anthropic

import splitter  # deterministic spaCy-based headline splitter (PASS 1)

# ── Defamation screen ───────────────────────────
# Blocks minted headlines that pair a named person (subject slot) with an
# accusation verb — i.e. fabricated damaging claims about a real individual.
# Loads its OWN spaCy pipeline WITH NER (the splitter disables NER, so its
# nlp object can't see PERSON entities and cannot be reused here).
ACCUSATION_VERBS = {
    'accuse', 'accused', 'charge', 'charged', 'arrest', 'arrested',
    'indict', 'indicted', 'allege', 'alleged', 'convict', 'convicted',
    'kill', 'killed', 'murder', 'murdered', 'assault', 'assaulted',
    'rape', 'raped', 'abuse', 'abused', 'steal', 'stole', 'stolen',
    'defraud', 'defrauded', 'lie', 'lied', 'bribe', 'bribed',
    'smuggle', 'smuggled', 'launder', 'laundered', 'embezzle', 'embezzled',
    'attack', 'attacked', 'beat', 'beaten', 'stab', 'stabbed', 'shoot', 'shot'
}

_SCREEN_NLP = None

def _screen_nlp():
    """Lazily load a spaCy pipeline WITH NER for the defamation screen."""
    global _SCREEN_NLP
    if _SCREEN_NLP is None:
        import spacy
        _SCREEN_NLP = spacy.load("en_core_web_sm")  # NER enabled (no disable)
    return _SCREEN_NLP

def is_defamatory(headline_text, subject_text):
    """Block when a named person in the subject slot is paired with an
    accusation verb anywhere in the minted headline."""
    if not headline_text:
        return False
    doc = _screen_nlp()(headline_text)
    person_ents = [ent for ent in doc.ents if ent.label_ == 'PERSON']
    if not person_ents:
        return False
    subj = (subject_text or "").strip()
    person_in_subject = any(ent.text in subj or subj in ent.text
                            for ent in person_ents) if subj else True
    if not person_in_subject:
        return False
    for token in doc:
        if (token.lemma_.lower() in ACCUSATION_VERBS
                or token.text.lower() in ACCUSATION_VERBS):
            return True
    return False


def _subject_is_person(subject_text):
    """True if the subject fragment names a real individual (spaCy PERSON).
    Used to pre-compute an is_person flag shipped to the browser, so the
    client-side screen needs no NLP."""
    s = (subject_text or "").strip()
    if not s:
        return False
    doc = _screen_nlp()(s)
    if any(ent.label_ == 'PERSON' for ent in doc.ents):
        return True
    # Fragment-alone NER is weak; retry in a minimal sentence frame.
    doc2 = _screen_nlp()(s + " spoke today.")
    return any(ent.label_ == 'PERSON' and ent.text in s for ent in doc2.ents)
# ─────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Feeds (same set as before) — category -> [(source_label, feed_url), ...]
# ---------------------------------------------------------------------------
FEEDS = {
    "core_news": [
        ("rss.nytimes.com",     "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"),
        ("www.theguardian.com", "https://www.theguardian.com/world/rss"),
    ],
    "business": [
        ("feeds.a.dj.com",    "https://feeds.a.dj.com/rss/RSSWorldNews.xml"),
        ("www.bloomberg.com", "https://feeds.bloomberg.com/markets/news.rss"),
    ],
    "science_tech": [
        ("www.sciencedaily.com",  "https://www.sciencedaily.com/rss/top/science.xml"),
        ("feeds.arstechnica.com", "https://feeds.arstechnica.com/arstechnica/index"),
        ("www.nature.com",        "https://www.nature.com/nature.rss"),
        ("phys.org-space",   "https://phys.org/rss-feed/space-news/"),
        ("phys.org-archaeo", "https://phys.org/rss-feed/science-news/archaeology/"),
        ("sci.news",         "https://www.sci.news/feed"),
    ],
    "culture": [
        ("www.rollingstone.com","https://www.rollingstone.com/feed/"),
        ("pagesix.com",         "https://pagesix.com/feed/"),
        ("tmz.com",             "https://www.tmz.com/rss.xml"),
        ("usmagazine.com",      "https://www.usmagazine.com/feed/"),
        ("justjared.com",       "https://www.justjared.com/feed/"),
    ],
    "offbeat": [
        ("www.sciencedaily.com", "https://www.sciencedaily.com/rss/strange_offbeat.xml"),
        ("www.marinelink.com",   "https://www.marinelink.com/news/rss"),
        ("upi-odd",      "https://rss.upi.com/news/odd_news.rss"),
    ],
    "world": [
        ("straitstimes.com",   "https://www.straitstimes.com/news/world/rss.xml"),
        ("abc.net.au",         "https://www.abc.net.au/news/feed/2942460/rss.xml"),
        ("timesofindia.com",   "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
        ("japantimes.co.jp",   "https://www.japantimes.co.jp/feed/"),
        ("irishtimes.com",     "https://www.irishtimes.com/cmlink/news-1.1319192"),
        ("dw.com",             "https://rss.dw.com/rdf/rss-en-all"),
        ("france24.com",       "https://www.france24.com/en/rss"),
        ("timesofisrael.com",  "https://www.timesofisrael.com/feed/"),
        ("aljazeera.com",      "https://www.aljazeera.com/xml/rss/all.xml"),
    ],
}

MAX_PER_FEED = 20
MODEL = "claude-haiku-4-5"
PP_MAX_WORDS = 6           # detach a trailing prepositional phrase only if <= this
PAIRS_PER_BATCH = 10       # how many headline pairs to send per API call
MAX_WORDS = 16             # reject headlines longer than this (multi-clause monsters)

# Pre-filter: reject bad raw material BEFORE the LLM ever sees it. Rejection is
# a simpler, more reliable judgment than recombination, so we do it in Python.
QUESTION_OPENERS = {"how", "what", "why", "who", "when", "where", "which", "whose"}

# Words that signal a finite verb is present. A headline with none of these and
# no obvious verb morphology is treated as verbless (deferred to Phase 2).
COMMON_VERBS = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "says", "say", "said", "will", "won", "gets", "get",
    "got", "makes", "make", "made", "goes", "go", "went", "sees", "see", "saw",
    "finds", "find", "found", "takes", "take", "took", "gives", "give", "gave",
    "holds", "hold", "held", "keeps", "keep", "kept", "leads", "lead", "led",
    "wins", "win", "loses", "lose", "lost", "breaks", "break", "broke",
    "rises", "rise", "rose", "falls", "fall", "fell", "hits", "hit", "adds",
    "add", "added", "calls", "call", "called", "shows", "show", "shaped",
    "surges", "surge", "surged", "mulling", "drills", "drill", "proves",
    "prove", "proved", "admits", "admit", "tops", "denies", "deny", "denied",
    "reveals", "reveal", "opens", "open", "opened", "enters", "enter",
    "shakes", "shake", "shook", "induces", "induce", "suppresses", "suppress",
    "illuminates", "illuminate", "deepens", "deepen", "challenges", "glean",
}


def looks_verbless(title):
    """Heuristic: does this headline appear to have NO finite verb?

    No spaCy here, so we approximate: if no token is a known common verb AND no
    token has typical finite-verb morphology (-s / -ed on a longish word), treat
    as verbless. Conservative — when in doubt, keep it (let the LLM try)."""
    words = [w.strip(".,:;!?'\"()").lower() for w in title.split()]
    for w in words:
        if w in COMMON_VERBS:
            return False
        if len(w) > 4 and (w.endswith("ed") or (w.endswith("s") and not w.endswith("ss"))):
            return False
    return True


def is_bad_raw_material(title):
    """Return a reason string if the headline should be rejected, else None."""
    t = title.strip()
    if not t:
        return "empty"
    if t.endswith("...") or t.endswith("\u2026"):
        return "truncated"
    if not t[0].isalnum():
        return "symbol-opener"
    words = t.split()
    first = words[0].strip(".,:;!?'\"()").lower()
    if first in QUESTION_OPENERS:
        return "question-opener"
    if len(words) > MAX_WORDS:
        return "too-long"
    internal = t[:-1] if t else t
    if sum(internal.count(p) for p in (". ", "? ", "! ", ": ")) >= 2:
        return "multi-clause"
    if looks_verbless(t):
        return "verbless"
    return None

# ---------------------------------------------------------------------------
# The instruction. This is the heart of the system — the "pure principled
# chance" rule, stated as precisely as possible.
# ---------------------------------------------------------------------------
SPLIT_RULES = """You split news headlines into grammatical parts. Do ONLY this \
— do not combine or change anything.

For each headline given, identify:
  - SUBJECT: the words before the main verb (the doer of the action)
  - VERB: the single main verb (the action word of the core clause)
  - REST: everything after the main verb

FINDING THE MAIN VERB:
- It is usually EARLY; the subject before it is usually SHORT (1-4 words: a \
name, organization, or short noun phrase).
- Look THROUGH helper words to the real verb: modals (may, might, could, \
should, would, can, will, must), "have/has/had" or "is/are/was/were" + \
participle, and "to" + verb. In "Scientists may have debunked X", the main \
verb is "debunked". In "Watson keen to start again", it is "start".
- THE VERB IS THE WHOLE FINITE VERB, INCLUDING A LEADING COPULA OR AUXILIARY. \
If the core clause is "X is fake", the VERB is "is" (not empty, not buried in \
REST). If it is "X are being profiled", the VERB is "are" and "being profiled" \
goes in REST. Never leave a finite "is/are/was/were/has/have/do/does/did" \
sitting at the front of REST — that word IS the verb to record, so it can be \
re-conjugated later. This is the single most common mistake; check for it every \
time.
- If a modal governs the verb (e.g. "could be"), record the modal in a \
separate MODAL field and put the bare verb in VERB.
- Prefer the EARLIEST workable verb (shortest subject). If your subject is \
more than ~6 words, you went too far — look again at the front.

BEWARE NOUNS THAT LOOK LIKE VERBS (critical):
- Many headline words can be either a noun or a verb: "Shares", "Prices", \
"Costs", "Reports", "Talks", "Hits", "Plans", "Strikes", "Cuts", "Aims", \
"Funds", "Studies", "Bonds". When such a word sits right after the opening \
noun, it is almost always still part of the SUBJECT (a noun), NOT the verb. \
In "SpaceX Shares Close 19% Higher", the subject is "SpaceX Shares" and the \
verb is "Close" — NOT "Shares". In "Prices Likely to Stay Higher", the subject \
is "Prices", there is no early finite verb, and "Stay" (after "to") is the verb.
- Test before committing: read SUBJECT + VERB aloud as a tiny sentence. If it \
is not something a person could SAY as an action ("SpaceX shares told" — no; \
"Switzerland apprehended" — no; a country does not get apprehended), you picked \
the wrong verb. Back up and find the real one, or treat the headline as \
verbless if its true subject only appears later.
- If picking a verb would strand its real subject (the doer named later in the \
headline), do not pick it. Better to mark the headline verbless than to split \
at a verb whose subject you would throw away.

SPECIAL CASES:
- If the main verb is the FIRST word (no subject before it), set "subject" to \
empty string "". Such a headline can only be used as a predicate later.
- If there is NO finite main verb at all, set "verb" to empty string "".

Return ONLY a JSON array, one object per headline in the order given, each with:
  "subject": string (may be empty)
  "modal": string (the governing modal if any, else empty)
  "verb": string (the bare main verb, may be empty)
  "rest": string (everything after the verb)
  "plural": true if the subject is grammatically plural, false if singular. \
Use clues from the ORIGINAL headline — a later possessive ("his","their"), \
pronoun ("he","they"), or the noun itself — to decide.
"""

COMBINE_RULES = """You recombine pre-split news headline parts by pure \
principled chance. You will be given two already-split headlines, each as \
SUBJECT / MODAL / VERB / REST / PLURAL.

Produce TWO recombined headlines by crossing them:
  A) Subject 1 + (Verb 2 adjusted) + Rest 2
  B) Subject 2 + (Verb 1 adjusted) + Rest 1

CROSSING IS MANDATORY: In every headline you output, the SUBJECT must come from \
ONE source headline and the VERB+REST from the OTHER. NEVER take the subject \
and predicate from the same headline — that just echoes an original headline \
and is useless. If you find yourself reproducing one of the two input \
headlines unchanged (or nearly so), you have failed to cross — discard it. \
subject_src and predicate_src must always be DIFFERENT sources.

THE ONE PERMITTED CHANGE — verb agreement only:
- Re-conjugate the moved verb to agree with its NEW subject, using that \
subject's PLURAL flag (true = plural). Singular: "drives"; plural: "drive". \
Keep the original tense (past stays past: "drove").
- PRESERVE NEGATION. If the moved part is marked NEGATED, the negation word \
(not/never/n't) is already inside the VERB text — keep it. Never drop it. A \
negated verb stays negated after the cross: "never knew" stays "never knew".
- COPULAS AND AUXILIARIES AGREE TOO. If the moved verb is "is/are", flip it to \
match the new subject: new singular subject -> "is", new plural -> "are". Same \
for "was/were" and "has/have". Example: predicate verb "are" + REST "fake", \
crossed onto singular subject "Trump" -> "Trump IS fake", never "Trump are \
fake". This is mandatory — a mismatched copula is the most glaring error.
- If the NEW subject carried a MODAL, keep that modal and put the verb in base \
form after it: subject "This World Cup" (modal "could") + verb "broadcast" -> \
"This World Cup could broadcast ...". A verb after a modal needs no agreement.
- Change NOTHING else. Every word except the pivoting verb must appear EXACTLY \
as given. Do not substitute, smooth, add, remove, fix awkwardness, or invent. \
Strange, surreal, awkward results are GOOD — preserve them.

SKIP RULES:
- If a subject is empty (the headline was verb-first or verbless), do not use \
it as a subject; you may skip that direction of the cross.
- If a verb is empty, you may instead attach the other headline's short \
trailing prepositional phrase if one exists; otherwise skip that direction.
- NEVER output one whole headline stuck onto another. If you cannot form a \
clean cross, omit it.

Return ONLY a JSON array (0, 1, or 2 objects), each with:
  "headline": the recombined string
  "subject_src": source label of the subject's headline
  "subject_orig": original text of the subject's headline
  "predicate_src": source label of the predicate's headline
  "predicate_orig": original text of the predicate's headline
"""


def clean_title(title):
    """Strip HTML markup and decode entities from a raw feed title."""
    if not title:
        return title
    no_tags = re.sub(r"<[^>]+>", "", title)
    decoded = html.unescape(no_tags)
    return re.sub(r"\s+", " ", decoded).strip()


def harvest():
    """Fetch headlines. Returns list of dicts: {title, source, category}."""
    items = []
    seen = set()
    for category, feeds in FEEDS.items():
        for source, url in feeds:
            try:
                parsed = feedparser.parse(url)
            except Exception as e:
                print(f"  ! {source}: {e}", file=sys.stderr)
                continue
            count = 0
            rejected = 0
            for entry in parsed.entries:
                if count >= MAX_PER_FEED:
                    break
                title = clean_title(getattr(entry, "title", "").strip())
                if not title or title in seen:
                    continue
                seen.add(title)
                reason = is_bad_raw_material(title)
                if reason:
                    rejected += 1
                    continue
                link = getattr(entry, "link", "") or ""
                items.append({"title": title, "source": source,
                              "category": category, "link": link})
                count += 1
            print(f"  {source}: {count} kept, {rejected} rejected", file=sys.stderr)
    return items


def make_pairs(items):
    """Shuffle ALL headlines together and pair them at random, ignoring
    category. Cross-category collisions are the point — a maritime headline
    can now land on a sports subject, etc."""
    pool = items[:]
    random.shuffle(pool)
    pairs = []
    for i in range(0, len(pool) - 1, 2):
        pairs.append((pool[i], pool[i + 1]))
    return pairs


def _call_json(client, system, user_msg, max_tokens=4000):
    """Call the model, return parsed JSON array, or [] on failure."""
    resp = client.messages.create(
        model=MODEL, max_tokens=max_tokens,
        system=system, messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  ! could not parse model output: {e}", file=sys.stderr)
        return []


def split_batch(client, items):
    """PASS 1: split each headline into parts using the DETERMINISTIC spaCy
    splitter (no model call). Returns list of dicts aligned with `items`, each
    with subject/modal/verb/rest/negated/plural plus source info.

    The `client` argument is kept for signature compatibility but unused — the
    split is now pure grammar, not LLM improvisation."""
    return splitter.split_items(items)


import re as _re_mod

# Irregular/copula forms LemmInflect handles, but we special-case the copulas
# and a couple of high-frequency auxiliaries for safety and capitalization.
_COPULA_WANT = {
    ("is", True): "are", ("are", False): "is",
    ("was", True): "were", ("were", False): "was",
    ("has", True): "have", ("have", False): "has",
}

_LEMM = None


def _lemm():
    """Lazy-load LemmInflect once."""
    global _LEMM
    if _LEMM is None:
        import lemminflect
        _LEMM = lemminflect
    return _LEMM


def _match_case(template, word):
    """Carry template's capitalization onto word."""
    if template and template[0].isupper():
        return word[:1].upper() + word[1:]
    return word


def _agree_present_verb(verb, plural):
    """Given a single present-tense verb token, return it inflected to agree
    with the new subject's number: singular -> VBZ (crushes), plural -> VBP
    (crush). Returns None if we can't/shouldn't change it (past tense,
    participle, modal, unknown)."""
    low = verb.lower()
    # copulas/auxiliaries first (deterministic, keeps irregulars correct)
    if (low, plural) in _COPULA_WANT:
        return _match_case(verb, _COPULA_WANT[(low, plural)])
    if low in ("is", "are", "was", "were", "has", "have", "be", "been",
               "will", "would", "can", "could", "may", "might", "must",
               "shall", "should", "do", "does", "did"):
        # modals/aux we don't number-inflect here (do/does handled below)
        if low in ("do", "does"):
            want = "do" if plural else "does"
            return _match_case(verb, want)
        return None
    lemm = _lemm()
    from lemminflect import getLemma, getInflection
    lemmas = getLemma(low, upos="VERB")
    if not lemmas:
        return None
    lemma = lemmas[0]
    # Only adjust if the verb is currently a PRESENT finite form. Detect by
    # checking whether the surface equals the VBZ or VBP inflection; if it
    # looks past (VBD) or -ing/-ed participle, leave tense alone.
    vbz = (getInflection(lemma, "VBZ") or (None,))[0]
    vbp = (getInflection(lemma, "VBP") or (None,))[0]
    vbd = (getInflection(lemma, "VBD") or (None,))[0]
    # If it's clearly past tense, don't touch (past agrees with any subject).
    if vbd and low == vbd.lower() and low != (vbp or "").lower():
        return None
    # If it's an -ing participle ("crushing"), convert to the finite present
    # that agrees — this is the "Google crushing" -> "Google crushes" case.
    want = vbz if not plural else vbp
    if not want:
        return None
    return _match_case(verb, want)


_VBP_PRONOUN_SUBJECTS = {"i", "you", "we", "they"}

def _effective_plural(subject, plural):
    """B1: I/you/we/they take the bare VBP verb form even when singular
    ("I explore", not "I explores"). The agreement engine only knows
    singular-vs-plural, so route these pronoun subjects through the plural
    branch (which selects VBP). He/she/it keep their real number."""
    first = (subject or "").strip().split()
    head = first[0].lower().strip(".,:;!?'\"()") if first else ""
    if head in _VBP_PRONOUN_SUBJECTS:
        return True
    return plural


def _fix_verb_agreement(headline, predicate_verb, plural):
    """Replace the predicate's pivoting verb in the headline with a form that
    agrees with the NEW subject's number. We know the exact verb token from the
    splitter (pred_rec['verb']), so we target it precisely rather than guessing.
    Past tense is preserved; only present/participle finite verbs are flipped."""
    if not predicate_verb:
        return headline
    # The predicate verb may be multi-word ("never knew", "is affecting");
    # operate on the LAST word, which carries the finite inflection... except
    # for copulas where the first word is the finite one. Handle the common
    # single-word case robustly; for "is affecting" the agreeing token is "is".
    parts = predicate_verb.split()
    target = parts[0] if parts and parts[0].lower() in (
        "is", "are", "was", "were", "has", "have", "do", "does") else parts[-1] if parts else predicate_verb

    # GUARD (Bug 1a): if the verb is governed by a modal or preceded by "to",
    # it must be BASE FORM and takes no number agreement. The bare verb token
    # alone can't reveal this — we must look at the word before it in the actual
    # headline ("should makes" -> "should make", "to saves" -> "to save").
    _MODALS = {"will", "would", "can", "could", "may", "might",
               "must", "shall", "should", "to"}
    m = _re_mod.search(r'(\b\w+\b)\s+\b' + _re_mod.escape(target) + r'\b', headline)
    if m and m.group(1).lower() in _MODALS:
        lemmas = _lemm().getLemma(target.lower(), upos="VERB")
        base = lemmas[0] if lemmas else target.lower()
        if base != target.lower():
            base = _match_case(target, base)
            pat = _re_mod.compile(r'\b' + _re_mod.escape(target) + r'\b')
            return pat.sub(base, headline, count=1)
        return headline  # already base form, nothing to do

    new = _agree_present_verb(target, plural)
    if not new or new == target:
        return headline
    # Replace the FIRST whole-word occurrence of target in the headline.
    pat = _re_mod.compile(r'\b' + _re_mod.escape(target) + r'\b')
    return pat.sub(new, headline, count=1)


def combine_pair(client, a, b):
    """PASS 2: recombine two already-split headlines. Returns list of dicts."""
    def fmt(label, p):
        return (f'{label}: subject="{p.get("subject","")}" '
                f'modal="{p.get("modal","")}" verb="{p.get("verb","")}" '
                f'rest="{p.get("rest","")}" plural={p.get("plural", False)} '
                f'negated={p.get("negated", False)} '
                f'(source {p["source"]}, original: {p["original"]})')
    user_msg = ("Recombine these two split headlines:\n\n"
                + fmt("Headline 1", a) + "\n" + fmt("Headline 2", b))
    recs = _call_json(client, COMBINE_RULES, user_msg, max_tokens=1000)
    # Python guard: drop any result that failed to cross — i.e. where subject
    # and predicate came from the same original headline, or where the output
    # merely reproduces one of the two input headlines.
    originals = {a["original"].strip().lower(), b["original"].strip().lower()}

    def resolve(side_src):
        """Map the model's source label back to the true split record (a or b),
        so we attach the COMPLETE original headline + link, never the model's
        possibly-fragmentary echo. Match on the labels we actually fed it
        ('Headline 1'/'Headline 2') or on the source name as a fallback."""
        s = (side_src or "").strip().lower()
        if s in ("headline 1", "1", a["source"].strip().lower()):
            return a
        if s in ("headline 2", "2", b["source"].strip().lower()):
            return b
        return None

    clean = []
    for r in recs:
        if not isinstance(r, dict) or not r.get("headline"):
            continue
        subj_rec = resolve(r.get("subject_src"))
        pred_rec = resolve(r.get("predicate_src"))
        # must resolve to two different source headlines (a real cross)
        if subj_rec is None or pred_rec is None or subj_rec is pred_rec:
            continue
        # must not simply echo an input headline verbatim
        if r["headline"].strip().lower() in originals:
            continue
        # DETERMINISTIC VERB AGREEMENT (LemmInflect). The model is unreliable at
        # conjugating the pivoting verb to the new subject, so fix it in Python.
        # We know the predicate's exact verb token (pred_rec['verb']) and the new
        # subject's number (subj_rec['plural']). Past tense is preserved; only
        # present/participle finite verbs flip. Kills 'Google crushing',
        # 'John Stamos Shake', 'Trump Administration get'.
        r["headline"] = _fix_verb_agreement(
            r["headline"], pred_rec.get("verb", ""),
            _effective_plural(subj_rec.get("subject", ""), bool(subj_rec.get("plural", False))))
        # Authoritative source text + link, taken from harvest — not the model.
        r["subject_src"] = subj_rec["source"]
        r["subject_orig"] = subj_rec["original"]
        r["subject_link"] = subj_rec.get("link", "")
        r["predicate_src"] = pred_rec["source"]
        r["predicate_orig"] = pred_rec["original"]
        r["predicate_link"] = pred_rec.get("link", "")
        clean.append(r)
    return clean
def _assemble(subj_rec, pred_rec):
    """Deterministically build one recombined headline: the SUBJECT side from
    subj_rec crossed with the PREDICATE side (verb + rest) from pred_rec.
    No model. Parts come straight from the splitter:
      subject + [modal] + verb(+negation already attached) + rest
    Verb agreement is fixed afterward by _fix_verb_agreement, using the NEW
    subject's number. Capitalization of the first letter is normalized."""
    subject = (subj_rec.get("subject") or "").strip()
    modal = (pred_rec.get("modal") or "").strip()
    verb = (pred_rec.get("verb") or "").strip()
    rest = (pred_rec.get("rest") or "").strip()
    if not subject or not verb:
        return None
    pieces = [subject]
    if modal:
        pieces.append(modal)
    pieces.append(verb)
    if rest:
        pieces.append(rest)
    headline = " ".join(pieces)
    headline = re.sub(r"\s+n.t\b", "n't", headline)
    headline = re.sub(r"\s+([,.;:!?])", r"\1", headline)
    headline = re.sub(r"\s{2,}", " ", headline)
    # Capitalize first character, leave the rest as-is.
    if headline:
        headline = headline[0].upper() + headline[1:]
    return headline


def combine_pair_deterministic(a, b):
    """PASS 2, deterministic. Cross two split headlines BOTH ways with pure
    Python assembly — no LLM. Returns list of result dicts shaped exactly like
    the old combine_pair output, so the rest of the pipeline is unchanged."""
    originals = {a["original"].strip().lower(), b["original"].strip().lower()}

    # Two crossings: A-subject + B-predicate, and B-subject + A-predicate.
    candidates = [(a, b), (b, a)]
    clean = []
    for subj_rec, pred_rec in candidates:
        headline = _assemble(subj_rec, pred_rec)
        if not headline:
            continue
        # Reject a non-cross / verbatim echo of either input headline.
        if headline.strip().lower() in originals:
            continue
        # Deterministic verb agreement to the NEW subject's number.
        headline = _fix_verb_agreement(
            headline, pred_rec.get("verb", ""),
            _effective_plural(subj_rec.get("subject", ""), bool(subj_rec.get("plural", False))))
        r = {
            "headline": headline,
            "subject_src": subj_rec["source"],
            "subject_orig": subj_rec["original"],
            "subject_link": subj_rec.get("link", ""),
            "predicate_src": pred_rec["source"],
            "predicate_orig": pred_rec["original"],
            "predicate_link": pred_rec.get("link", ""),
            "subject": subj_rec.get("subject", ""),
        }
        clean.append(r)
    return clean
def _verb_forms(verb):
    """Pre-compute singular (VBZ) and plural (VBP) forms of the pivot verb,
    server-side, so the browser needs no conjugation logic. Negation and any
    leading words are preserved; only the finite verb token is inflected.
    Returns (sing, plur)."""
    if not verb:
        return verb, verb
    parts = verb.split()
    idx = 0 if parts and parts[0].lower() in (
        "is", "are", "was", "were", "has", "have", "do", "does") else len(parts) - 1
    target = parts[idx]
    sing = _agree_present_verb(target, False) or target
    plur = _agree_present_verb(target, True) or target

    def rebuild(newtok):
        p = list(parts)
        p[idx] = newtok
        return " ".join(p)
    return rebuild(sing), rebuild(plur)


def _base_after_modal(verb):
    """Reduce a predicate verb to the BASE form a modal requires."""
    if not verb:
        return verb
    parts = verb.split()
    idx = len(parts) - 1
    target = parts[idx]
    if len(parts) >= 2 and parts[idx - 1].lower() in ("be", "been", "being"):
        return verb
    lemmas = _lemm().getLemma(target.lower(), upos="VERB")
    base = lemmas[0] if lemmas else target.lower()
    base = _match_case(target, base)
    p = list(parts)
    p[idx] = base
    return " ".join(p)


def build_parts(splits):
    """Emit split records as two part-pools (subjects, predicates) for
    browser-side crossing. Each part carries an origin id so the JS engine can
    enforce same-origin rejection (never cross a subject with the predicate
    from its own source headline). Predicates ship BOTH verb forms."""
    subjects, predicates = [], []
    for i, s in enumerate(splits):
        if not s.get("verb") or not s.get("subject"):
            continue
        sing, plur = _verb_forms(s.get("verb", ""))
        if (s.get("modal") or "").strip():
            base = _base_after_modal(s.get("verb", ""))
            sing = plur = base
        # Predicate-pool "shows" reject: a predicate led by show/shows/showed
        # is the stranded-attribution wreck ("Video Shows say ..."). Keep the
        # record as a SUBJECT (its own "X shows Y" is fine) but do not let it
        # enter the PREDICATE pool.
        _vbase = (s.get("verb", "") or "").strip().split()
        _skip_pred = bool(_vbase) and _vbase[0].lower() in ("show", "shows", "showed")
        subjects.append({
            "id": i,
            "text": s["subject"],
            "plural": bool(s.get("plural", False)),
            "is_person": _subject_is_person(s["subject"]),
            "src": s.get("source", ""),
            "orig": s.get("original", ""),
            "link": s.get("link", ""),
        })
        if _skip_pred:
            continue
        predicates.append({
            "id": i,
            "modal": s.get("modal", ""),
            "verb_sing": sing,
            "verb_plur": plur,
            "rest": s.get("rest", ""),
            "src": s.get("source", ""),
            "orig": s.get("original", ""),
            "link": s.get("link", ""),
        })
    return subjects, predicates
def build_data():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("ERROR: ANTHROPIC_API_KEY is not set. See setup instructions.",
              file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=key)

    items = harvest()
    random.shuffle(items)

    # PASS 1: split all headlines, in batches.
    SPLIT_BATCH = 25
    splits = []
    for start in range(0, len(items), SPLIT_BATCH):
        batch = items[start:start + SPLIT_BATCH]
        splits.extend(split_batch(client, batch))
        print(f"  split batch {start // SPLIT_BATCH + 1} "
              f"({len(splits)} total)", file=sys.stderr)

    # Pair the split headlines at random.
    random.shuffle(splits)
    pairs = [(splits[i], splits[i + 1]) for i in range(0, len(splits) - 1, 2)]
    print(f"  formed {len(pairs)} pairs", file=sys.stderr)

    # PASS 2: recombine each pair.
    results = []
    for n, (a, b) in enumerate(pairs, 1):
        recs = combine_pair_deterministic(a, b)
        for rec in recs:
            if rec.get("headline"):
                if is_defamatory(rec["headline"], rec.get("subject", "")):
                    continue
                rec["category"] = "all"
                results.append(rec)
        if n % 10 == 0:
            print(f"  combined {n}/{len(pairs)} pairs "
                  f"({len(results)} headlines)", file=sys.stderr)
    subjects, predicates = build_parts(splits)
    return {
        "headlines": results,
        "subjects": subjects,
        "predicates": predicates,
        "accusation_verbs": sorted(ACCUSATION_VERBS),
        "metadata": {
            "generated_at": datetime.datetime.utcnow().isoformat(),
            "total": len(results),
            "subject_count": len(subjects),
            "predicate_count": len(predicates),
        },
    }


def render_html(data, template_path="template.html", out_path="lucknooz.html"):
    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    payload = json.dumps(data, ensure_ascii=False)
    out = template.replace("/*__LUCKNOOZ_DATA__*/", payload)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"Wrote {out_path} ({data['metadata']['total']} headlines)", file=sys.stderr)


if __name__ == "__main__":
    data = build_data()
    render_html(data)
