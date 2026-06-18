#!/usr/bin/env python3
"""
Deterministic headline splitter for LuckNooz / UNS
==================================================
Uses spaCy's dependency parse to split a news headline into

    SUBJECT  |  MODAL  |  VERB  |  REST

with a NEGATED flag and a PLURAL flag, doing the mechanical grammar work the
language model was previously improvising (and getting wrong). The model now
only RECOMBINES; it no longer hunts for verbs, invents copulas, or drops
negations.

Output schema is identical to the old LLM split pass, so combine_pair() and the
rest of generate.py need no changes:

    {"subject": str, "modal": str, "verb": str, "rest": str,
     "negated": bool, "plural": bool}

Design decisions, each tied to an observed failure:
  * ROOT-verb anchored. We find the clause ROOT via the parse, not "the first
    verb-looking token", so "trade rumors gain momentum" splits at "gain", and
    "SpaceX Shares Close higher" splits at "Close" — the noun pileup stays in
    the subject. (fixes: noun-as-verb)
  * Negation preserved. If the ROOT has a `neg` child ("never", "isn't", "not")
    the NEGATED flag is set and the negation word is kept with the verb, so
    "Goethe never knew ..." can never become "Goethe knew ...". (fixes: dropped
    negation)
  * Colon handling is deterministic: text before a single leading "label:" colon
    is dropped ONCE (headlinese "India news: X" -> "X"); everything else is kept
    verbatim. (fixes: inconsistent colon behavior)
  * Subject is the ROOT's actual subject span (nsubj / nsubjpass) plus its
    subtree, so we never strand the real doer. If the ROOT has no subject, the
    headline is verb-first (subject="") and usable only as a predicate.
  * We never INVENT words. spaCy only labels; it never rewrites. Recombination/
    agreement remains the model's single permitted change.
"""

import re

_NLP = None

# Quality cutoff, set from the subject-length histogram across Daniel's own
# feeds: ~90-95% of good splits have subjects of 5 words or fewer. A subject
# longer than this almost always means spaCy grabbed a late/wrong verb and swept
# half the headline into the subject (the "misattachment" failure). Splits whose
# subject exceeds this are treated as VERBLESS (set aside), not mis-crossed.
# Raise it to admit more (and risk more garbage); lower it to be stricter.
MAX_SUBJECT_WORDS = 5


def _load():
    global _NLP
    if _NLP is None:
        import spacy
        # parser + tagger needed; NER/lemmatizer not required for the split
        _NLP = spacy.load("en_core_web_sm", disable=["ner"])
    return _NLP


# A single leading "Label: rest" colon in headlinese (e.g. "India news:",
# "Review:", "World Cup 2026:"). We strip the label ONCE. We do NOT strip when
# the colon is deep in the line or when the prefix is long (likely a real clause).
_LEADING_LABEL = re.compile(r'^\s*([^:]{1,28}):\s+(?=\S)')

# A trailing attribution clause in headlinese: "..., source says",
# "..., Court Docs Say", "..., ECB's Nagel Says", "..., expert says",
# "..., police say". These confuse the parser into treating the attribution
# verb as the clause root and stranding the real subject. Strip ONCE from the
# end, before parsing. Matches a comma, then up to ~5 words, ending in
# say/says/said/reports/reported/confirms/confirmed/announces/announced.
_TRAILING_ATTR = re.compile(
    r',\s+[^,]{0,60}?\b'
    r'(?:say|says|said|reports?|reported|confirms?|confirmed|'
    r'announces?|announced|claims?|claimed|warns?|warned)\s*$',
    re.IGNORECASE,
)


def _strip_trailing_attr(text):
    """Remove a trailing ', X says'-style attribution once."""
    m = _TRAILING_ATTR.search(text)
    if not m:
        return text, ""
    head = text[:m.start()].rstrip()
    # only strip if something substantial remains
    if len(head.split()) < 3:
        return text, ""
    return head, text[m.start():]


# Headlines that splice two clauses with an em-dash, spaced hyphen, or
# semicolon: "My background is affecting my studies — what do I do?",
# "Settler shoots Palestinian; reports say he was disabled". The FIRST clause
# is the real headline; the tail is commentary/attribution. Keep clause 1.
_CLAUSE_SPLIT = re.compile(r'\s+(?:[\u2014\u2013]|--|-)\s+|;\s+')


def _keep_first_clause(text):
    """If the headline splices clauses with an em/en-dash, spaced hyphen, or
    semicolon, keep the first clause when it is itself substantial. This stops
    the parser from rooting on a trailing fragment like 'what do I do?'."""
    parts = _CLAUSE_SPLIT.split(text, maxsplit=1)
    if len(parts) == 2:
        first = parts[0].strip()
        # keep clause 1 only if it has enough to be a headline on its own
        if len(first.split()) >= 4:
            return first
    return text


def _strip_leading_label(text):
    """Remove a short 'Label: ' prefix once; return (clean_text, dropped_label)."""
    m = _LEADING_LABEL.match(text)
    if not m:
        return text, ""
    label = m.group(1)
    # Only strip if the label has no internal sentence punctuation and the
    # remainder is substantial (avoid eating real content like "5:30").
    if any(c in label for c in ".?!"):
        return text, ""
    remainder = text[m.end():]
    if len(remainder.split()) < 3:
        return text, ""
    return remainder, label


def _is_plural(tok):
    """Plural if the token is tagged plural, or its subtree carries plural cues."""
    if tok.tag_ in ("NNS", "NNPS"):
        return True
    if tok.lemma_.lower() in ("they", "we", "those", "these"):
        return True
    # coordinated subject ("Japan and Canada") reads plural
    for child in tok.children:
        if child.dep_ == "conj":
            return True
        if child.dep_ == "cc" and child.lemma_.lower() == "and":
            return True
    return False


def split_headline(text):
    """Return the split dict for one headline, or a verbless/empty fallback."""
    nlp = _load()
    clean, _label = _strip_leading_label(text.strip())
    clean, _attr = _strip_trailing_attr(clean)
    clean = _keep_first_clause(clean)
    doc = nlp(clean)

    # Find the ROOT; prefer a verbal root, else the highest verb in the parse.
    root = None
    for tok in doc:
        if tok.dep_ == "ROOT":
            root = tok
            break
    if root is None:
        return _verbless(clean)

    # TRAILING-ATTRIBUTION GUARD. Headlinese often ends "..., expert says" or
    # "..., sources say" or "Court Docs Say". spaCy frequently makes that
    # attribution verb the ROOT, which strands the real clause. If the ROOT is
    # a say/said/says-type verb sitting in the last few tokens AND there is a
    # comma before it, re-anchor on the main clause's verb (a ccomp/advcl/dep
    # child that is itself a VERB/AUX), which carries the real subject.
    ATTRIB = {"say", "says", "said", "report", "reports", "reported",
              "add", "adds", "added"}
    if root.lemma_.lower() in ATTRIB and root.i >= len(doc) - 3:
        has_comma_before = any(t.text == "," and t.i < root.i for t in doc)
        if has_comma_before:
            inner = None
            for c in root.children:
                if c.dep_ in ("ccomp", "advcl", "dep", "parataxis") and \
                   c.pos_ in ("VERB", "AUX"):
                    inner = c
                    break
            if inner is not None:
                root = inner

    # EARLIEST-SUBJECT PREFERENCE. Headlines lead with the real subject, so the
    # true main verb is usually the EARLIEST finite verb that has a subject to
    # its left. If spaCy rooted on a later verb (e.g. "Japan ... is done" rooting
    # on "done"), look for an earlier finite verb (VERB/AUX) that owns an nsubj
    # starting near the front, and prefer it. This keeps the subject short and
    # the verb early, matching headline grammar.
    def _has_left_subject(v):
        return any(c.dep_ in ("nsubj", "nsubjpass", "csubj") and c.i < v.i
                   for c in v.children)

    finite = [t for t in doc
              if t.pos_ in ("VERB", "AUX") and _has_left_subject(t)]
    if finite:
        earliest = min(finite, key=lambda t: t.i)
        # only override if the current root's subject starts later than the
        # earliest candidate's subject (i.e. we'd otherwise strand the front)
        def _subj_start(v):
            subs = [c.i for c in v.children
                    if c.dep_ in ("nsubj", "nsubjpass", "csubj")]
            return min(subs) if subs else 10_000
        if _subj_start(earliest) < _subj_start(root):
            root = earliest

    # If the ROOT is not a verb (e.g. a copula construction tags the adjective
    # as ROOT, or a nominal headline), try to locate the finite verb.
    verb = root
    if root.pos_ not in ("VERB", "AUX"):
        # copular: "X is fake" -> ROOT "fake" (ADJ) with a `cop` child "is"
        cop = next((c for c in root.children if c.dep_ == "cop"), None)
        if cop is not None:
            verb = cop
        else:
            # nominal/verbless headline (no finite verb)
            aux = next((c for c in root.children if c.dep_ in ("aux", "auxpass")), None)
            if aux is None:
                return _verbless(clean)
            verb = aux

    # Subject span: the verb's (or its head's) nominal subject subtree.
    subj_tok = _find_subject(verb, root)
    if subj_tok is None:
        # verb-first headline: no subject. Usable only as a predicate.
        subject = ""
    else:
        subject = _span_text(subj_tok)

    # Modal: a modal auxiliary governing the verb ("may", "could", "will"...).
    modal = ""
    neg = False
    head_for_children = root if verb is root else verb.head
    for c in list(verb.children) + list(root.children):
        if c.dep_ == "aux" and c.tag_ == "MD":
            modal = c.text
        if c.dep_ == "neg":
            neg = True

    # Determine where REST begins: everything from the token after the verb
    # (and after any negation/aux that belongs with it) to the end.
    verb_text, rest = _verb_and_rest(doc, subj_tok, verb, modal, neg)

    plural = _is_plural(subj_tok) if subj_tok is not None else False

    if not verb_text:
        return _verbless(clean)

    # QUALITY CUTOFF. A subject longer than MAX_SUBJECT_WORDS almost always means
    # the parser grabbed a late/wrong verb and swept half the headline into the
    # subject. Set such splits aside as verbless rather than crossing them badly.
    clean_subject = _clean_surface(subject)
    if clean_subject and len(clean_subject.split()) > MAX_SUBJECT_WORDS:
        return _verbless(clean)

    # NO-PREDICATE GUARD (Bug A). If the verb sits at the end of the headline
    # with no real REST after it, the "subject" swallowed the whole clause and
    # there is no genuine predicate to cross. Catches "...compensation began"
    # and similar verb-at-end fragments. Set aside as verbless.
    clean_rest = _clean_surface(rest)
    if clean_subject and len(clean_rest.split()) < 1:
        return _verbless(clean)

    return {
        "subject": clean_subject,
        "modal": modal.strip(),
        "verb": _clean_surface(verb_text),
        "rest": _clean_surface(rest),
        "negated": neg,
        "plural": bool(plural),
    }


def _find_subject(verb, root):
    """Locate the nominal subject token for the clause."""
    candidates = []
    for src in (verb, root, verb.head):
        for c in src.children:
            if c.dep_ in ("nsubj", "nsubjpass", "csubj"):
                candidates.append(c)
    if not candidates:
        return None
    # earliest subject in the sentence (headlines lead with the subject)
    return sorted(candidates, key=lambda t: t.i)[0]


def _span_text(tok):
    """Contiguous subject text. Take the subtree, but keep only the unbroken
    run of tokens starting at the subtree's leftmost index — so a discontinuous
    subtree (e.g. spaCy attaching a trailing 'Court Docs Say' to the subject)
    cannot stitch across a gap and swallow unrelated words."""
    sub = sorted(tok.subtree, key=lambda t: t.i)
    if not sub:
        return tok.text
    indices = [t.i for t in sub]
    start = indices[0]
    # Walk forward while indices are contiguous; stop at the first gap.
    contiguous = [sub[0]]
    for prev, cur in zip(sub, sub[1:]):
        if cur.i == prev.i + 1:
            contiguous.append(cur)
        else:
            break
    # Ensure the subject head itself is included; if the head sits past the gap,
    # fall back to just the head's own contiguous left-to-head span.
    if tok not in contiguous:
        contiguous = [t for t in sub if t.i <= tok.i]
        # re-trim to contiguous run ending at the head
        trimmed = [tok]
        idx = tok.i
        by_i = {t.i: t for t in sub}
        while (idx - 1) in by_i:
            idx -= 1
            trimmed.insert(0, by_i[idx])
        contiguous = trimmed
    return "".join(t.text_with_ws for t in contiguous).strip()


def _verb_and_rest(doc, subj_tok, verb, modal, neg):
    """Assemble the verb (with negation word if present) and the trailing REST,
    taking everything to the right of the verb as REST verbatim."""
    # Index boundary: subject occupies the left; verb sits at verb.i.
    # Collect negation token if adjacent.
    neg_tok = None
    for c in verb.children:
        if c.dep_ == "neg":
            neg_tok = c
    # The verb phrase head index
    vi = verb.i
    # Verb text: include a preceding negation ("never knew") or contracted neg.
    verb_text = verb.text
    if neg_tok is not None:
        if neg_tok.i < vi:
            verb_text = neg_tok.text + " " + verb.text
        else:
            verb_text = verb.text + " " + neg_tok.text
    # REST = all tokens after the verb (and after a post-verb neg), excluding
    # the verb itself and excluding any modal already captured.
    rest_start = vi + 1
    if neg_tok is not None and neg_tok.i == vi + 1:
        rest_start = neg_tok.i + 1
    rest_toks = [t for t in doc[rest_start:]]
    rest = "".join(t.text_with_ws for t in rest_toks).strip()
    # Trim a trailing attribution tail ", sources say" / ", Court Docs Say" /
    # ", expert says" so it doesn't dangle in REST after re-anchoring.
    rest = re.sub(r",\s+[^,]{1,30}\b[Ss]a(?:y|ys|id)\.?$", "", rest).strip()
    return verb_text, rest


def _clean_surface(s):
    """Fix spaCy tokenization artifacts in assembled text: contraction spacing
    ('is n't' -> "isn't", 'do n't' -> "don't"), spaced possessives, and
    orphaned quote marks left by stripping a label."""
    if not s:
        return s
    # rejoin contracted negation spaCy split off: "is n't" / "do n't" / "ca n't"
    # handle both straight (') and curly (\u2019) apostrophes
    s = re.sub(r"\b([A-Za-z]+)\s+n(['\u2019])t\b", r"\1n\2t", s)
    # rejoin clitics: "Trump 's", "they 're", "we 'll", "I 've", "he 'd", "I 'm"
    s = re.sub(r"\s+(['\u2019])(s|re|ll|ve|d|m)\b", r"\1\2", s)
    # drop a single orphaned leading/trailing quote left by label-stripping
    s = s.strip()
    s = re.sub(r"^['\u2018\u2019\"]\s*", "", s)
    s = re.sub(r"\s*['\u2018\u2019\"]$", "", s)
    return s.strip()


def _verbless(text):
    return {"subject": "", "modal": "", "verb": "", "rest": text.strip(),
            "negated": False, "plural": False}


def split_items(items, report=True):
    """Split a list of harvested items, preserving source/original/link.
    Drop-in replacement for the LLM split_batch().

    When report=True, prints a subject-length histogram at the end so you can
    see the distribution of where the main verb landed (subject word-count) and
    pick a quality cutoff from your OWN corpus. A long subject usually means the
    parser grabbed a late/wrong verb."""
    out = []
    hist = {}          # subject word-count -> tally (verbal splits only)
    verbless = 0
    for it in items:
        d = split_headline(it["title"])
        d["source"] = it["source"]
        d["original"] = it["title"]
        d["link"] = it.get("link", "")
        out.append(d)
        if d["verb"]:
            n = len(d["subject"].split()) if d["subject"] else 0
            hist[n] = hist.get(n, 0) + 1
        else:
            verbless += 1

    if report:
        total = len(out)
        verbal = total - verbless
        print("\n  --- subject-length histogram (verb position) ---")
        print(f"  {total} headlines: {verbal} verbal, {verbless} verbless "
              f"({100*verbless//max(total,1)}% verbless)")
        cumulative = 0
        for n in sorted(hist):
            cumulative += hist[n]
            pct = 100 * hist[n] / max(verbal, 1)
            cum_pct = 100 * cumulative / max(verbal, 1)
            bar = "#" * int(pct / 2)
            label = "verb-first" if n == 0 else f"subj {n} word{'s' if n != 1 else ''}"
            print(f"  {label:14} {hist[n]:4}  {pct:5.1f}%  (cum {cum_pct:5.1f}%)  {bar}")
        print("  ------------------------------------------------\n")
# TAIL DUMP. Print the actual subject / verb / predicate for every verbal
    # split whose subject is long (>= TAIL_MIN words). This is the band where
    # the histogram can't tell you good from garbage — only your eye can. Read
    # these and decide where MAX_SUBJECT_WORDS really belongs.
    TAIL_MIN = 4
    tail = [
        d for d in out
        if d["verb"] and d["subject"] and len(d["subject"].split()) >= TAIL_MIN
    ]
    if tail:
        print(f"  --- tail splits (subject >= {TAIL_MIN} words): {len(tail)} ---")
        # longest subjects first, so the worst misattachments surface at the top
        tail.sort(key=lambda d: len(d["subject"].split()), reverse=True)
        for d in tail:
            n = len(d["subject"].split())
            neg = " [NEG]" if d.get("negated") else ""
            verb = d["verb"]
            pred = d.get("rest", "")
            print(f"  [{n}w]{neg} SUBJ: {d['subject']}")
            print(f"        VERB: {verb}  PRED: {pred}")
            print(f"        SRC : {d.get('original', '')}")
            print()
        print("  ------------------------------------------------\n")

    return out


if __name__ == "__main__":
    # Self-test against the specific failures Daniel catalogued.
    tests = [
        "Jordan Kyrou and New York Islanders trade rumors gain momentum ahead of NHL offseason",
        "SpaceX Shares Close 19% Higher After Historic $75 Billion IPO",
        "Prices Likely to Stay Higher Even If Conflict Ends, ECB's Nagel Says",
        "Goethe never knew this 40-million-year-old ant was hidden in his collection",
        "Bill Gates isn't happy with US govt taking stake in Intel, IBM & other US companies",
        "German students up in arms about funding cuts",
        "India news: 'Biryani date' controversy sparks debate on consent and entitlement",
        "Donald Trump's Name Has Been Removed From the Kennedy Center, Court Docs Say",
        "These FIFA World Cup ticket sites are fake",
        "Japan and Canada can do more to accelerate AI adoption, expert says",
        # --- new cases from the latest batch ---
        "My diverse academic background is affecting my PhD studies — what do I do?",
        "Trump to meet Modi at G7 in France amid claims US-Iran deal is done",
        "Hate as entertainment: Youth finding community in nihilistic online antisemitism, warns ADL",
        "Costa Rica Clears Way for \u201cMacho Coca\u201d Extradition to U.S.",
        "Settler shoots Palestinian in West Bank; reports say he was mentally disabled",
    ]
    for t in tests:
        d = split_headline(t)
        print("IN :", t)
        print("OUT:", {k: d[k] for k in ("subject", "modal", "verb", "rest", "negated", "plural")})
        print()
