#!/usr/bin/env python3
"""
LuckNooz API test
=================
A tiny, cheap first call to confirm the recombination instruction works and
your API key is wired up correctly. Runs just four hand-picked headline pairs,
prints the results and the exact cost, and writes nothing to disk.

Run:  python test_api.py
"""

import os
import sys
import json
import anthropic

PP_MAX_WORDS = 6
SYSTEM_RULES = """You are the recombination engine for LuckNooz, a project that \
remixes real news headlines by pure principled chance.

You will be given pairs of real headlines. For each pair (Headline 1 and \
Headline 2), do the following:

1. Split each headline into a SUBJECT and a PREDICATE at the point right before \
its main (finite) verb. The subject is everything before the main verb; the \
predicate begins with the main verb.

2. Check whether each predicate ENDS in a complete prepositional phrase of %d \
words or fewer (counting the preposition itself). If so, you may treat that \
trailing prepositional phrase as a separate detachable piece.

3. Produce TWO recombined headlines by crossing the pieces between the two \
headlines: give Subject 1 the Predicate of Headline 2, and Subject 2 the \
Predicate of Headline 1. Where a trailing prepositional phrase was detached, \
you may also swap those tails between the two headlines.

4. CRITICAL RULE — change NOTHING except verb agreement. The ONLY edit you are \
permitted to make is to re-conjugate the pivoting main verb so it agrees \
grammatically with its NEW subject (tense, number, person). Every other word \
must appear EXACTLY as written in the source headlines. Do not substitute \
synonyms, do not smooth or improve the phrasing, do not add or remove words, \
do not fix awkwardness, do not invent anything. Surreal, strange, and awkward \
results are GOOD and must be preserved. You are remixing real words by chance, \
not writing new headlines.

Return ONLY a JSON array, no other text. Each element is an object with keys:
  "headline", "subject_orig", "predicate_orig".
""" % PP_MAX_WORDS

PAIRS = [
    ("Labor scraps plan to make spy agency's 9/11-era questioning powers permanent",
     "Canadian Bonds Rally After BOC Holds Rates, Cites Weak Economy"),
    ("Chinese detector edges closer to solving the mystery of neutrino mass",
     "One day after discovery, Meta pulls facial recognition code from its smart glasses"),
    ("Israeli strikes in southern Lebanon kill 17, reports say",
     "Scientists are seriously asking if bees and ChatGPT are conscious"),
    ("Trump Says Colombia Will Accept Deportees, Ending Tariff Standoff",
     "Traders Keep Bets on a Fed Hike in 2026 After CPI Data"),
]

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY is not set in this terminal.")
    print("Run:  source ~/.zshrc   then try again.")
    sys.exit(1)

lines = []
for n, (h1, h2) in enumerate(PAIRS, 1):
    lines.append(f"Pair {n}:")
    lines.append(f"  Headline 1: {h1}")
    lines.append(f"  Headline 2: {h2}")
user_msg = "Here are the headline pairs:\n\n" + "\n".join(lines)

client = anthropic.Anthropic()
resp = client.messages.create(
    model="claude-haiku-4-5", max_tokens=4000,
    system=SYSTEM_RULES, messages=[{"role": "user", "content": user_msg}],
)
text = "".join(b.text for b in resp.content if b.type == "text").strip()
if text.startswith("```"):
    text = text.split("```")[1]
    if text.startswith("json"):
        text = text[4:]
    text = text.strip()

try:
    data = json.loads(text)
except json.JSONDecodeError:
    print("Could not parse model output. Raw response:\n")
    print(text)
    sys.exit(1)

print(f"\n=== {len(data)} recombined headlines ===\n")
for d in data:
    print(" •", d.get("headline", "(missing)"))
    if d.get("subject_orig"):
        print("     subject from:", d["subject_orig"])
    if d.get("predicate_orig"):
        print("     predicate from:", d["predicate_orig"])
    print()

cost = (resp.usage.input_tokens / 1e6) * 1.0 + (resp.usage.output_tokens / 1e6) * 5.0
print(f"Tokens in: {resp.usage.input_tokens}, out: {resp.usage.output_tokens}")
print(f"Cost of this call: ${cost:.5f}")
