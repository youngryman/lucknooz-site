#!/usr/bin/env python3
"""
Feed tester — checks candidate celebrity/entertainment RSS feeds before you
add them to generate.py. Run:  python test_feeds_celeb.py
Prints, for each, how many headlines it returned and a few samples, so you
only add the live ones whose flavor you like. Watch for SHORT, name-forward
headlines -- that's what we're hunting for.
"""
import feedparser

CANDIDATES = [
    ("TMZ",                      "https://www.tmz.com/rss.xml"),
    ("Page Six",                 "https://pagesix.com/feed/"),
    ("Us Weekly",                "https://www.usmagazine.com/feed/"),
    ("E! News",                  "https://www.eonline.com/news.rss"),
    ("Just Jared",               "https://www.justjared.com/feed/"),
    ("Hollywood Reporter",       "https://www.hollywoodreporter.com/feed/"),
    ("People",                   "https://people.com/feed/"),
    ("Entertainment Weekly",     "https://ew.com/feed/"),
]

for name, url in CANDIDATES:
    print("=" * 70)
    print(f"{name}\n  {url}")
    try:
        f = feedparser.parse(url)
        n = len(f.entries)
        if n == 0:
            print("  RESULT: 0 entries -- likely dead or wrong URL. SKIP.")
        else:
            print(f"  RESULT: {n} entries -- LIVE. Sample headlines:")
            for e in f.entries[:4]:
                print("    -", e.title)
    except Exception as e:
        print(f"  RESULT: ERROR -- {e}")
print("=" * 70)
print("\nTell Claude which feeds worked and whose flavor you liked.")
