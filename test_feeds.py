#!/usr/bin/env python3
"""
Feed tester — checks candidate foreign English-language RSS feeds before you
add them to generate.py. Run:  python test_feeds.py
Prints, for each, how many headlines it returned and a few samples, so you
only add the live ones whose flavor you like.
"""
import feedparser

CANDIDATES = [
    # Asia-Pacific
    ("Straits Times (Singapore)", "https://www.straitstimes.com/news/world/rss.xml"),
    ("ABC News (Australia)",       "https://www.abc.net.au/news/feed/2942460/rss.xml"),
    ("Times of India",            "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"),
    ("Japan Times",               "https://www.japantimes.co.jp/feed/"),
    # Europe
    ("Irish Times",               "https://www.irishtimes.com/cmlink/news-1.1319192"),
    ("Deutsche Welle (English)",  "https://rss.dw.com/rdf/rss-en-all"),
    ("France 24 (English)",       "https://www.france24.com/en/rss"),
    # Middle East
    ("Times of Israel",           "https://www.timesofisrael.com/feed/"),
    ("Al Jazeera English",        "https://www.aljazeera.com/xml/rss/all.xml"),
    # Africa
    ("Guardian Nigeria",          "https://guardian.ng/feed/"),
    ("Daily Maverick (S. Africa)","https://www.dailymaverick.co.za/feed/"),
    # Americas
    ("CBC (Canada)",              "https://www.cbc.ca/webfeed/rss/rss-world"),
    ("Tico Times (Costa Rica)",   "https://ticotimes.net/feed"),
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
