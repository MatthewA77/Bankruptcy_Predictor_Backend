# news.py
import feedparser
import urllib.parse
from datetime import datetime
from typing import List, Dict

def _fmt_published(entry) -> str:
    # feedparser returns 'published' or 'updated' depending on feed
    ts = entry.get("published") or entry.get("updated") or ""
    return ts

def get_company_news(query: str, *, limit: int = 10) -> List[Dict]:
    """
    Fetch top recent news from Google News RSS for a given company name/ticker.
    Returns a list of dicts with: title, link, publisher, published, summary.
    """
    q = urllib.parse.quote(query)
    # language & region can be adjusted; here we use English/US
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)

    items = []
    for entry in feed.entries[:limit]:
        items.append({
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "publisher": getattr(entry, "source", {}).get("title", "") if hasattr(entry, "source") else "",
            "published": _fmt_published(entry),
            "summary": entry.get("summary", ""),  # often short; safe to show or omit
        })
    return items
