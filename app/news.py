from pygooglenews import GoogleNews
from newspaper import Article

def get_company_news(query: str, limit: int = 5):
    """
    Fetches company-related news articles using Google News and newspaper3k.
    
    Args:
        query (str): Search query (usually company name + ticker).
        limit (int): Max number of articles to return.

    Returns:
        list[dict]: Each dict has title, publisher, link, published, summary, top_image.
    """
    gn = GoogleNews(lang="en", country="US")
    res = gn.search(query) or {}
    items = []
    for e in (res.get("entries") or [])[:limit]:
        try:
            art = Article(e.link)
            art.download()
            art.parse()
            art.nlp()
            items.append({
                "title": e.title,
                "publisher": e.source.get("title") if hasattr(e, "source") else None,
                "link": e.link,
                "published": e.published,
                "summary": art.summary,
                "top_image": art.top_image,
            })
        except Exception:
            items.append({
                "title": getattr(e, "title", query),
                "publisher": None,
                "link": getattr(e, "link", ""),
                "published": getattr(e, "published", ""),
                "summary": None,
                "top_image": None,
            })
    return items

    