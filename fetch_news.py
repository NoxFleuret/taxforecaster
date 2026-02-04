import requests
import pandas as pd
import datetime
import xml.etree.ElementTree as ET

def get_latest_financial_news():
    """
    Fetches latest financial news from reliable RSS feeds (CNBC Indonesia, Antara, etc.).
    Uses standard library XML parsing to avoid external dependencies like feedparser.
    Returns a list of dictionaries: [{'title':, 'link':, 'published':, 'source':}]
    """
    rss_urls = [
        # --- LOCAL (INDONESIA) ---
        {"source": "CNBC Indonesia", "url": "https://www.cnbcindonesia.com/market/rss", "category": "Local 🇮🇩"},
        {"source": "Antara Ekonomi", "url": "https://www.antaranews.com/rss/ekonomi-finansial", "category": "Local 🇮🇩"},
        {"source": "Kontan", "url": "https://feeds.feedburner.com/kontan/keuangan", "category": "Local 🇮🇩"},
        {"source": "Bisnis.com", "url": "https://market.bisnis.com/rss", "category": "Local 🇮🇩"},
        {"source": "Investor.id", "url": "https://investor.id/rss", "category": "Local 🇮🇩"},
        {"source": "CNN Indonesia", "url": "https://www.cnnindonesia.com/ekonomi/rss", "category": "Local 🇮🇩"},
        {"source": "Tempo Bisnis", "url": "http://rss.tempo.co/bisnis", "category": "Local 🇮🇩"},
        {"source": "Republika Ekonomi", "url": "https://www.republika.co.id/rss/ekonomi/", "category": "Local 🇮🇩"},
        {"source": "Okezone Economy", "url": "https://www.okezone.com/rss/economy", "category": "Local 🇮🇩"},
        {"source": "Sindonews Ekbis", "url": "https://ekbis.sindonews.com/rss", "category": "Local 🇮🇩"},
        {"source": "Kompas Money", "url": "https://money.kompas.com/feed", "category": "Local 🇮🇩"},
        {"source": "Liputan6 Bisnis", "url": "https://feed.liputan6.com/bisnis", "category": "Local 🇮🇩"},
        {"source": "BeritaSatu Ekonomi", "url": "https://www.beritasatu.com/xml/economy", "category": "Local 🇮🇩"},
        {"source": "Tribun Bisnis", "url": "https://www.tribunnews.com/rss/bisnis", "category": "Local 🇮🇩"},
        {"source": "Katadata", "url": "https://katadata.co.id/feed", "category": "Local 🇮🇩"},

        # --- GLOBAL ---
        {"source": "CNBC World", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", "category": "Global 🌎"},
        {"source": "Reuters Business", "url": "https://feeds.reuters.com/reuters/businessNews", "category": "Global 🌎"},
        {"source": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rssindex", "category": "Global 🌎"},
        {"source": "MarketWatch", "url": "http://feeds.marketwatch.com/marketwatch/topstories/", "category": "Global 🌎"},
        {"source": "BBC Business", "url": "http://feeds.bbci.co.uk/news/business/rss.xml", "category": "Global 🌎"},
        {"source": "Investing.com", "url": "https://www.investing.com/rss/news.rss", "category": "Global 🌎"},
        {"source": "NYT Business", "url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "category": "Global 🌎"},
        {"source": "The Guardian", "url": "https://www.theguardian.com/uk/business/rss", "category": "Global 🌎"},
        {"source": "CNN Money", "url": "http://rss.cnn.com/rss/money_topstories.rss", "category": "Global 🌎"},
        {"source": "WSJ Markets", "url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "category": "Global 🌎"},
        {"source": "The Economist", "url": "https://www.economist.com/finance-and-economics/rss.xml", "category": "Global 🌎"},
        {"source": "Forbes Business", "url": "https://www.forbes.com/business/feed/", "category": "Global 🌎"},
        {"source": "Business Insider", "url": "https://feeds.businessinsider.com/custom/all", "category": "Global 🌎"},
        {"source": "Financial Times", "url": "https://www.ft.com/rss/home", "category": "Global 🌎"},
        {"source": "Quartz", "url": "https://qz.com/rss", "category": "Global 🌎"}
    ]
    
    news_items = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    all_sources_data = []
    
    for feed in rss_urls:
        feed_items = []
        try:
            response = requests.get(feed['url'], headers=headers, timeout=5)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                
                # Iterate over items (standard RSS 2.0)
                count = 0
                for item in root.findall('.//item'):
                    if count >= 3: break
                    
                    title = item.find('title').text if item.find('title') is not None else "No Title"
                    link = item.find('link').text if item.find('link') is not None else "#"
                    pub_date_raw = item.find('pubDate').text if item.find('pubDate') is not None else ""
                    
                    try:
                        parts = pub_date_raw.split(' ')
                        if len(parts) >= 5:
                            pub_date = f"{parts[1]} {parts[2]} {parts[4][:5]}"
                        else:
                            pub_date = pub_date_raw[:16]
                    except:
                        pub_date = "Recently"
                    
                    feed_items.append({
                        'title': title,
                        'link': link,
                        'published': pub_date,
                        'source': feed['source'],
                        'category': feed['category']
                    })
                    count += 1
        except Exception as e:
            print(f"Error fetching {feed['source']}: {e}")
        
        if feed_items:
            all_sources_data.append(feed_items)
            
    # Interleave items: Round Robin selection
    news_items = []
    max_items_per_source = 3
    for i in range(max_items_per_source):
        for src_group in all_sources_data:
            if i < len(src_group):
                news_items.append(src_group[i])
            
    return news_items[:140]

if __name__ == "__main__":
    news = get_latest_financial_news()
    print(f"Fetched {len(news)} items.")
    for n in news:
        print(f"- [{n['source']}] ({n['published']}) {n['title']}")
