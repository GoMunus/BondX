from pygooglenews import GoogleNews
import pandas as pd

def scrape_news():
    gn = GoogleNews(lang='en', country='IN')
    search = gn.search('corporate bonds India')

    news_data = []
    for entry in search['entries'][:10]:
        news_data.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published
        })

    df = pd.DataFrame(news_data)
    df.to_csv("data/news.csv", index=False)
    print("✅ News saved to data/news.csv")

if __name__ == "__main__":
    scrape_news()
