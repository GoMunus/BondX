import pandas as pd
import requests
from bs4 import BeautifulSoup

def scrape_moneycontrol_ratings():
    url = "https://www.moneycontrol.com/markets/bonds/ratings/"  
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")

    tables = pd.read_html(resp.text)
    ratings = tables[0]   # first table = ratings
    ratings.to_csv("data/ratings.csv", index=False)
    print("✅ Credit ratings saved to data/ratings.csv")

if __name__ == "__main__":
    scrape_moneycontrol_ratings()
