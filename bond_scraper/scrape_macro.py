import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

def scrape_rbi_liquidity():
    url = "https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx?prid=54182"  # sample URL
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # Find the first table in the page
        table = soup.find("table")
        if table is None:
            raise ValueError("No <table> found on the RBI page")

        # Convert table to DataFrame
        df = pd.read_html(str(table))[0]

        # Ensure "data" folder exists
        os.makedirs("data", exist_ok=True)

        # Save properly in vertical CSV
        df.to_csv("data/rbi_liquidity.csv", index=False)
        print("✅ RBI liquidity data saved to data/rbi_liquidity.csv")

    except Exception as e:
        print("❌ RBI liquidity scraping failed:", e)
        # Example: turn your list of strings into a dataframe
lines = [
    "Date : Aug 10, 2022",
    "Money Market Operations as on August 08, 2022",
    "(Amount in ₹ crore, Rate in Per cent)",
    "MONEY MARKETS, Volume, Weighted Average Rate, Range",
    "Overnight Segment, 509610.50, 5.12, 3.50-7.10",
    "Call Money, 12608.51, 5.11, 3.50-5.30",
]

# Split by commas to form proper CSV rows
rows = [line.split(",") for line in lines]

df = pd.DataFrame(rows)
df.to_csv("data/rbi_liquidity.csv", index=False)

if __name__ == "__main__":
    scrape_rbi_liquidity()
