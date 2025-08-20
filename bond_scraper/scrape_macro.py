import pandas as pd

def scrape_rbi_yields():
    url = "https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"  
    try:
        tables = pd.read_html(url)
        macro = tables[0]
        macro.to_csv("data/macro.csv", index=False)
        print("✅ RBI macro data saved to data/macro.csv")
    except Exception as e:
        print("❌ RBI macro scraping failed:", e)

if __name__ == "__main__":
    scrape_rbi_yields()
