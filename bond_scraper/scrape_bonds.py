import requests
import pandas as pd

def fetch_corporate_bonds():
    session = requests.Session()

    # Step 1: Get NSE homepage to load cookies automatically
    homepage = "https://www.nseindia.com"
    session.get(homepage, headers={
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36"
    })

    # Step 2: Request the corporate bonds JSON
    url = "https://www.nseindia.com/json/corporate-bonds.json"
    headers = {
        "accept": "application/json, text/plain, */*",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36",
        "referer": "https://www.nseindia.com/market-data/debt-market-reporting-corporate-bonds-traded-on-exchange",
    }

    resp = session.get(url, headers=headers)
    print("Status:", resp.status_code)

    if resp.status_code != 200:
        print("❌ Failed to fetch bonds JSON")
        return None

    data = resp.json()

    # Step 3: Parse JSON → DataFrame
    # NSE JSON format typically: {"data": [...list of bonds...]}
    bonds = data.get("data") or data
    df = pd.DataFrame(bonds)

    # Step 4: Save to CSV
    output_file = "nse_corporate_bonds.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df)} records to {output_file}")

    return df

if __name__ == "__main__":
    df = fetch_corporate_bonds()
    if df is not None:
        print(df.head())
