import requests
from bs4 import BeautifulSoup
import pandas as pd

df = pd.read_csv("data/processed/ncsl_2025_scraped.csv")

headers = {"User-Agent": "Mozilla/5.0"}

for i, row in df.head(5).iterrows():
    response = requests.get(row["bill_url"], headers=headers, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    print(f"{row['bill_id']}: {len(text)} characters")