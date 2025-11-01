import pandas as pd
import numpy as np
import re
import sys

df = pd.read_csv("input file", nrows=1000)

df['date'] = pd.to_datetime(df['date'], errors='coerce')

def extract_media_url(media):
    # Search for a URL pattern
    match = re.search(r"(https://[^'\)]+)", str(media))
    return match.group(1) if match else "no_media"

df['media_url'] = df['media'].apply(extract_media_url)
df['media_type'] = df['media'].apply(lambda x: "video" if "Video" in str(x) else "photo")

df = df.rename(columns={'inferred company': 'inferred_company'})
df['inferred_company'] = df['inferred_company']
df['username'] = df['username'].str.lower()

final_columns = [
    'date', 'likes', 'username', 'inferred_company',
    'media_url','content', 'media_type'
]
df_final = df[final_columns]

output_file = "output file"
df_final.to_csv(output_file, index=False)

print(f"Cleaned data saved to: {output_file}")

data = pd.read_csv(output_file)
data.info()