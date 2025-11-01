import pandas as pd
import numpy as np
import re

df = pd.read_csv("problem_1_test_dataset\content_simulation_test_company.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# df = df.dropna(subset=['content'])

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"<hyperlink>|<mention>", "", text)

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)

    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

df['content'] = df['content'].apply(clean_text)

def extract_media_url(media):
    match = re.search(r"(https://[^'\)]+)", str(media))
    return match.group(1) if match else "no_media"

df['media_url'] = df['media'].apply(extract_media_url)
df['media_type'] = df['media'].apply(lambda x: "video" if "Video" in str(x) else "photo")

df = df.rename(columns={'inferred company': 'inferred_company'})
df['inferred_company'] = df['inferred_company'].str.lower().str.strip()
df['username'] = df['username'].str.lower()

df = df[[
    'date', 'likes', 'content', 'username', 'inferred_company',
    'media_url', 'media_type'
]]

df.to_csv("test_dataset.csv", index=False)
print("complete")