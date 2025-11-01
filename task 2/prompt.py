import pandas as pd

def make_prompt(row):
    date = str(row.get("date", ""))
    likes = str(row.get("likes", ""))
    username = str(row.get("username", ""))
    media_type = str(row.get("media_type", ""))
    # Checks for "inferred company" or "inferred_company"
    company = str(row.get("inferred company", row.get("inferred_company", "")))
    caption = str(row.get("media_caption", ""))

    prompt = f"""
You are a content-generation model trained to simulate Twitter marketing posts.

Your task:
Given structured metadata input, generate realistic marketing tweet text.

Rules:
1. Marketing tone under 280 characters.
2. Align with brand identity from username and company.
3. Reference media_caption if present.
4. Do not mention likes explicitly.
5. No explanation. Output tweet text only.

Input:
<date> {date}
<likes> {likes}
<username> {username}
<company> {company}
<media> {media_type}
<media_caption> {caption}

Output (content):
"""
    return "\n".join(line.strip() for line in prompt.strip().split("\n"))

try:
    df = pd.read_csv('input file')

    df['prompt'] = df.apply(make_prompt, axis=1)

    print("DataFrame with new 'prompt' column:")
    print(df.head())

    output_filename = 'output file'
    df.to_csv(output_filename, index=False)
    
    print(f"\nSuccessfully processed file and saved results to {output_filename}")

except FileNotFoundError:
    print("Error: 'your_input_file.csv' not found.")
    print("Please make sure the file is in the same directory or provide the full path.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check your CSV file's column names. They must match what the function expects:")
    print("Expected keys: 'date', 'likes', 'username', 'media_type', 'inferred company' (or 'inferred_company')")