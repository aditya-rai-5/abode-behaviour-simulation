import pandas as pd

def make_prompt(row):
    """
    Build a text-to-text training prompt for Task-2 (content simulation).
    Input: row is a dict or pandas Series with keys:
      - date
      - likes
      - username
      - media
      - inferred_company
      - media_caption (optional)

    Output: string prompt
    """

    date = str(row.get("date", ""))
    likes = str(row.get("likes", ""))
    username = str(row.get("username", ""))
    media_type = str(row.get("media_type", ""))
    # Checks for "inferred company" or "inferred_company"
    company = str(row.get("inferred company", row.get("inferred_company", "")))
    # caption = str(row.get("media_caption", ""))

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

Output (content):
"""
# <caption> {caption}

    # Cleans up leading whitespace from the f-string
    return "\n".join(line.strip() for line in prompt.strip().split("\n"))

# --- Main script to iterate over a CSV ---

# 1. Load your CSV file
# Replace 'your_input_file.csv' with the actual path to your file
try:
    df = pd.read_csv('train_dataset.csv')

    # 2. Apply the function to every row
    # axis=1 tells pandas to apply the function row-by-row
    # Your `make_prompt` function receives each row as a pandas Series
    df['prompt'] = df.apply(make_prompt, axis=1)

    # 3. (Optional) Display the first 5 rows to check the new 'prompt' column
    print("DataFrame with new 'prompt' column:")
    print(df.head())

    # 4. (Optional) Save the result to a new CSV file
    # index=False avoids saving the pandas row index as an extra column
    output_filename = 'output_with_prompts.csv'
    df.to_csv(output_filename, index=False)
    
    print(f"\nSuccessfully processed file and saved results to {output_filename}")

except FileNotFoundError:
    print("Error: 'your_input_file.csv' not found.")
    print("Please make sure the file is in the same directory or provide the full path.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check your CSV file's column names. They must match what the function expects:")
    print("Expected keys: 'date', 'likes', 'username', 'media_type', 'inferred company' (or 'inferred_company')")