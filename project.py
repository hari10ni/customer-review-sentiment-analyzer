"""
text_cleaner.py
---------------
Data Prep Module — Customer Review Sentiment Analyzer
Author : Harini
Task   : Read review files (CSV / JSON / TXT) and clean the text
         Output : cleaned CSV ready for Subash Kumar's sentiment engine
"""

import re
import json
import csv
import os
import unicodedata

# ── Optional: install with  pip install pandas emoji  ──────────────────────
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import emoji
    EMOJI_LIB_AVAILABLE = True
except ImportError:
    EMOJI_LIB_AVAILABLE = False
# ───────────────────────────────────────────────────────────────────────────


# ── 1. TEXT CLEANING ───────────────────────────────────────────────────────

def remove_emojis(text: str) -> str:
    """Remove all emoji characters from text."""
    if EMOJI_LIB_AVAILABLE:
        return emoji.replace_emoji(text, replace="")

    # Fallback: unicode range-based removal
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"   # emoticons
        "\U0001F300-\U0001F5FF"   # symbols & pictographs
        "\U0001F680-\U0001F6FF"   # transport & map
        "\U0001F1E0-\U0001F1FF"   # flags
        "\U00002500-\U00002BEF"   # chinese char / misc
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for a single review string.

    Steps:
        1. Convert to string (handles NaN / None)
        2. Remove emojis
        3. Remove URLs
        4. Remove HTML tags
        5. Remove special characters & punctuation (keep letters, digits, spaces)
        6. Convert to lowercase
        7. Collapse extra whitespace
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = remove_emojis(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize unicode (e.g. accented chars → plain ASCII where possible)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Keep only letters, digits and spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Lowercase
    text = text.lower()

    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ── 2. FILE READERS ────────────────────────────────────────────────────────

def read_txt(filepath: str) -> list[dict]:
    """Read a plain-text file — one review per line."""
    reviews = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                reviews.append({"id": i, "raw_review": line})
    return reviews


def read_csv(filepath: str, review_column: str = None) -> list[dict]:
    """
    Read a CSV file.
    If review_column is given, that column is used.
    Otherwise the first column that contains text is auto-detected.
    """
    if PANDAS_AVAILABLE:
        df = pd.read_csv(filepath, encoding="utf-8", on_bad_lines="skip")
        if review_column and review_column in df.columns:
            col = review_column
        else:
            # Pick the first object/string column
            col = df.select_dtypes(include="object").columns[0]
        print(f"  [CSV] Using column: '{col}'")
        records = [
            {"id": i + 1, "raw_review": str(row)}
            for i, row in enumerate(df[col].dropna())
        ]
        return records

    # Fallback without pandas
    records = []
    with open(filepath, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        col = review_column or reader.fieldnames[0]
        for i, row in enumerate(reader, start=1):
            val = row.get(col, "").strip()
            if val:
                records.append({"id": i, "raw_review": val})
    return records


def read_json(filepath: str, review_key: str = "review") -> list[dict]:
    """
    Read a JSON file.
    Supports:
      - List of strings          : ["good product", "bad quality", ...]
      - List of objects          : [{"review": "...", ...}, ...]
      - Dict with a list inside  : {"reviews": [...]}
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    # Unwrap if the JSON is a dict containing a list
    if isinstance(data, dict):
        for key in (review_key, "reviews", "data", "results"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    records = []
    for i, item in enumerate(data, start=1):
        if isinstance(item, str):
            records.append({"id": i, "raw_review": item})
        elif isinstance(item, dict):
            text = item.get(review_key) or item.get("text") or item.get("comment") or ""
            records.append({"id": i, "raw_review": str(text)})
    return records


# ── 3. MAIN PIPELINE ───────────────────────────────────────────────────────

def load_reviews(filepath: str, **kwargs) -> list[dict]:
    """Dispatch to the right reader based on file extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        return read_txt(filepath)
    elif ext == ".csv":
        return read_csv(filepath, review_column=kwargs.get("review_column"))
    elif ext == ".json":
        return read_json(filepath, review_key=kwargs.get("review_key", "review"))
    else:
        raise ValueError(f"Unsupported file type: {ext}  (supported: .txt, .csv, .json)")


def process_file(input_path: str, output_path: str = None, **kwargs) -> str:
    """
    Full pipeline:
        Read  →  Clean  →  Save cleaned CSV

    Returns the path to the output CSV.
    """
    print(f"\n📂 Reading : {input_path}")
    reviews = load_reviews(input_path, **kwargs)
    print(f"   Found   : {len(reviews)} reviews")

    # Clean each review
    for record in reviews:
        record["cleaned_review"] = clean_text(record["raw_review"])

    # Determine output path
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + "_cleaned.csv"

    # Write output CSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "raw_review", "cleaned_review"])
        writer.writeheader()
        writer.writerows(reviews)

    print(f"✅ Saved   : {output_path}")
    return output_path


# ── 4. QUICK DEMO (run this file directly to test) ─────────────────────────

if __name__ == "__main__":

    # ── Demo 1: TXT ──────────────────────────────────────────────────────
    demo_txt = "sample_reviews.txt"
    with open(demo_txt, "w", encoding="utf-8") as f:
        f.write("Great product! 😍 Loved it so much!!!\n")
        f.write("Worst purchase ever 😡 Total waste of money...\n")
        f.write("It's okay, nothing special. Visit https://example.com for more.\n")
        f.write("<b>Highly recommend</b> to everyone!!!\n")

    process_file(demo_txt)

    # ── Demo 2: CSV ──────────────────────────────────────────────────────
    demo_csv = "sample_reviews.csv"
    with open(demo_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["review", "rating"])
        writer.writerow(["Absolutely love this 🥰🥰", 5])
        writer.writerow(["Terrible!!! Do NOT buy 👎👎", 1])
        writer.writerow(["Average product, does what it says.", 3])

    process_file(demo_csv, review_column="review")

    # ── Demo 3: JSON ─────────────────────────────────────────────────────
    demo_json = "sample_reviews.json"
    with open(demo_json, "w", encoding="utf-8") as f:
        json.dump([
            {"review": "Super fast delivery 🚀, will order again!"},
            {"review": "Packaging was damaged 😤 Not happy at all."},
            {"review": "Good value for money."},
        ], f)

    process_file(demo_json)

    print("\n✨ All demo files processed! Check the *_cleaned.csv output files.")