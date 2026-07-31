"""
01_clean_zomato.py
--------------------
Cleans the raw Zomato Bangalore Restaurants dataset (zomato.csv from Kaggle,
by Himanshu Poddar) into an analysis-ready CSV.

Usage:
    python scripts/01_clean_zomato.py

Input:
    data/raw/zomato.csv

Output:
    data/processed/zomato_clean.csv
"""

import pandas as pd
import numpy as np
import re
import ast
import os

RAW_PATH = os.path.join("data", "raw", "zomato.csv")
OUT_PATH = os.path.join("data", "processed", "zomato_clean.csv")


def load_raw(path: str) -> pd.DataFrame:
    print(f"Loading raw data from {path} ...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    # url, address, phone, menu_item add no analytical value for this project
    drop_cols = [c for c in ["url", "address", "phone", "menu_item"] if c in df.columns]
    return df.drop(columns=drop_cols)


def clean_rate(df: pd.DataFrame) -> pd.DataFrame:
    def parse_rate(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        if x in ("NEW", "-", "nan"):
            return np.nan
        x = x.split("/")[0].strip()
        try:
            return float(x)
        except ValueError:
            return np.nan

    df["rate"] = df["rate"].apply(parse_rate)
    return df


def clean_cost(df: pd.DataFrame) -> pd.DataFrame:
    cost_col = "approx_cost(for two people)"
    df[cost_col] = (
        df[cost_col]
        .astype(str)
        .str.replace(",", "", regex=False)
    )
    df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")
    df = df.rename(columns={cost_col: "cost_for_two"})
    return df


def clean_booleans(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["online_order", "book_table"]:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    return df


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    df["name"] = df["name"].astype(str).str.strip().str.title()
    df["location"] = df["location"].astype(str).str.strip()
    df["rest_type"] = df["rest_type"].astype(str).str.strip()
    df["cuisines"] = df["cuisines"].astype(str).str.strip()
    if "listed_in(type)" in df.columns:
        df = df.rename(columns={"listed_in(type)": "listing_type"})
    if "listed_in(city)" in df.columns:
        df = df.rename(columns={"listed_in(city)": "listing_city"})
    return df


def parse_reviews_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    The reviews_list column is a stringified list of tuples like:
    [('Rated 4.0', 'RATED\\n Great food...'), ...]
    We extract it into a clean list of (rating, review_text) and also
    a separate review_count and a concatenated review_text field for NLP later.
    """
    if "reviews_list" not in df.columns:
        return df

    def parse_row(raw):
        if pd.isna(raw):
            return []
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
        reviews = []
        for rate_str, text in parsed:
            rate_match = re.search(r"[\d.]+", str(rate_str)) if rate_str else None
            r = float(rate_match.group()) if rate_match else np.nan
            t = str(text).replace("RATED\n", "").strip() if text else ""
            reviews.append((r, t))
        return reviews

    parsed_reviews = df["reviews_list"].apply(parse_row)
    df["review_count_parsed"] = parsed_reviews.apply(len)
    df["review_text_combined"] = parsed_reviews.apply(
        lambda lst: " || ".join([t for _, t in lst if t])
    )
    return df


def drop_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["name", "rate", "cost_for_two"])
    df = df[df["rate"].between(1, 5)]
    df = df.drop_duplicates()
    after = len(df)
    print(f"Dropped {before - after:,} rows during cleaning ({after:,} remain)")
    return df


def deduplicate_to_restaurant_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    This dataset lists each restaurant once PER service category it offers
    (listing_type: Buffet, Cafes, Delivery, Desserts, Dine-out, Drinks &
    nightlife, Pubs and bars). A single popular restaurant can appear 5-7+
    times. For restaurant-level analysis (ratings, cost, cuisine, location),
    we collapse to one row per (name, location), keeping the row with the
    highest vote count as the most complete/current snapshot.

    The full un-deduplicated data is preserved separately (zomato_clean_full.csv)
    in case listing_type-level analysis (e.g. "how do delivery ratings compare
    to dine-out ratings for the same restaurant") is wanted later.
    """
    before = len(df)
    df_dedup = (
        df.sort_values("votes", ascending=False)
        .drop_duplicates(subset=["name", "location"], keep="first")
        .reset_index(drop=True)
    )
    after = len(df_dedup)
    print(f"Deduplicated {before:,} restaurant-listing rows -> {after:,} unique restaurants")
    return df_dedup


def main():
    df = load_raw(RAW_PATH)
    df = drop_useless_columns(df)
    df = clean_rate(df)
    df = clean_cost(df)
    df = clean_booleans(df)
    df = clean_text_fields(df)
    df = parse_reviews_list(df)
    df = drop_bad_rows(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    full_out_path = OUT_PATH.replace(".csv", "_full.csv")
    df.to_csv(full_out_path, index=False)
    print(f"Saved full listing-level data to {full_out_path}")

    df_restaurant = deduplicate_to_restaurant_level(df)
    df_restaurant.to_csv(OUT_PATH, index=False)
    print(f"Saved deduplicated restaurant-level data to {OUT_PATH}")
    print(df_restaurant.dtypes)
    print(df_restaurant.head())


if __name__ == "__main__":
    main()