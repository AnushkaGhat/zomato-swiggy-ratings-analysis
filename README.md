# Zomato vs Swiggy Ratings Analysis

## Problem Statement
Food delivery ratings are treated as an objective signal of restaurant quality —
but do they actually mean the same thing across platforms, and what really
drives them? This project analyzes Zomato restaurant data (Bangalore) to find
what factors actually correlate with rating, mines customer reviews with NLP
to find *why* restaurants get low ratings, and (phase 2) compares against
live Swiggy data for the same localities to test whether the two platforms'
ratings are even measuring the same thing.

## Key Questions
1. What actually drives a high rating — cost, cuisine, location, delivery
   options — or is it mostly noise?
2. Where do star ratings and review sentiment disagree, and what does that
   reveal about restaurant pain points?
3. Can we predict a restaurant's rating bucket from its structural features?
4. (Phase 2) Does the same restaurant get rated differently on Zomato vs
   Swiggy?

## Data Sources
- **Zomato Bangalore Restaurants** — Kaggle, Himanshu Poddar
  (https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)
- **Swiggy (Bangalore)** — self-scraped, same localities as above (phase 2)

## Project Structure
```
zomato-swiggy-analysis/
├── data/
│   ├── raw/            # original downloaded/scraped data (not committed)
│   └── processed/       # cleaned, analysis-ready data
├── notebooks/            # EDA, NLP, modeling notebooks
├── scripts/              # reusable cleaning/scraping/modeling scripts
├── dashboard/             # Power BI / Tableau files
└── outputs/               # exported charts, model results
```

## Pipeline
1. `scripts/01_clean_zomato.py` — clean raw CSV into analysis-ready format
2. `notebooks/02_eda.ipynb` — exploratory analysis
3. `notebooks/03_nlp_sentiment.ipynb` — sentiment + topic modeling on reviews
4. `notebooks/04_modeling.ipynb` — predict rating bucket, feature importance
5. `scripts/05_scrape_swiggy.py` — Swiggy data collection (phase 2)
6. `dashboard/` — interactive Power BI dashboard

## Tools Used
Python (pandas, scikit-learn, nltk/spaCy), SQL, Power BI, Selenium

## Findings
*(to be filled in as analysis progresses)*

## Author
