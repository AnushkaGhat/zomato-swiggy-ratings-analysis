# Zomato vs Swiggy Ratings Analysis

**Live dashboard:** _add your GitHub Pages link here once set up_

## Problem Statement
Food delivery ratings are treated as an objective signal of restaurant quality —
but do they actually mean the same thing across platforms, and what really
drives them? This project analyzes Zomato restaurant data (Bangalore) to find
what factors actually correlate with rating, mines customer reviews with NLP
to find *why* restaurants get low ratings, and builds a predictive model to
identify the strongest drivers of rating, backed by evidence rather than
assumptions.

## Key Questions
1. What actually drives a high rating — cost, cuisine, location, delivery
   options — or is it mostly noise?
2. Where do star ratings and review sentiment disagree, and what does that
   reveal about restaurant pain points?
3. Can we predict a restaurant's rating bucket from its structural features?
4. (Phase 2, not yet built) Does the same restaurant get rated differently
   on Zomato vs Swiggy?

## Data Sources
- **Zomato Bangalore Restaurants** — Kaggle, Himanshu Poddar
  (https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants)
- **Swiggy (Bangalore)** — self-scraped, same localities as above (phase 2, planned)

## Project Structure
```
zomato-swiggy-analysis/
├── data/
│   ├── raw/            # original downloaded/scraped data (not committed)
│   └── processed/       # cleaned, analysis-ready data (not committed — regenerate via script)
├── notebooks/            # EDA, NLP, modeling notebooks
├── scripts/              # reusable cleaning/scraping/modeling scripts
├── dashboard/             # interactive HTML dashboard (Chart.js)
└── outputs/               # exported charts, model results
```

## Pipeline
1. `scripts/01_clean_zomato.py` — cleans raw CSV, dedupes to restaurant-level
   (raw data has one row per restaurant × service-listing-type; 41,376 raw
   listings collapse to 9,215 unique restaurants)
2. `notebooks/02_eda.ipynb` — exploratory analysis
3. `notebooks/03_nlp_sentiment.ipynb` — sentiment scoring + complaint theme extraction
4. `notebooks/04_modeling.ipynb` — predicts rating tier (Low/Medium/High), feature importance
5. `dashboard/index.html` — interactive summary dashboard of all findings

## Tools Used
Python (pandas, scikit-learn, nltk/VADER), Jupyter, Chart.js, git/GitHub

## Findings

**1. Ratings barely use the 1–5 scale.** Nearly all restaurants cluster
between 3.0 and 4.5 (mean = 3.63). Zomato ratings function less like a
precise quality measure and more like a narrow band of "good enough"
consensus.

**2. Cost has a moderate, not decisive, relationship with rating**
(correlation = 0.33). More expensive tends to trend higher, but with a lot
of noise and exceptions — price alone doesn't explain much.

**3. Table booking availability is a strong signal — but a proxy, not a
cause.** Restaurants offering table booking rate meaningfully higher on
average. This isn't booking itself driving ratings; it's a marker for a
cluster of traits (upscale, sit-down, consistent service) that tend to
travel together.

**4. Niche/international cuisines and upscale localities cluster together
at the top.** Mediterranean, European, and Asian cuisines outrank
mainstream categories. Geographically, Bangalore's premium dining corridors
(Lavelle Road, Koramangala 5th Block, Church Street) top the location
leaderboard — reinforcing the same "seriousness" pattern as table booking.

**5. The star rating hides real signal.** Review sentiment (VADER) and star
rating only loosely agree (correlation = 0.38). At every star level — even
4.5 — some reviews read as genuinely negative.

**6. Complaints center on delivery/fulfillment failures, not food taste.**
Words disproportionately common in negative reviews — *refund, deliver,
worst, received, late* — point to operational/delivery breakdowns more than
kitchen quality.

**7. A tuned model confirms review-derived signals matter most.** A
HistGradientBoosting classifier predicting rating tier (Low/Medium/High)
reaches **66.4% accuracy** on fully deduplicated, held-out data — up from an
untuned 60.4% baseline. The top features are `dish_liked_count`,
`sentiment_score`, and `review_text_length` — review- and engagement-derived
signals outweigh every structural factor (cost, cuisine, location) taken
alone.

**Data quality note:** an early version of the model showed 82.5% accuracy,
which was traced to train/test leakage — the raw dataset lists each
restaurant once per service category (delivery, dine-out, cafes, etc.), so
near-duplicate copies of the same restaurant were leaking between train and
test sets. After deduplicating to one row per restaurant, 60–66% is the
honest, trustworthy accuracy range for this feature set.

## Next Steps
- Swiggy scraping + cross-platform comparison (phase 2)
- Richer NLP (topic modeling / transformer-based sentiment) to sharpen the
  complaint-theme analysis further

## Author
Anushka Ghat
