# Wayfair Price Scraper (Prototype)

## What you have
- `wayfair_scraper.py` — Playwright-based scraper that extracts the price & title from Wayfair PDP, or searches by SKU if URL is blank.
- `products_sample.csv` — Template input with columns: `sku, our_price, wayfair_url`.
- `requirements.txt` — Python dependencies.

## Quickstart
```bash
pip install -r requirements.txt
python -m playwright install
python wayfair_scraper.py --input products_sample.csv --out results.csv
```

## Output
A CSV with: `sku, our_price, competitor, competitor_price, delta, cheaper_than_us, product_title, product_url, availability`.

## Tips
- Prefer providing exact Wayfair product URLs for better accuracy.
- Run sparingly, add delays, and respect Wayfair's terms of use.
- If Wayfair changes their site structure, update selectors in `extract_wayfair_price()`.
