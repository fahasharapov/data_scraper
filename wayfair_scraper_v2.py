\
#!/usr/bin/env python3
"""
Wayfair Price Scraper v2 (Multi-seller + Confidence + Excel)
------------------------------------------------------------
Input: Excel/CSV with columns: sku, our_price, search_description
Output: Timestamped .xlsx with rows per (sku x top listings)

Features:
- Searches Wayfair using search_description (not SKU)
- Collects top 2–3 product listings from search (ignores sponsored where possible)
- Extracts seller name, item price, shipping price, total price, availability, product title
- Computes a match confidence (0–100) vs search_description
- Saves timestamped Excel file
- Runs on-demand (manual)

Usage:
  pip install -r requirements.txt
  python -m playwright install
  python wayfair_scraper_v2.py --input test_items.xlsx --outdir results/

Notes:
- This is a best-effort scraper; HTML structure can change.
- Use respectfully: slow pacing, low volume, follow site terms.
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from playwright.async_api import async_playwright

SEARCH_URL = "https://www.wayfair.com/keyword.php?keyword={q}"

def sleep_jitter(base=1.3, spread=0.9):
    t = max(0.4, random.uniform(base - spread, base + spread))
    time.sleep(t)

def tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    return re.findall(r"[a-z0-9]+", s)

def match_confidence(query: str, title: str) -> int:
    q = tokenize(query)
    t = tokenize(title)
    if not q or not t:
        return 0
    set_q, set_t = set(q), set(t)
    inter = len(set_q & set_t)
    union = len(set_q | set_t)
    base = 100.0 * inter / max(1, union)

    def bigrams(xs):
        return set(zip(xs, xs[1:])) if len(xs) >= 2 else set()
    inter2 = len(bigrams(q) & bigrams(t))
    boost = min(20, inter2 * 5)
    score = min(100, int(round(base + boost)))
    return score

def parse_price(text: str) -> Optional[float]:
    if not text:
        return None
    clean = text.replace(",", "")
    m = re.search(r"(\d{1,3}(?:\.\d{2})?)", clean)
    if m:
        try:
            return float(m.group(1))
        except:
            return None
    digits = "".join(ch for ch in clean if ch.isdigit() or ch == ".")
    try:
        return float(digits) if digits else None
    except:
        return None

def parse_price_from_jsonld(blobs: List[str]) -> Optional[float]:
    for j in blobs:
        try:
            data = json.loads(j.strip())
        except Exception:
            continue
        candidates = data if isinstance(data, list) else [data]
        for obj in candidates:
            offers = obj.get("offers")
            if not offers:
                continue
            def pick_price(offers_obj):
                if isinstance(offers_obj, dict):
                    return offers_obj.get("price") or offers_obj.get("lowPrice")
                if isinstance(offers_obj, list):
                    for off in offers_obj:
                        p = off.get("price") or off.get("lowPrice")
                        if p: return p
                return None
            p = pick_price(offers)
            if p:
                try:
                    return float(str(p).replace(",", "").strip())
                except:
                    pass
    return None

async def extract_jsonld(page) -> List[str]:
    scripts = await page.query_selector_all('script[type="application/ld+json"]')
    blobs = []
    for s in scripts:
        try:
            blobs.append((await s.inner_text()) or "")
        except:
            pass
    return blobs

async def get_top_search_results(page, query: str, limit: int = 3) -> List[str]:
    url = SEARCH_URL.format(q=query.replace(" ", "+"))
    await page.goto(url, wait_until="domcontentloaded", timeout=45000)
    sleep_jitter()

    product_card_selectors = [
        '[data-enzyme-id="plp-product-card"] a[href*="/pdp/"]',
        'a[data-testid="product-card-title"]',
        'a[href*="/pdp/"]'
    ]
    links = []
    seen = set()
    for sel in product_card_selectors:
        els = await page.query_selector_all(sel)
        for a in els:
            try:
                container = await a.evaluate_handle("el => el.closest('[data-enzyme-id], article, div')")
                if container:
                    try:
                        txt = (await container.evaluate("el => el.innerText")).lower()
                        if "sponsored" in txt:
                            continue
                    except:
                        pass
                href = await a.get_attribute("href")
                if not href: 
                    continue
                if not href.startswith("http"):
                    href = "https://www.wayfair.com" + href
                if href in seen:
                    continue
                seen.add(href)
                links.append(href)
                if len(links) >= limit:
                    break
            except Exception:
                continue
        if len(links) >= limit:
            break
    return links

async def extract_listing_details(context, url: str, search_query: str) -> Dict[str, Any]:
    page = await context.new_page()
    await page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
    data = {
        "product_url": url,
        "product_title": None,
        "seller_name": None,
        "item_price": None,
        "shipping_price": 0.0,
        "total_price": None,
        "availability": None,
        "match_confidence": 0,
        "competitor": "Wayfair",
    }
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        sleep_jitter()

        # Title
        try:
            tsel = 'h1[data-testid="product-title"], h1'
            t = await page.locator(tsel).first.inner_text(timeout=4000)
            data["product_title"] = t.strip()
        except Exception:
            pass

        # JSON-LD blobs
        blobs = await extract_jsonld(page)
        price_from_jsonld = parse_price_from_jsonld(blobs)

        # Visible price selectors
        visible_price = None
        price_sels = [
            '[data-testid="pd-price"]',
            '[data-testid="sale-price"]',
            'span[data-test="current-price"]',
            'div:has-text("$") span',
        ]
        for sel in price_sels:
            try:
                el = await page.query_selector(sel)
                if el:
                    txt = (await el.inner_text()).strip()
                    p = parse_price(txt)
                    if p:
                        visible_price = p
                        break
            except:
                pass

        # Seller name
        seller_candidates = [
            'span:has-text("Sold by")',
            'div:has-text("Sold by")',
            '[data-testid="merchant-info"]',
            '[class*="MerchantInfo"]',
        ]
        for sel in seller_candidates:
            try:
                el = await page.query_selector(sel)
                if el:
                    txt = (await el.inner_text()).strip()
                    m = re.search(r"Sold by\s+(.+)$", txt, re.IGNORECASE)
                    data["seller_name"] = m.group(1).strip() if m else txt
                    break
            except:
                pass
        if not data["seller_name"]:
            data["seller_name"] = "Wayfair"

        # Shipping price
        ship_candidates = [
            '[data-testid="shipping-info"]',
            'div:has-text("Free Shipping")',
            'div:has-text("Shipping")',
            'div:has-text("Delivery")',
        ]
        shipping = None
        for sel in ship_candidates:
            try:
                el = await page.query_selector(sel)
                if el:
                    txt = (await el.inner_text()).strip()
                    if "free" in txt.lower():
                        shipping = 0.0
                        break
                    maybe = parse_price(txt)
                    if maybe is not None:
                        shipping = maybe
                        break
            except:
                pass
        data["shipping_price"] = float(shipping) if shipping is not None else 0.0

        item_price = price_from_jsonld if price_from_jsonld is not None else visible_price
        data["item_price"] = item_price

        # Availability
        avail_candidates = [
            '[data-testid="availability"]',
            'div:has-text("In Stock")',
            'div:has-text("Out of Stock")',
        ]
        for sel in avail_candidates:
            try:
                el = await page.query_selector(sel)
                if el:
                    data["availability"] = (await el.inner_text()).strip()
                    break
            except:
                pass

        if item_price is not None:
            data["total_price"] = round(float(item_price) + float(data["shipping_price"]), 2)

        data["match_confidence"] = match_confidence(search_query, data.get("product_title") or "")

    finally:
        await page.close()
    return data

async def run(args):
    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: input file not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Read input
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"sku", "our_price", "search_description"}
    if not required.issubset(df.columns):
        print(f"ERROR: Input must have columns: {sorted(required)}. Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir or ".")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ))

        page = await context.new_page()

        for _, r in df.iterrows():
            sku = str(r.get("sku", "")).strip()
            our_price = float(r.get("our_price", 0) or 0)
            query = str(r.get("search_description", "")).strip()
            if not query:
                continue

            try:
                links = await get_top_search_results(page, query, limit=3)
            except Exception:
                links = []
            if not links:
                rows.append({
                    "sku": sku,
                    "our_price": our_price,
                    "competitor": "Wayfair",
                    "seller_name": None,
                    "product_title": None,
                    "item_price": None,
                    "shipping_price": None,
                    "total_price": None,
                    "delta": None,
                    "cheaper_than_us": None,
                    "match_confidence": 0,
                    "product_url": None,
                    "availability": None,
                })
                continue

            for url in links:
                try:
                    data = await extract_listing_details(context, url, query)
                except Exception:
                    data = {
                        "product_url": url, "product_title": None, "seller_name": None,
                        "item_price": None, "shipping_price": None, "total_price": None,
                        "availability": None, "match_confidence": 0, "competitor": "Wayfair"
                    }

                total = data.get("total_price")
                delta = (total - our_price) if total is not None else None
                cheaper = (total is not None and total < our_price)

                rows.append({
                    "sku": sku,
                    "our_price": our_price,
                    "competitor": data.get("competitor"),
                    "seller_name": data.get("seller_name"),
                    "product_title": data.get("product_title"),
                    "item_price": data.get("item_price"),
                    "shipping_price": data.get("shipping_price"),
                    "total_price": total,
                    "delta": delta,
                    "cheaper_than_us": cheaper,
                    "match_confidence": data.get("match_confidence", 0),
                    "product_url": data.get("product_url"),
                    "availability": data.get("availability"),
                })
                sleep_jitter(1.7, 1.2)

        await browser.close()

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(by=["sku", "match_confidence", "total_price"],
                                ascending=[True, False, True])

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path(args.outdir) / f"wayfair_results_{ts}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="results")

    print(f"Wrote: {out_path}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to Excel/CSV with columns: sku, our_price, search_description")
    ap.add_argument("--outdir", default=".", help="Directory to write the timestamped Excel")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(args))
