#!/usr/bin/env python3
"""
Zoro Scraper v7 — Stealth + Shadow DOM + Debug Mode
- Top 5 results per query
- Shadow DOM–aware link collection
- Auto-scrolling w/ human-like delays
- Always-on basic stealth (webdriver masking, languages, plugins, etc.)
- Debug mode: saves screenshots & HTML when no results (and on product errors)
- Outputs timestamped Excel; columns match Wayfair + search_description
"""

import argparse
import asyncio
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
    Page,
)

SEARCH_URL = "https://www.zoro.com/search?q={q}"
HOME_URL = "https://www.zoro.com/"


# ---------------- Utilities ----------------
def sleep_jitter(base=2.0, spread=1.2):
    t = max(0.5, random.uniform(base - spread, base + spread))
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
    def bigrams(xs): return set(zip(xs, xs[1:])) if len(xs) >= 2 else set()
    inter2 = len(bigrams(q) & bigrams(t))
    boost = min(20, inter2 * 5)
    return min(100, int(round(base + boost)))

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

# ---------------- Stealth & Humanization ----------------
STEALTH_INIT_SCRIPT = r"""
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
window.navigator.chrome = { runtime: {}, apps: {}, csi: function(){}, loadTimes: function(){} };
Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
const origQuery = window.navigator.permissions && window.navigator.permissions.query;
if (origQuery) {
  const newQuery = origQuery.bind(window.navigator.permissions);
  window.navigator.permissions.query = (parameters) => (
    parameters.name === 'notifications'
      ? Promise.resolve({ state: Notification.permission })
      : newQuery(parameters)
  );
}
"""

async def human_mouse_move(page: Page, width=1366, height=768):
    try:
        x = random.randint(int(width*0.2), int(width*0.8))
        y = random.randint(int(height*0.2), int(height*0.8))
        await page.mouse.move(x, y, steps=random.randint(6, 12))
        if random.random() < 0.35:
            await page.mouse.move(x + random.randint(-80, 80), y + random.randint(-60, 60), steps=random.randint(5, 10))
    except Exception:
        pass

async def gentle_autoscroll(page: Page, max_rounds: int = 10, wait_ms_min: int = 900, wait_ms_max: int = 1500):
    prev_count = -1
    for _ in range(max_rounds):
        try:
            await page.evaluate("window.scrollBy(0, 700)")
        except Exception:
            pass
        await page.wait_for_timeout(random.randint(wait_ms_min, wait_ms_max))
        if random.random() < 0.25:
            try:
                await page.evaluate("window.scrollBy(0, -200)")
            except Exception:
                pass
            await page.wait_for_timeout(random.randint(300, 800))
        await human_mouse_move(page)
        try:
            count = await page.evaluate("document.querySelectorAll('a[href*=\"/i/\"]').length")
        except Exception:
            count = prev_count
        if count == prev_count:
            break
        prev_count = count

# ---------------- Link Collection ----------------
async def _is_visible(page: Page, selector: str, timeout: int = 1000) -> bool:
    loc = page.locator(selector)
    if await loc.count() == 0:
        return False
    try:
        return await loc.first.is_visible(timeout=timeout)
    except PlaywrightTimeoutError:
        return False
    except Exception:
        return False


async def is_datadome_challenge(page: Page) -> bool:
    """Return True when the current page is the DataDome captcha challenge."""
    try:
        url = page.url.lower()
        if any(token in url for token in ("captcha-delivery.com", "/captcha/", "/deny/")):
            return True

        challenge_selectors = [
            "form#captcha-form",
            "form[action*='datadome']",
            "div[class*='captcha'] >> text=/please verify/i",
            "text=/Access to this page has been denied/i",
            "text=/Please verify you are a human/i",
            "iframe[src*='captcha-delivery.com']",
        ]
        for sel in challenge_selectors:
            if await _is_visible(page, sel):
                return True
        return False
        if await page.locator("iframe[src*='captcha-delivery.com']").count() > 0:
            return True

        if await page.locator("input[name='datadome']").count() > 0:
            return True

        content = await page.content()
        challenge_markers = (
            "Access to this page has been denied",
            "Please verify you are a human",
        )
        text = content.lower()
        return any(marker.lower() in text for marker in challenge_markers)
    except Exception:
        return False

async def refresh_session_cookie(page: Page) -> bool:
    try:
        await page.context.clear_cookies()
    except Exception:
        pass
    try:
        await page.goto(HOME_URL, wait_until="domcontentloaded", timeout=60000)
    except Exception:
        return False
    try:
        await page.wait_for_load_state("networkidle", timeout=15000)
    except PlaywrightTimeoutError:
        pass
    await page.wait_for_timeout(900 + random.randint(0, 600))
    await human_mouse_move(page)
    try:
        await page.wait_for_selector("body", timeout=3000)
    except PlaywrightTimeoutError:
        pass
    return not await is_datadome_challenge(page)

async def collect_links_dom_and_shadow(page: Page, limit: int = 5) -> List[str]:
    try:
        js = f"""() => {{
            const sels = [
              '#plp-root a[href]',
              'a[data-testid="product-link"]',
              'a[data-qa="plp-product-link"]',
              'article[data-testid="product-card"] a[href]',
              'a[href*="/i/"]',
              'a[href*="/product-detail/"]',
              'a[href*="/item/"]'
            ];
            const seen = new Set();
            const add = (h) => {{
              if(!h) return;
              const full = h.startsWith('http') ? h : ('https://www.zoro.com' + h);
              if (!full.includes('/c/') && !full.includes('/category/')) {{
                seen.add(full);
              }}
            }};
            for (const s of sels) document.querySelectorAll(s).forEach(a => add(a.getAttribute('href')));
            // shadow roots
            document.querySelectorAll('*').forEach(el => {{
              if (el.shadowRoot) {{
                for (const s of sels) {{
                  el.shadowRoot.querySelectorAll(s).forEach(a => add(a.getAttribute('href')));
                }}
              }}
            }});
            return Array.from(seen).slice(0, {limit});
        }}"""
        links = await page.evaluate(js)
        return links or []
    except Exception:
        return []

async def get_top_search_results(
    page: Page,
    query: str,
    limit: int,
    debug: bool,
    debug_dir: Path,
    sku: str,
) -> Tuple[List[str], bool]:
async def get_top_search_results(page: Page, query: str, limit: int, debug: bool, debug_dir: Path, sku: str) -> List[str]:
    async def _has_product_candidates() -> bool:
        try:
            candidate_selectors = [
                "article[data-testid='product-card'] a[href]",
                "a[data-qa='plp-product-link']",
                "a[data-testid='product-link']",
            ]
            for sel in candidate_selectors:
                if await page.locator(sel).count() > 0:
                    return True
            return False
        except Exception:
            return False

    url = SEARCH_URL.format(q=query.replace(" ", "+"))
    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
    try:
        await page.wait_for_load_state("networkidle", timeout=15000)
    except PlaywrightTimeoutError:
        pass
    if await is_datadome_challenge(page):
        print("  ⚠️ DataDome challenge detected; refreshing session…")
        if await refresh_session_cookie(page):
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
            except PlaywrightTimeoutError:
                pass
        else:
            print("  ⚠️ Unable to refresh session; challenge persists.")
    if await is_datadome_challenge(page):
        return []
    await page.wait_for_timeout(800 + random.randint(0, 900))
    await human_mouse_move(page)

    try:
        await page.wait_for_selector(
            "article[data-testid='product-card'] a[href], a[data-qa='plp-product-link'], a[data-testid='product-link'], text='Access to this page has been denied', text='Please verify you are a human'",
            timeout=12000,
        )
    except PlaywrightTimeoutError:
        pass

    try:
        await page.wait_for_selector(
            'article[data-testid="product-card"] a[href], a[data-qa="plp-product-link"], a[href*="/i/"]',
            timeout=12000
        )
    except PlaywrightTimeoutError:
        pass

    if await is_datadome_challenge(page) and not await _has_product_candidates():
        print("  ⚠️ DataDome challenge detected; refreshing session…")
        if await refresh_session_cookie(page):
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
            except PlaywrightTimeoutError:
                pass
            try:
                await page.wait_for_selector(
                    "article[data-testid='product-card'] a[href], a[data-qa='plp-product-link'], a[data-testid='product-link']",
                    timeout=12000,
                )
            except PlaywrightTimeoutError:
                pass
        else:
            print("  ⚠️ Unable to refresh session; challenge persists.")
            return [], True

    if await is_datadome_challenge(page) and not await _has_product_candidates():
        return [], True

    await page.wait_for_timeout(800 + random.randint(0, 900))
    await human_mouse_move(page)

    await gentle_autoscroll(page)
    links = await collect_links_dom_and_shadow(page, limit=limit)
    if not links:
        await page.wait_for_timeout(1200)
        links = await collect_links_dom_and_shadow(page, limit=limit)

    # Debug capture when nothing found
    if debug and not links:
        try:
            (debug_dir).mkdir(parents=True, exist_ok=True)
            png = debug_dir / f"{sku}_search.png"
            html = debug_dir / f"{sku}_search.html"
            await page.screenshot(path=str(png), full_page=True)
            content = await page.content()
            with open(html, "w", encoding="utf-8") as f:
                f.write(content[:50000])  # first 50KB
            print(f"  [debug] Saved search screenshot: {png.name}")
            print(f"  [debug] Saved search HTML: {html.name}")
        except Exception as e:
            print(f"  [debug] capture failed: {e}")

    # Log count
    if links:
        print(f"  Found {len(links)} raw link(s); taking top {min(limit, len(links))}.")
    return links[:limit], False

# ---------------- Product Extraction ----------------
async def extract_listing_details(context, url: str, search_query: str, debug: bool, debug_dir: Path, sku: str, idx: int) -> Dict[str, Any]:
    page = await context.new_page()
    data = {
        "product_url": url,
        "product_title": None,
        "seller_name": None,
        "item_price": None,
        "shipping_price": None,
        "total_price": None,
        "availability": None,
        "match_confidence": 0,
        "competitor": "Zoro",
    }
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
        try:
            await page.wait_for_selector('h1, [data-qa="pdp-title"]', timeout=7000)
        except PlaywrightTimeoutError:
            pass
        await page.wait_for_timeout(random.randint(900, 1700))

        # Title
        try:
            tsel = 'h1, h1[data-qa="pdp-title"]'
            t = await page.locator(tsel).first.inner_text(timeout=3000)
            data["product_title"] = (t or "").strip()
        except Exception:
            pass

        # Price
        try:
            price_sel = '[data-qa="pdp-price"], [itemprop="price"], span:has-text("$")'
            el = await page.query_selector(price_sel)
            if el:
                txt = (await el.inner_text()).strip()
                p = parse_price(txt)
                if p is not None:
                    data["item_price"] = p
        except Exception:
            pass

        # Seller (fallback to Zoro)
        try:
            seller_sel = 'span:has-text("Sold by"), div:has-text("Sold by"), [data-qa="merchant-info"]'
            el = await page.query_selector(seller_sel)
            if el:
                txt = (await el.inner_text()).strip()
                m = re.search(r"Sold by\s+(.+)$", txt, re.IGNORECASE)
                data["seller_name"] = m.group(1).strip() if m else txt
        except Exception:
            pass
        if not data["seller_name"]:
            data["seller_name"] = "Zoro"

        # Shipping (0.0 if “Free Shipping”, else numeric if found, else None)
        try:
            ship_candidates = ['div:has-text("Free Shipping")', '[data-qa="shipping-info"]', 'div:has-text("Shipping")']
            ship = None
            for sel in ship_candidates:
                el = await page.query_selector(sel)
                if not el:
                    continue
                txt = (await el.inner_text()).strip()
                if "free shipping" in txt.lower():
                    ship = 0.0
                    break
                maybe = parse_price(txt)
                if maybe is not None:
                    ship = maybe
                    break
            data["shipping_price"] = ship
        except Exception:
            pass

        # Availability
        try:
            avail_sel = '[data-qa="availability"], div:has-text("In Stock"), div:has-text("Out of Stock")'
            el = await page.query_selector(avail_sel)
            if el:
                data["availability"] = (await el.inner_text()).strip()
        except Exception:
            pass

        # Total if shipping known
        if (data["item_price"] is not None) and (data["shipping_price"] is not None):
            data["total_price"] = round(float(data["item_price"]) + float(data["shipping_price"]), 2)

        # Confidence
        data["match_confidence"] = match_confidence(search_query, data.get("product_title") or "")

    except Exception as e:
        if debug:
            try:
                debug_dir.mkdir(parents=True, exist_ok=True)
                png = debug_dir / f"{sku}_{idx}.png"
                await page.screenshot(path=str(png), full_page=True)
                print(f"  [debug] Saved PDP screenshot (error case): {png.name}")
            except Exception as e2:
                print(f"  [debug] PDP capture failed: {e2}")
        raise e
    finally:
        try:
            await page.close()
        except Exception:
            pass
    return data

# ---------------- Main ----------------
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
    debug_dir = outdir / "debug" if args.debug else outdir  # separate debug folder

    rows = []
    headless = str(args.headless).lower() not in ("0", "false", "f", "no", "n")
    limit = 5

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(
            locale="en-US",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"),
            viewport={"width": 1366, "height": 768},
        )
        # stealth early
        await context.add_init_script(STEALTH_INIT_SCRIPT)

        page = await context.new_page()
        await refresh_session_cookie(page)
        total_items = len(df.index)

        for idx, r in df.iterrows():
            sku = str(r.get("sku", "")).strip()
            our_price = float(r.get("our_price", 0) or 0)
            query = str(r.get("search_description", "")).strip()
            if not query:
                continue

            print(f"[{idx+1}/{total_items}] Searching for: {query}")
            sleep_jitter(1.5, 1.0)
            links = []
            challenge_persisted = False

            for attempt in range(2):
                try:
                    links, challenge_persisted = await get_top_search_results(
                        page,
                        query,
                        limit=limit,
                        debug=args.debug,
                        debug_dir=debug_dir,
                        sku=sku,
                    )
                    if challenge_persisted:
                        print("  ⚠️ DataDome challenge persists; skipping further retries for this query.")
                        break
                    if links:
                        break
                    else:
                        print(f"  ↪ retrying… (attempt {attempt+1}/2)")
                        sleep_jitter(2.2, 1.2)
                except DataDomeChallengeError:
                    print("  ⚠️ DataDome challenge persists; skipping further retries for this query.")
                    break
                except Exception as e:
                    print(f"  retry {attempt+1} error: {e}")
                    sleep_jitter(2.5, 1.2)

            if not links:
                print("  ✗ No results found.")
                rows.append({
                    "sku": sku,
                    "search_description": query,
                    "our_price": our_price,
                    "competitor": "Zoro",
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

            print(f"  ✓ Found {len(links)} listing(s).")
            for i, url in enumerate(links, start=1):
                sleep_jitter(1.2, 0.9)
                try:
                    data = await extract_listing_details(context, url, query, debug=args.debug, debug_dir=debug_dir, sku=sku, idx=i)
                except Exception as e:
                    # on error, still record a row so you can see the URL that failed
                    data = {
                        "product_url": url,
                        "product_title": None,
                        "seller_name": None,
                        "item_price": None,
                        "shipping_price": None,
                        "total_price": None,
                        "availability": None,
                        "match_confidence": 0,
                        "competitor": "Zoro",
                    }

                total = data.get("total_price") if data.get("total_price") is not None else data.get("item_price")
                delta = (total - our_price) if total is not None else None
                cheaper = (total is not None and total < our_price)

                rows.append({
                    "sku": sku,
                    "search_description": query,
                    "our_price": our_price,
                    "competitor": data.get("competitor"),
                    "seller_name": data.get("seller_name"),
                    "product_title": data.get("product_title"),
                    "item_price": data.get("item_price"),
                    "shipping_price": data.get("shipping_price"),
                    "total_price": data.get("total_price"),
                    "delta": delta,
                    "cheaper_than_us": cheaper,
                    "match_confidence": data.get("match_confidence", 0),
                    "product_url": data.get("product_url"),
                    "availability": data.get("availability"),
                })

        await browser.close()

    out_df = pd.DataFrame(rows)
    out_df["__total_sort__"] = out_df["total_price"].apply(lambda x: float("inf") if pd.isna(x) else x)
    out_df = out_df.sort_values(by=["sku", "match_confidence", "__total_sort__"],
                                ascending=[True, False, True]).drop(columns="__total_sort__")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path(args.outdir) / f"zoro_results_{ts}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="results")
    print(f"\n✅ Done! Wrote: {out_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Excel/CSV with columns: sku, our_price, search_description")
    p.add_argument("--outdir", default=".", help="Directory to write results")
    p.add_argument("--headless", default="true", help="true/false; default true")
    p.add_argument("--debug", default="false", help="true/false; saves screenshots & HTML when no results / errors")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.headless = str(args.headless).lower() not in ("0", "false", "f", "no", "n")
    args.debug = str(args.debug).lower() in ("1", "true", "t", "yes", "y")
    asyncio.run(run(args))
