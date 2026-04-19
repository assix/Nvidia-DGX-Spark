#!/usr/bin/env python3
"""
Fetch and summarise DGX Spark build guides from:
  https://build.nvidia.com/spark
"""

import json
import re
import textwrap
import urllib.request
from datetime import datetime, timedelta, timezone

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────

WIDTH = 88
SPARK_GUIDES_URL = "https://build.nvidia.com/spark"

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_url(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (NVIDIA-AI-News-Fetcher/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


# ── DGX Spark Playbooks ──────────────────────────────────────────────────────

def _relative_to_date(rel: str) -> str:
    """Convert a relative-time string (e.g. '2w', '1mo', '6mo') to YYYY-MM-DD."""
    now = datetime.now(timezone.utc)
    m = re.match(r'^(\d+)(w|mo|d|h)$', rel.strip().lower())
    if not m:
        return "unknown"
    value, unit = int(m.group(1)), m.group(2)
    if unit == "w":
        delta = timedelta(weeks=value)
    elif unit == "mo":
        delta = timedelta(days=value * 30)
    elif unit == "d":
        delta = timedelta(days=value)
    else:  # hours
        delta = timedelta(hours=value)
    return (now - delta).strftime("%Y-%m-%d")


def _parse_guides_from_next_data(html: str) -> list[dict]:
    """Try to extract playbooks from the __NEXT_DATA__ JSON embedded in the page."""
    nd = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if not nd:
        return []
    try:
        data = json.loads(nd.group(1))
    except json.JSONDecodeError:
        return []

    # Try the Next.js data-fetch endpoint for the /spark page
    build_id = data.get("buildId", "")
    if build_id:
        try:
            api_url = f"https://build.nvidia.com/_next/data/{build_id}/spark.json"
            raw = fetch_url(api_url)
            page_data = json.loads(raw)
            props = page_data.get("pageProps", {})
        except Exception:
            props = data.get("props", {}).get("pageProps", {})
    else:
        props = data.get("props", {}).get("pageProps", {})

    # Field names vary by API version — try common candidates
    guides_raw = (
        props.get("guides")
        or props.get("playbooks")
        or props.get("items")
        or props.get("recipes")
        or []
    )
    playbooks: list[dict] = []
    for g in guides_raw:
        if not isinstance(g, dict):
            continue
        title = g.get("title") or g.get("name", "")
        slug  = g.get("slug")  or g.get("id", "")
        url   = f"https://build.nvidia.com/spark/{slug}" if slug else ""
        desc  = g.get("description") or g.get("summary", "")
        released = (
            g.get("publishedAt") or g.get("createdAt") or g.get("created_at") or ""
        )
        updated = (
            g.get("updatedAt") or g.get("updated_at") or g.get("modifiedAt") or released
        )
        duration = g.get("duration") or g.get("estimatedTime") or g.get("time", "")
        tags = [
            (t.get("name") or t.get("label") if isinstance(t, dict) else str(t))
            for t in g.get("tags", [])
        ]
        if title:
            playbooks.append({
                "title":        title,
                "url":          url,
                "description":  desc,
                "release_date": released[:10] if released else "unknown",
                "updated_date": updated[:10]  if updated  else "unknown",
                "duration":     str(duration),
                "tags":         [t for t in tags if t],
            })
    return playbooks


def _parse_guides_from_html(html: str) -> list[dict]:
    """Parse playbook cards from raw HTML using BeautifulSoup (if available) or regex."""
    playbooks: list[dict] = []

    if REQUESTS_AVAILABLE:
        soup = BeautifulSoup(html, "html.parser")
        # Each guide is rendered as an <h3> containing an <a> link
        for h3 in soup.find_all("h3"):
            a = h3.find("a", href=True)
            if not a:
                continue
            href = a["href"]
            # Only cards that are sub-pages of /spark/
            if "/spark/" not in href:
                continue
            url   = href if href.startswith("http") else f"https://build.nvidia.com{href}"
            title = a.get_text(strip=True)
            # Sibling / parent text for description, relative time, duration
            card       = h3.parent or h3
            card_text  = card.get_text(" ", strip=True)
            # Description: first sentence-like text that isn't the title or a short tag
            desc = ""
            for el in card.find_all(["p", "span"]):
                t = el.get_text(strip=True)
                if t and t != title and len(t) > 20:
                    desc = t
                    break
            # Relative time e.g. "2w", "1mo", "6mo"
            rel_m = re.search(r'\b(\d+(?:w|mo|d))\b', card_text)
            rel   = rel_m.group(1) if rel_m else ""
            # Duration e.g. "30 min", "1 hr"
            dur_m = re.search(r'\b(\d+\s*(?:min|hr|hour)s?)\b', card_text, re.I)
            dur   = dur_m.group(1) if dur_m else ""
            approx = _relative_to_date(rel) if rel else "unknown"
            if title:
                playbooks.append({
                    "title":        title,
                    "url":          url,
                    "description":  desc,
                    "release_date": "unknown",          # index page doesn't expose it
                    "updated_date": approx,
                    "duration":     dur,
                    "tags":         [],
                })
    else:
        # Pure-regex fallback — extract <a href="/spark/…"> with nearby text
        for m in re.finditer(
            r'href="(/spark/[^"]+)"[^>]*>([^<]{3,})</a>',
            html,
        ):
            href, title = m.group(1), m.group(2).strip()
            if title:
                playbooks.append({
                    "title":        title,
                    "url":          f"https://build.nvidia.com{href}",
                    "description":  "",
                    "release_date": "unknown",
                    "updated_date": "unknown",
                    "duration":     "",
                    "tags":         [],
                })

    # De-duplicate by URL
    seen: set[str] = set()
    unique: list[dict] = []
    for p in playbooks:
        if p["url"] not in seen:
            seen.add(p["url"])
            unique.append(p)
    return unique


def fetch_spark_playbooks() -> list[dict]:
    """Fetch and return DGX Spark build guides from build.nvidia.com/spark."""
    raw  = fetch_url(SPARK_GUIDES_URL)
    html = raw.decode("utf-8", errors="replace")

    # Try structured data from Next.js first; fall back to HTML parsing
    playbooks = _parse_guides_from_next_data(html) or _parse_guides_from_html(html)
    return playbooks


# ── Display ───────────────────────────────────────────────────────────────────

def print_playbook(idx: int, item: dict):
    bar = "─" * WIDTH
    print(f"\n{bar}")
    print(f"  [{idx:2d}] {item['title']}")
    print(f"  Released  : {item['release_date']}")
    print(f"  Updated   : {item['updated_date']}")
    if item["duration"]:
        print(f"  Est. time : {item['duration']}")
    if item["tags"]:
        print(f"  Tags      : {', '.join(item['tags'])}")
    print(f"  URL       : {item['url']}")
    if item["description"]:
        desc    = item["description"][:280].strip()
        wrapped = textwrap.fill(
            desc, width=WIDTH - 4, initial_indent="  ", subsequent_indent="  "
        )
        print(f"\n{wrapped}{'…' if len(item['description']) > 280 else ''}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"{'═' * WIDTH}")
    print(f"  DGX Spark Build Guides — build.nvidia.com/spark")
    print(f"  Fetched: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'═' * WIDTH}")
    print(f"\n  ⏳  Fetching DGX Spark playbooks …", end="", flush=True)
    try:
        spark_guides = fetch_spark_playbooks()
        if spark_guides:
            print(f" ✓  ({len(spark_guides)} guides found)")
            print(
                f"\n  NOTE: 'Release date' is available only when the guide page exposes\n"
                f"  it; index page dates are approximate last-updated times inferred\n"
                f"  from relative timestamps (e.g. '2w', '1mo').\n"
            )
            for i, guide in enumerate(spark_guides, 1):
                print_playbook(i, guide)
        else:
            print(" ✗  (no guides parsed — site may require JS rendering)")
            print(
                "  Tip: install 'requests' and 'beautifulsoup4' for improved parsing:\n"
                "       pip install requests beautifulsoup4"
            )
    except Exception as exc:
        print(f" ✗  ({exc})")

    print(f"\n{'═' * WIDTH}\n")


if __name__ == "__main__":
    main()
