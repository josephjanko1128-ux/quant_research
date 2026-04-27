#!/usr/bin/env python3
"""
SEC PDF Scraper - Comments on Roundtable on Trade-Through Prohibitions
and Roundtable on Rule 611 of Regulation NMS (File Number 4-862)

Downloads all PDF comment letters from:
  https://www.sec.gov/comments/4-862/4-862.htm

Usage:
  python scrape_sec_pdfs.py
  python scrape_sec_pdfs.py --output ./my_pdfs
  python scrape_sec_pdfs.py --delay 1.5 --output ./pdf
"""

import os
import re
import time
import argparse
import urllib.request
import urllib.error
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser


# ── Configuration ─────────────────────────────────────────────────────────────

SEC_BASE = "https://www.sec.gov"
COMMENTS_URL = "https://www.sec.gov/comments/4-862/4-862.htm"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research/academic use; contact: research@example.com)",
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
}


# ── HTML parser to extract PDF links ──────────────────────────────────────────

class PDFLinkParser(HTMLParser):
    """Extracts all href links ending in .pdf from an HTML page."""

    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.pdf_links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            if href.lower().endswith(".pdf"):
                full_url = urljoin(self.base_url, href)
                if full_url not in self.pdf_links:
                    self.pdf_links.append(full_url)


# ── Helpers ────────────────────────────────────────────────────────────────────

def fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def pdf_filename_from_url(url: str) -> str:
    """Derive a clean filename from the PDF URL."""
    path = urlparse(url).path          # e.g. /comments/4-862/4862-123-456.pdf
    return os.path.basename(path)     # e.g. 4862-123-456.pdf


def download_pdf(url: str, dest_path: str) -> bool:
    """Download a single PDF. Returns True on success."""
    if os.path.exists(dest_path):
        print(f"  [skip] already exists: {os.path.basename(dest_path)}")
        return True

    req = urllib.request.Request(url, headers={**HEADERS, "Accept": "application/pdf,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(dest_path, "wb") as f:
            f.write(data)
        size_kb = len(data) / 1024
        print(f"  [ok]   {os.path.basename(dest_path)}  ({size_kb:.1f} KB)")
        return True
    except urllib.error.HTTPError as e:
        print(f"  [err]  HTTP {e.code} — {os.path.basename(dest_path)}")
        return False
    except Exception as e:
        print(f"  [err]  {e} — {os.path.basename(dest_path)}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download all PDF comments for SEC Roundtable on Trade-Through "
                    "Prohibitions / Rule 611 of Regulation NMS (File No. 4-862)."
    )
    parser.add_argument(
        "--output", "-o",
        default="./pdf",
        help="Directory to save PDFs (default: ./pdf)"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=1.0,
        help="Seconds to wait between downloads (default: 1.0, be a good citizen)"
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory : {output_dir}")
    print(f"Source page      : {COMMENTS_URL}")
    print(f"Request delay    : {args.delay}s\n")

    # 1. Fetch the comments index page
    print("Fetching comment index page …")
    try:
        html = fetch_html(COMMENTS_URL)
    except Exception as e:
        print(f"ERROR: Could not fetch index page: {e}")
        raise SystemExit(1)

    # 2. Parse PDF links
    link_parser = PDFLinkParser(base_url=COMMENTS_URL)
    link_parser.feed(html)
    pdf_urls = link_parser.pdf_links

    if not pdf_urls:
        print("No PDF links found on the page. The SEC site structure may have changed.")
        print("Trying fallback: looking for all /comments/4-862/ PDF links via regex …")
        # Fallback: regex search in raw HTML
        raw_links = re.findall(r'href="(/comments/4-862/[^"]+\.pdf)"', html, re.IGNORECASE)
        pdf_urls = [urljoin(SEC_BASE, lnk) for lnk in raw_links]

    if not pdf_urls:
        print("Still no PDFs found. Please check the URL manually:")
        print(f"  {COMMENTS_URL}")
        raise SystemExit(1)

    print(f"Found {len(pdf_urls)} PDF link(s).\n")

    # 3. Download each PDF
    ok = 0
    fail = 0
    for i, url in enumerate(pdf_urls, start=1):
        filename = pdf_filename_from_url(url)
        dest = os.path.join(output_dir, filename)
        print(f"[{i}/{len(pdf_urls)}] {url}")
        if download_pdf(url, dest):
            ok += 1
        else:
            fail += 1
        if i < len(pdf_urls):
            time.sleep(args.delay)

    print(f"\n{'='*60}")
    print(f"Done.  Downloaded: {ok}   Failed: {fail}   Skipped: {len(pdf_urls)-ok-fail}")
    print(f"PDFs saved to: {output_dir}")


if __name__ == "__main__":
    main()