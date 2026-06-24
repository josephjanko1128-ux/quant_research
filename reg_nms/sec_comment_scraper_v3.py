import csv
import os
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.sec.gov"
COMMENTS_URL = "https://www.sec.gov/rules-regulations/public-comments/s7-2026-20"

OUTPUT_DIR = "sec_s7_2026_20_comments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({
    # Replace with your actual contact info
    "User-Agent": "Your Name your@email.com",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
})

COMMENT_EXTENSIONS = (
    ".html",
    ".pdf",
    ".txt",
    ".doc",
    ".docx",
)


def safe_filename(name):
    name = re.sub(r"[^\w\s.-]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:150]


print("Loading comments page...")
resp = session.get(COMMENTS_URL, timeout=30)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "html.parser")

comment_urls = set()

for a in soup.find_all("a", href=True):
    href = a["href"]

    if "/comments/" not in href:
        continue

    if href.lower().endswith(COMMENT_EXTENSIONS):
        comment_urls.add(urljoin(BASE_URL, href))

print(f"Discovered {len(comment_urls)} comment URLs")

records = []

for comment_url in sorted(comment_urls):

    lower = comment_url.lower()

    try:
        if lower.endswith((".pdf", ".txt", ".doc", ".docx")):

            records.append({
                "title": os.path.basename(comment_url),
                "url": comment_url,
                "type": "direct_comment_file",
            })

            continue

        if lower.endswith(".html"):

            r = session.get(comment_url, timeout=30)
            r.raise_for_status()

            comment_soup = BeautifulSoup(r.text, "html.parser")

            title = (
                comment_soup.title.get_text(strip=True)
                if comment_soup.title
                else os.path.basename(comment_url)
            )

            records.append({
                "title": title,
                "url": comment_url,
                "type": "html_comment_page",
            })

            for a in comment_soup.find_all("a", href=True):

                href = a["href"]

                if href.lower().endswith(
                    (".pdf", ".txt", ".doc", ".docx")
                ):
                    records.append({
                        "title": title,
                        "url": urljoin(BASE_URL, href),
                        "type": "attachment",
                    })

            time.sleep(0.2)

    except Exception as e:
        print("ERROR:", comment_url, e)

print(f"Collected {len(records)} downloadable items")

manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")

with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["title", "type", "url", "local_file"]
    )
    writer.writeheader()

    for i, rec in enumerate(records, start=1):

        url_path = rec["url"].split("?")[0]
        ext = os.path.splitext(url_path)[1] or ".html"

        filename = (
            f"{i:05d}_"
            f"{safe_filename(rec['title'] or 'comment')}"
            f"{ext}"
        )

        filepath = os.path.join(OUTPUT_DIR, filename)

        try:
            r = session.get(rec["url"], timeout=60)
            r.raise_for_status()

            with open(filepath, "wb") as out:
                out.write(r.content)

            writer.writerow({
                "title": rec["title"],
                "type": rec["type"],
                "url": rec["url"],
                "local_file": filename,
            })

            print("Saved", filename)

        except Exception as e:
            print("DOWNLOAD ERROR:", rec["url"], e)

        time.sleep(0.2)

print("Done.")
