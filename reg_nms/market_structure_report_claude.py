
import json
import os
from pathlib import Path

import pandas as pd
import pypdf
from anthropic import Anthropic
from bs4 import BeautifulSoup
from docx import Document

INPUT_DIR = "sec_s7_2026_20_comments"
OUTPUT_DIR = "market_report"

Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(f"{OUTPUT_DIR}/analyses").mkdir(exist_ok=True)

ANTHROPIC_API_KEY='enter here'

os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=300.0)


def extract_text(path: Path):

    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        return ""

    suffix = path.suffix.lower()

    try:

        if suffix == ".pdf":
            reader = pypdf.PdfReader(str(path))
            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        if suffix == ".html":
            html = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            soup = BeautifulSoup(
                html,
                "html.parser"
            )

            return soup.get_text(
                "\n",
                strip=True
            )

        if suffix == ".txt":
            return path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

        if suffix == ".docx":
            doc = Document(str(path))

            return "\n".join(
                p.text
                for p in doc.paragraphs
            )

    except Exception as e:
        print("EXTRACTION ERROR:", path, e)

    return ""


ANALYSIS_PROMPT = """
You are an expert in market microstructure, SEC rulemaking,
exchange structure, broker-dealer economics, and retail trading.

Analyze the SEC comment letter.

Return ONLY valid JSON.

{
  "author": "",
  "affiliation": "",
  "stakeholder_type": "",
  "overall_position": "support|oppose|mixed|unclear",
  "tone": "",
  "executive_summary": "",
  "market_structure_topics": [],

  "positions": {
    "order_competition_rule": "",
    "best_execution": "",
    "tick_size_reform": "",
    "access_fee_reform": "",
    "transparency": ""
  },

  "economic_arguments": [],
  "liquidity_arguments": [],
  "competition_arguments": [],
  "retail_investor_arguments": [],

  "recommended_actions": [],
  "key_quotes": []
}
"""


def analyze_letter(text):

    text = text[:180000]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        temperature=0,
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": (
                    ANALYSIS_PROMPT +
                    "\n\nLETTER:\n\n" +
                    text
                )
            }
        ]
    )

    raw = response.content[0].text.strip()

    start = raw.find("{")
    end = raw.rfind("}")

    return json.loads(raw[start:end + 1])


def build_final_report(all_analyses):

    prompt = f"""
You are a former SEC economist and market structure specialist.

Using the analyses below:

{json.dumps(all_analyses, indent=2)}

Produce a detailed markdown report.

Include:

# Executive Summary

# Stakeholder Breakdown

# Support vs Opposition Statistics

# Major Themes

# Market Structure Issues Raised

# Economic Arguments

# Liquidity Arguments

# Competition Arguments

# Retail Investor Arguments

# Consensus Positions

# Outlier Positions

# Recommended SEC Actions

# Conclusion
"""

    chunks = []

    with client.messages.stream(
        model="claude-sonnet-4-6",
        temperature=0,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)
            print(text, end="", flush=True)

    print()  # newline after streaming
    return "".join(chunks)


def main():

    # Check input directory exists before running analysis
    if not Path(INPUT_DIR).exists():
        print(f"ERROR: Input directory does not exist: {INPUT_DIR}")
        return

    analyses = []
    files = []

    for ext in (
        "*.pdf",
        "*.html",
        "*.txt",
        "*.docx",
    ):
        files.extend(
            Path(INPUT_DIR).rglob(ext)
        )

    if not files:
        print(f"No supported files found in {INPUT_DIR}")
        return

    print(f"Found {len(files)} files")

    for i, file in enumerate(files, start=1):

        print(f"[{i}/{len(files)}] {file.name}")

        out_file = (
            Path(OUTPUT_DIR)
            / "analyses"
            / f"{file.stem}.json"
        )

        #
        # Skip Claude analysis if we've already processed this file
        #
        if out_file.exists():

            print(
                f"SKIPPING: Existing analysis found "
                f"({out_file.name})"
            )

            try:

                with open(
                    out_file,
                    "r",
                    encoding="utf-8"
                ) as f:
                    analysis = json.load(f)

                analysis["source_file"] = file.name

                analyses.append(analysis)

            except Exception as e:

                print(
                    f"ERROR reading existing analysis "
                    f"{out_file.name}: {e}"
                )

            continue

        text = extract_text(file)

        if len(text) < 200:

            print(
                f"SKIPPING: insufficient text "
                f"({len(text)} chars)"
            )

            continue

        try:

            analysis = analyze_letter(text)

            analysis["source_file"] = file.name

            analyses.append(analysis)

            with open(
                out_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    analysis,
                    f,
                    indent=2
                )

            print(
                f"SAVED: {out_file.name}"
            )

        except Exception as e:

            print(
                "ANALYSIS ERROR:",
                file,
                e
            )

    # Don't continue if nothing was analyzed
    if not analyses:
        print("No valid analyses generated. Exiting.")
        return

    with open(
        f"{OUTPUT_DIR}/all_analyses.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            analyses,
            f,
            indent=2
        )

    rows = []

    for a in analyses:

        rows.append({
            "author": a.get("author"),
            "affiliation": a.get("affiliation"),
            "stakeholder_type": a.get("stakeholder_type"),
            "overall_position": a.get("overall_position"),
            "tone": a.get("tone"),
            "source_file": a.get("source_file"),
        })

    pd.DataFrame(rows).to_csv(
        f"{OUTPUT_DIR}/aggregate_dataset.csv",
        index=False
    )

    print("Building final report...")

    report = build_final_report(
        analyses
    )

    with open(
        f"{OUTPUT_DIR}/market_structure_report.md",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

    print()
    print("Finished.")
    print("Outputs written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

