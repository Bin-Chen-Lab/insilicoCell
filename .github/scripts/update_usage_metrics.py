#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path


METRICS_URL = "https://apps.octad.org/insilicocell/metrics.json"
RELEASES_URL = "https://api.github.com/repos/Bin-Chen-Lab/insilicoCell/releases?per_page=100&page={}"


def fetch_json(url: str):
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "insilicocell-metrics-workflow"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


metrics = fetch_json(METRICS_URL)
releases = []
for page in range(1, 101):
    batch = fetch_json(RELEASES_URL.format(page))
    releases.extend(batch)
    if len(batch) < 100:
        break
else:
    raise SystemExit("More than 10,000 releases; extend pagination before reporting a total")
release_downloads = sum(
    int(asset.get("download_count", 0))
    for release in releases
    for asset in release.get("assets", [])
)

block = (
    "<!-- USAGE_METRICS_START -->\n"
    f"Total unique InsilicoCell installations: {int(metrics['total_unique_insilicocell_installations']):,}  \n"
    f"Total completed predictions: {int(metrics['total_completed_predictions']):,}  \n"
    f"Total GitHub release downloads: {release_downloads:,}  \n"
    f"Tracking began: {metrics['tracking_began']}\n"
    "<!-- USAGE_METRICS_END -->"
)

readme = Path("README.md")
original = readme.read_text(encoding="utf-8")
updated, replacements = re.subn(
    r"<!-- USAGE_METRICS_START -->.*?<!-- USAGE_METRICS_END -->",
    block,
    original,
    flags=re.DOTALL,
)
if replacements != 1:
    raise SystemExit("README usage-metrics marker is missing or duplicated")
readme.write_text(updated, encoding="utf-8")
