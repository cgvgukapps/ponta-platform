"""
fetch_reference_data.py — download open reference datasets from figshare, outside every repository.

Reference data is for measuring Ponta's engines, never product data, and it is never copied into a
repository (the blueprint's rule for sample data). This tool downloads a named dataset's files from
figshare's public API into ``--dest`` (default ``~/ponta-reference-data/<name>``), checks each file
against the MD5 that figshare publishes, and writes ``SOURCE.json`` beside them (title, licence,
attribution, file list). Files already present with the right checksum are skipped.

Datasets:

- ``idsse``: Bassek, Rein, Weber and Memmert (2025), *An integrated dataset of spatiotemporal and
  event data in elite soccer*, Scientific Data 12, 195. Seven German league matches with official
  match information, events and TRACAB positions. CC BY 4.0; credit the Deutsche Fußball Liga (DFL)
  and cite the paper.
- ``wyscout``: Pappalardo et al. (2019), *A public data set of spatio-temporal match events in
  soccer competitions*, Scientific Data 6, 236. Wyscout event logs for seven competitions of 2017/18.
  CC BY 4.0; cite the paper. The players file holds real names: Ponta's readers never read them.

    python tools/fetch_reference_data.py idsse
    python tools/fetch_reference_data.py wyscout --dest ~/ponta-reference-data/wyscout-opendata
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from matrix_engine.sample_data import repo_root

API = "https://api.figshare.com/v2/articles/{}"
#: Generic client name for figshare's API (no personal details).
HEADERS = {"User-Agent": "ponta-reference-data-fetch/1.0"}

DATASETS: dict[str, dict] = {
    "idsse": {
        "articles": [28196177],
        "attribution": "Data: Deutsche Fußball Liga (DFL). Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). "
                       "An integrated dataset of spatiotemporal and event data in elite soccer. Scientific Data 12, 195. "
                       "https://doi.org/10.1038/s41597-025-04505-y",
    },
    "wyscout": {
        # Events, matches, competitions, teams, players and the two name mappings (no PlayeRank, no code).
        "articles": [7770599, 7770422, 7765316, 7765310, 7765196, 11743836, 11743818],
        "attribution": "Pappalardo, L., Cintia, P., Rossi, A. et al. (2019). A public data set of spatio-temporal match "
                       "events in soccer competitions. Scientific Data 6, 236. https://doi.org/10.1038/s41597-019-0247-7",
    },
}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        return r.read()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(path)


def fetch(name: str, dest: Path) -> dict:
    spec = DATASETS[name]
    dest.mkdir(parents=True, exist_ok=True)
    files, licences, titles = [], set(), []
    for article in spec["articles"]:
        meta = json.loads(_get(API.format(article)))
        titles.append(meta["title"])
        licences.add(meta.get("license", {}).get("name"))
        for f in meta.get("files", []):
            path = dest / f["name"]
            if path.exists() and _md5(path) == f["computed_md5"]:
                status = "present"
            else:
                print(f"downloading {f['name']} ({f['size'] / 1e6:.1f} MB)", flush=True)
                _download(f["download_url"], path)
                if _md5(path) != f["computed_md5"]:
                    path.unlink()
                    sys.exit(f"checksum mismatch for {f['name']}; file removed")
                status = "downloaded"
            files.append({"article": article, "name": f["name"], "bytes": f["size"], "md5": f["computed_md5"], "status": status})
    source = {"dataset": name, "titles": titles, "licences": sorted(x for x in licences if x), "attribution": spec["attribution"],
              "figshare_articles": spec["articles"], "files": [{k: v for k, v in f.items() if k != "status"} for f in files]}
    (dest / "SOURCE.json").write_text(json.dumps(source, indent=1, ensure_ascii=False) + "\n")
    return {"dest": str(dest), "files": len(files), "downloaded": sum(f["status"] == "downloaded" for f in files)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--dest", type=Path, help="target folder (default ~/ponta-reference-data/<dataset>)")
    args = ap.parse_args()
    dest = (args.dest or Path.home() / "ponta-reference-data" / args.dataset).expanduser().resolve()
    root = repo_root().resolve()
    if dest == root or root in dest.parents:
        sys.exit("refusing to download reference data inside the repository")
    print(json.dumps(fetch(args.dataset, dest)))


if __name__ == "__main__":
    main()
