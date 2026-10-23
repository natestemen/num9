"""Write the site's data files from the records: data/records.json, the
plays above 200 (the 236 record first) with their moves.

    python site/build_data.py <output directory>
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOVE = re.compile(r"\b\d@\d+:\d+:\d+:[0-3]\b")


def play(path, name):
    text = path.read_text()
    score = int(re.search(r"^score (\d+)", text, re.M).group(1))
    moves = MOVE.findall(text)
    top = max(int(m.split("@")[1].split(":")[0]) for m in moves)
    return {"name": name, "score": score, "top": top, "moves": " ".join(moves)}


def main(out):
    plays = [play(ROOT / "records/free-order-236.txt", "free-order-236")]
    index = (ROOT / "records/above-200/INDEX.md").read_text()
    for file in re.findall(r"\]\(([^)]+\.txt)\)", index):
        plays.append(play(ROOT / "records/above-200" / file, file.removesuffix(".txt")))
    data = Path(out) / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "records.json").write_text(json.dumps({"plays": plays}, separators=(",", ":")))
    print(f"{len(plays)} plays")


if __name__ == "__main__":
    main(sys.argv[1])
