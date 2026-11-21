"""Rebuild player ratings from real match data.

Source: the Sackmann archive mirror (the original JeffSackmann/tennis_atp
and tennis_wta repos return 404 as of Oct/Nov 2026):
  https://github.com/Aneeshers/tennis-sackmann-archive
  Hugging Face mirror: Aneeshers/tennis-sackmann-archive

Usage:
  git clone https://github.com/Aneeshers/tennis-sackmann-archive /tmp/sackmann
  python scripts/ingest_sackmann.py --src /tmp/sackmann --tour atp --since 2023 \
      --out data/players_atp.csv

Computes, per player, from matches since --since:
  - overall + per-surface Elo (K=32, processed in date order)
  - serve points won  = (w_1stWon + w_2ndWon + loser equivalent) / svpt
  - return points won = points won when returning / opponent svpt
Only matches with serve stats count toward spw/rpw (tour 1991+, Challenger
2008+, qualifying 2011+ in the Sackmann schema; Futures files are
results-only, so ITF players get Elo but league-average spw/rpw).
Output CSV matches the schema of data/players_seed.csv.
"""
import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tennis_predictor.elo import update_elo  # noqa: E402

LEAGUE_AVG = {"atp": (0.64, 0.36), "wta": (0.58, 0.42)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to the cloned sackmann archive")
    ap.add_argument("--tour", choices=["atp", "wta"], default="atp")
    ap.add_argument("--since", type=int, default=2023)
    ap.add_argument("--min-matches", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.src) / args.tour / f"{args.tour}_matches_*.csv")))
    files = [f for f in files if int(Path(f).stem.rsplit("_", 1)[-1]) >= args.since]
    if not files:
        raise SystemExit(f"No match files found under {args.src}/{args.tour}")

    elo, elo_surf = defaultdict(lambda: 2000.0), defaultdict(lambda: 2000.0)
    serve_won, serve_tot = defaultdict(int), defaultdict(int)
    ret_won, ret_tot = defaultdict(int), defaultdict(int)
    played = defaultdict(int)
    names = {}

    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                w, l = row["winner_name"], row["loser_name"]
                names[w], names[l] = w, l
                surf = row.get("surface") or "Hard"
                elo[w], elo[l] = update_elo(elo[w], elo[l])
                key_w, key_l = (w, surf), (l, surf)
                elo_surf[key_w], elo_surf[key_l] = update_elo(elo_surf[key_w], elo_surf[key_l])
                played[w] += 1
                played[l] += 1
                try:
                    w_sv = int(row["w_svpt"]); l_sv = int(row["l_svpt"])
                    w_pts = int(row["w_1stWon"]) + int(row["w_2ndWon"])
                    l_pts = int(row["l_1stWon"]) + int(row["l_2ndWon"])
                except (KeyError, ValueError, TypeError):
                    continue
                if w_sv and l_sv:
                    serve_won[w] += w_pts; serve_tot[w] += w_sv
                    serve_won[l] += l_pts; serve_tot[l] += l_sv
                    ret_won[w] += l_sv - l_pts; ret_tot[w] += l_sv
                    ret_won[l] += w_sv - w_pts; ret_tot[l] += w_sv

    avg_spw, avg_rpw = LEAGUE_AVG[args.tour]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["name", "tour", "elo", "elo_hard", "elo_clay", "elo_grass",
                      "serve_points_won", "return_points_won"])
        for name in sorted(names, key=lambda n: -elo[n]):
            if played[name] < args.min_matches:
                continue
            spw = serve_won[name] / serve_tot[name] if serve_tot[name] >= 200 else avg_spw
            rpw = ret_won[name] / ret_tot[name] if ret_tot[name] >= 200 else avg_rpw
            out.writerow([name, args.tour.upper(), f"{elo[name]:.0f}",
                          f"{elo_surf[(name, 'Hard')]:.0f}", f"{elo_surf[(name, 'Clay')]:.0f}",
                          f"{elo_surf[(name, 'Grass')]:.0f}", f"{spw:.3f}", f"{rpw:.3f}"])
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
