"""Player data loading. Seed CSV ships with the repo; replace it with the
output of scripts/ingest_sackmann.py for ratings computed from real results.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "players_seed.csv"
SURFACES = ("Hard", "Clay", "Grass", "Carpet")


@dataclass
class Player:
    name: str
    tour: str  # ATP or WTA
    elo: float
    elo_hard: float
    elo_clay: float
    elo_grass: float
    serve_points_won: float   # spw, 0-1
    return_points_won: float  # rpw, 0-1

    def elo_on(self, surface: str) -> float:
        return {"Hard": self.elo_hard, "Clay": self.elo_clay,
                "Grass": self.elo_grass, "Carpet": self.elo}.get(surface, self.elo)


def load_players(path: Path = DEFAULT_CSV) -> list[Player]:
    players = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            players.append(Player(
                name=row["name"], tour=row["tour"],
                elo=float(row["elo"]), elo_hard=float(row["elo_hard"]),
                elo_clay=float(row["elo_clay"]), elo_grass=float(row["elo_grass"]),
                serve_points_won=float(row["serve_points_won"]),
                return_points_won=float(row["return_points_won"]),
            ))
    return players


def find_player(players: list[Player], query: str) -> Player:
    q = query.strip().lower()
    for p in players:
        if p.name.lower() == q:
            return p
    for p in players:  # last-name / substring match
        if q in p.name.lower():
            return p
    raise ValueError(f"Player not found: {query!r}. Add them to the players CSV or use --spw/--rpw custom mode.")
