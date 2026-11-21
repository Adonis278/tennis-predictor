# Tennis Predictor

Predicts who has the better chance in a singles tennis match: pre-match
match winner, set winner, and live service-game hold probability.
Covers ATP and WTA out of the box, and ITF / UTR players once you add
them to the player data (see below).

Built from the Tennis Analytics Prediction Blueprint (Nov 2026): free
and open-source data first, no paid API required to run it.

## How it works

Two legs, blended 45 / 55:

1. **Surface Elo.** Each player has an overall Elo and a Hard / Clay /
   Grass Elo. `P = 1 / (1 + 10^(-diff/400))`.
2. **Point-level Markov model.** Each player's serve points won (spw)
   and return points won (rpw) combine into a point probability on each
   player's serve. From there a dynamic program rolls points up to
   games (hold %), games to sets (with a tiebreak), and sets to a
   best-of-3 / best-of-5 match. This is the Barnett & Clarke /
   Klaassen & Magnus i.i.d.-point model.

The Markov leg also powers the live tool: give it the server's point
win probability and the current point score, and it returns the live
hold probability for that service game.

Honest benchmark context: published Elo models reach about 70%
accuracy, a calibrated XGBoost reached AUC 0.716 vs Pinnacle's 0.742.
This tool estimates probabilities; it does not beat the market by
itself. Compare its number to a price with `--odds-a` before calling
anything +EV.

## Quick start

No dependencies for the core tool (Python 3.10+, standard library only).

```bash
python -m tennis_predictor.cli list
python -m tennis_predictor.cli predict Sinner Alcaraz --surface Clay --best-of 5
python -m tennis_predictor.cli predict Swiatek Sabalenka --surface Clay
python -m tennis_predictor.cli predict Alcaraz Djokovic --surface Hard --odds-a 1.80
python -m tennis_predictor.cli live-hold --p 0.64 --score 30-15
```

Example output:

```
Jannik Sinner vs Carlos Alcaraz (Clay, best of 5)
  Match win:  Jannik Sinner 42.4% | Carlos Alcaraz 57.6%
  Set win (Markov): Jannik Sinner 47.0%
  Hold serve: Jannik Sinner 76.7% | Carlos Alcaraz 78.4%
  Legs: Elo 40.1% / Markov match 44.3% for Jannik Sinner
  Better chance: Carlos Alcaraz (57.6%)
```

(Exact percentages come from the seed data; regenerate the data for
current ratings, see below.)

## API service (optional)

```bash
pip install -r requirements.txt
uvicorn tennis_predictor.app:app --reload
# POST /predict   {"player_a": "Sinner", "player_b": "Alcaraz", "surface": "Clay", "best_of": 5}
# POST /live-hold {"p": 0.64, "server_points": 2, "returner_points": 1}
# GET  /players
```

## Data

`data/players_seed.csv` ships with 22 top ATP / WTA players. The Elo
and serve/return figures in the seed file are approximate, rounded
seed values so the tool runs out of the box. Do not bet on seed
numbers.

Regenerate real ratings from match history with the Sackmann archive
mirror. The original Jeff Sackmann GitHub repos return 404 as of
Oct/Nov 2026; the mirror is:

- https://github.com/Aneeshers/tennis-sackmann-archive
- Hugging Face: Aneeshers/tennis-sackmann-archive

```bash
git clone https://github.com/Aneeshers/tennis-sackmann-archive /tmp/sackmann
python scripts/ingest_sackmann.py --src /tmp/sackmann --tour atp --since 2023 --out data/players_atp.csv
python scripts/ingest_sackmann.py --src /tmp/sackmann --tour wta --since 2023 --out data/players_wta.csv
```

Known data gaps (from the blueprint research):

- Sackmann serve stats: tour level from 1991, Challenger from 2008,
  qualifying from 2011. Futures / ITF files are results-only, so ITF
  players get Elo and league-average serve/return rates.
- Tennis-Data.co.uk has closing odds (Pinnacle, Bet365) but no serve
  stats, main tour only.
- UTR: there is no maintained open-source UTR scraper and no free
  UTR dataset. The official UTR Engage API is per-user OAuth,
  commercial approval, and prohibits bulk use and UTR-like
  predictions. Add UTR players manually to the CSV (name, tour,
  Elo, surface Elos, spw, rpw) to predict their matches.

Sackmann data is CC BY-NC-SA 4.0 (non-commercial).

## Tests

```bash
pip install pytest
pytest
```

Checks include the model symmetries (equal players give 50%) and the
reference value from the literature: a server winning 63% of points
holds about 79.4% of the time.

## Layout

```
tennis_predictor/
  markov.py   point -> game -> set -> match, plus live hold
  elo.py      Elo win probability and updates
  model.py    blend, prediction object, +EV check
  data.py     player CSV loading
  cli.py      command line interface
  app.py      FastAPI service (optional)
scripts/
  ingest_sackmann.py   rebuild ratings from the Sackmann mirror
data/
  players_seed.csv     approximate seed ratings, 22 players
tests/
  test_markov.py
```

## Roadmap (from the blueprint)

- Week 1-2: backtest the blend against Tennis-Data.co.uk closing odds
- Week 3: live polling with Live Tennis API (free tier 100 calls/day;
  Basic $9.99/mo) feeding the live-hold model
- Week 4: ITF / UTR manual-entry flow, prediction logging, calibration
  report (log-loss and Brier score vs market)
