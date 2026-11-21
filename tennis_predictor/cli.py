"""Command line interface.

Examples:
  python -m tennis_predictor.cli predict Sinner Alcaraz --surface Clay --best-of 5
  python -m tennis_predictor.cli predict Swiatek Sabalenka --surface Clay
  python -m tennis_predictor.cli live-hold --p 0.64 --score 30-15
  python -m tennis_predictor.cli list
"""
import argparse

from .data import load_players, find_player
from .model import predict, ev_flag
from .markov import live_game_prob

POINTS = {"0": 0, "15": 1, "30": 2, "40": 3}


def main() -> None:
    ap = argparse.ArgumentParser(prog="tennis-predictor")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="Pre-match winner / set / hold probabilities")
    p.add_argument("player_a")
    p.add_argument("player_b")
    p.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass", "Carpet"])
    p.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    p.add_argument("--odds-a", type=float, default=None, help="Decimal odds for player A, to flag +EV")

    lh = sub.add_parser("live-hold", help="Live hold probability in the current service game")
    lh.add_argument("--p", type=float, required=True, help="Server's probability of winning a point on serve, 0-1")
    lh.add_argument("--score", default="0-0", help="Current point score, e.g. 30-15, 40-40 (deuce), AD-40")

    sub.add_parser("list", help="List players in the dataset")
    args = ap.parse_args()

    if args.cmd == "list":
        for pl in load_players():
            print(f"{pl.tour}  {pl.name:<28} Elo {pl.elo:.0f}  spw {pl.serve_points_won:.2f}  rpw {pl.return_points_won:.2f}")
        return

    if args.cmd == "live-hold":
        s, r = args.score.upper().split("-")
        if s == "AD":
            sp, rp = 4, 3
        elif r == "AD":
            sp, rp = 3, 4
        else:
            sp, rp = POINTS[s], POINTS[r]
        prob = live_game_prob(args.p, sp, rp)
        print(f"Server point win p={args.p:.2f}, score {args.score}: hold probability {prob:.1%}")
        return

    players = load_players()
    a = find_player(players, args.player_a)
    b = find_player(players, args.player_b)
    pred = predict(a, b, surface=args.surface, best_of=args.best_of)
    print(pred.summary())
    if args.odds_a:
        ev, plus = ev_flag(pred.match_prob_a, args.odds_a)
        print(f"  Odds check on {a.name} @ {args.odds_a}: EV {ev:+.1%} per unit -> {'+EV' if plus else 'not +EV'}")


if __name__ == "__main__":
    main()
