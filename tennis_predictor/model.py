"""Pre-match prediction: blend surface Elo with the Markov point model.

Blend weights are the blueprint default: 0.45 Elo / 0.55 Markov.
The Markov leg usually carries better serve/return information; Elo
carries recency / form. Neither leg alone beat Pinnacle in published
benchmarks (XGBoost AUC 0.716 vs Pinnacle 0.742), so treat the output
as a probability estimate, not an edge by itself - compare it to a
market price before calling anything +EV.
"""
from dataclasses import dataclass

from .data import Player
from .elo import elo_win_prob
from . import markov


@dataclass
class Prediction:
    player_a: str
    player_b: str
    surface: str
    best_of: int
    p_a_serve_point: float   # P(A wins a point on A's serve)
    p_b_serve_point: float   # P(B wins a point on B's serve)
    hold_a: float            # P(A holds serve)
    hold_b: float            # P(B holds serve)
    set_prob_a: float        # Markov P(A wins a set)
    elo_prob_a: float
    markov_match_prob_a: float
    match_prob_a: float      # blended headline number

    def summary(self) -> str:
        fav = self.player_a if self.match_prob_a >= 0.5 else self.player_b
        p_fav = max(self.match_prob_a, 1 - self.match_prob_a)
        return (
            f"{self.player_a} vs {self.player_b} ({self.surface}, best of {self.best_of})\n"
            f"  Match win:  {self.player_a} {self.match_prob_a:.1%} | {self.player_b} {1 - self.match_prob_a:.1%}\n"
            f"  Set win (Markov): {self.player_a} {self.set_prob_a:.1%}\n"
            f"  Hold serve: {self.player_a} {self.hold_a:.1%} | {self.player_b} {self.hold_b:.1%}\n"
            f"  Legs: Elo {self.elo_prob_a:.1%} / Markov match {self.markov_match_prob_a:.1%} for {self.player_a}\n"
            f"  Better chance: {fav} ({p_fav:.1%})"
        )


def point_probs(a: Player, b: Player) -> tuple[float, float]:
    """Combine each player's serve/return rates into point probabilities.

    p_a_serve = mean(A's serve points won, points A wins on B's... no -
    points A wins when B returns = 1 - B's return points won, inverted):
    (spw_a + (1 - rpw_b)) / 2, and symmetrically for B.
    Two average players (spw .64, rpw .36 on ATP) correctly give .64.
    """
    p_a = (a.serve_points_won + (1.0 - b.return_points_won)) / 2.0
    p_b = (b.serve_points_won + (1.0 - a.return_points_won)) / 2.0
    return p_a, p_b


def predict(a: Player, b: Player, surface: str = "Hard", best_of: int = 3) -> Prediction:
    p_a, p_b = point_probs(a, b)
    # Small Elo-driven nudge to point probabilities: 400 Elo ~ +4 pts of serve dominance
    elo_diff_surf = a.elo_on(surface) - b.elo_on(surface)
    nudge = max(-0.03, min(0.03, elo_diff_surf / 400.0 * 0.04))
    p_a = max(0.05, min(0.95, p_a + nudge))
    p_b = max(0.05, min(0.95, p_b - nudge))

    set_p = markov.set_prob(p_a, p_b, first_server="A")
    markov_match = markov.match_prob(set_p, best_of=best_of)
    elo_p = elo_win_prob(a.elo_on(surface), b.elo_on(surface))
    blended = 0.45 * elo_p + 0.55 * markov_match
    return Prediction(
        player_a=a.name, player_b=b.name, surface=surface, best_of=best_of,
        p_a_serve_point=p_a, p_b_serve_point=p_b,
        hold_a=markov.game_prob(p_a), hold_b=markov.game_prob(p_b),
        set_prob_a=set_p, elo_prob_a=elo_p,
        markov_match_prob_a=markov_match, match_prob_a=blended,
    )


def ev_flag(model_prob: float, decimal_odds: float) -> tuple[float, bool]:
    """Expected value per unit staked and whether it is +EV."""
    ev = model_prob * decimal_odds - 1.0
    return ev, ev > 0
