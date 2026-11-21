"""Surface Elo helpers."""


def elo_win_prob(elo_a: float, elo_b: float) -> float:
    """Standard Elo expected score for A vs B."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b) / 400.0))


def update_elo(elo_winner: float, elo_loser: float, k: float = 32.0):
    """One Elo update; returns (new_winner, new_loser)."""
    exp_w = elo_win_prob(elo_winner, elo_loser)
    delta = k * (1.0 - exp_w)
    return elo_winner + delta, elo_loser - delta
