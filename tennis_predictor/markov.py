"""Point -> game -> set -> match Markov (i.i.d. point) model.

Core maths from the blueprint: Barnett & Clarke / Klaassen & Magnus style.
Points are treated as i.i.d. with probability p that the server wins the
point. That assumption is not strictly true (Klaassen & Magnus, 2001),
but the deviations are small and this model is the industry-standard
baseline for live hold / set / match probabilities.
"""
from functools import lru_cache


def _clamp(p: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return max(lo, min(hi, p))


def game_prob(p: float) -> float:
    """Probability the server wins a service game, P(point) = p."""
    p = _clamp(p, 0.01, 0.99)
    q = 1.0 - p
    # Win 4-0, 4-1, 4-2
    straight = p**4 + 4 * p**4 * q + 10 * p**4 * q**2
    # Reach deuce (3-3), then win from deuce: p^2 / (p^2 + q^2)
    deuce = 20 * (p**3) * (q**3)
    win_from_deuce = (p * p) / (p * p + q * q)
    return straight + deuce * win_from_deuce


def live_game_prob(p: float, server_pts: int, returner_pts: int) -> float:
    """Live hold probability from a current point score in a game.

    server_pts / returner_pts are point counts (0,1,2,3+), not 15/30/40.
    Handles deuce / advantage states.
    """
    p = _clamp(p, 0.01, 0.99)
    q = 1.0 - p

    @lru_cache(maxsize=None)
    def rec(a: int, b: int) -> float:
        if a >= 4 and a - b >= 2:
            return 1.0
        if b >= 4 and b - a >= 2:
            return 0.0
        # Collapse deuce-zone states to (3,3), (4,3) adv server, (3,4) adv returner
        if a >= 3 and b >= 3:
            if a == b:
                return (p * p) / (p * p + q * q)
            if a == b + 1:  # advantage server: win next -> game, lose -> deuce
                deuce_p = (p * p) / (p * p + q * q)
                return p + q * deuce_p
            # advantage returner
            deuce_p = (p * p) / (p * p + q * q)
            return p * deuce_p
        return p * rec(a + 1, b) + q * rec(a, b + 1)

    return rec(int(server_pts), int(returner_pts))


def tiebreak_prob(p_a_serve: float, p_b_serve: float, a_pts: int = 0, b_pts: int = 0,
                  first_server: str = "A") -> float:
    """Probability A wins a tiebreak (first to 7, margin 2).

    Serve order: first server serves point 1, then 2 points each alternating.
    p_a_serve = prob A wins a point on A's serve; p_b_serve likewise for B.
    """
    pa = _clamp(p_a_serve, 0.01, 0.99)
    pb = _clamp(p_b_serve, 0.01, 0.99)

    def server_for(point_no: int) -> str:
        # point_no is 0-indexed total points already played
        if point_no == 0:
            return first_server
        block = (point_no - 1) // 2
        # after the first point, serve switches every 2 points
        if block % 2 == 0:
            return "B" if first_server == "A" else "A"
        return first_server

    @lru_cache(maxsize=None)
    def rec(a: int, b: int) -> float:
        if a >= 7 and a - b >= 2:
            return 1.0
        if b >= 7 and b - a >= 2:
            return 0.0
        if a >= 6 and b >= 6:
            # Tiebreak "deuce zone": use average point probability for A
            # (half the points on each serve) in the win-by-2 deuce formula.
            p_avg = (pa + (1.0 - pb)) / 2.0
            win_by_2 = (p_avg * p_avg) / (p_avg * p_avg + (1 - p_avg) ** 2)
            if a == b:
                return win_by_2
            srv_now = server_for(a + b)
            p_next_a = pa if srv_now == "A" else (1.0 - pb)
            if a == b + 1:  # A leads: win next point -> tiebreak, lose -> deuce
                return p_next_a + (1.0 - p_next_a) * win_by_2
            # B leads
            return p_next_a * win_by_2
        srv = server_for(a + b)
        p_point_for_a = pa if srv == "A" else (1.0 - pb)
        return p_point_for_a * rec(a + 1, b) + (1.0 - p_point_for_a) * rec(a, b + 1)

    return rec(int(a_pts), int(b_pts))


def set_prob(p_a_serve: float, p_b_serve: float, first_server: str = "A") -> float:
    """Probability A wins a set (first to 6, margin 2, tiebreak at 6-6)."""
    hold_a = game_prob(p_a_serve)
    hold_b = game_prob(p_b_serve)
    tb = tiebreak_prob(p_a_serve, p_b_serve, first_server=first_server)

    @lru_cache(maxsize=None)
    def rec(a_games: int, b_games: int, a_to_serve: bool) -> float:
        if a_games >= 6 and a_games - b_games >= 2:
            return 1.0
        if b_games >= 6 and b_games - a_games >= 2:
            return 0.0
        if a_games == 6 and b_games == 6:
            return tb
        if a_to_serve:
            # A holds -> A game, A broken -> B game; B serves next either way
            return hold_a * rec(a_games + 1, b_games, False) + (1 - hold_a) * rec(a_games, b_games + 1, False)
        else:
            # B serving: B holds with hold_b -> B game, else A game
            return (1 - hold_b) * rec(a_games + 1, b_games, True) + hold_b * rec(a_games, b_games + 1, True)

    return rec(0, 0, first_server == "A")


def match_prob(set_p: float, best_of: int = 3) -> float:
    """Probability of winning a best-of-3 / best-of-5 match given P(win a set)."""
    need = best_of // 2 + 1
    from math import comb
    total = 0.0
    for losses in range(need):
        # Win the deciding set after `losses` losses in the first need-1+losses sets
        total += comb(need - 1 + losses, losses) * (set_p ** need) * ((1 - set_p) ** losses)
    return total
