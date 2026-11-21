import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tennis_predictor.markov import game_prob, live_game_prob, set_prob, match_prob
from tennis_predictor.data import load_players, find_player
from tennis_predictor.model import predict


def test_game_symmetry():
    assert abs(game_prob(0.5) - 0.5) < 1e-9


def test_game_reference_value():
    # Blueprint reference: p = 0.63 on serve -> hold about 79.4%
    assert abs(game_prob(0.63) - 0.794) < 0.01


def test_live_game_from_love_and_deuce():
    assert live_game_prob(0.64, 0, 0) == game_prob(0.64)
    assert 0 < live_game_prob(0.64, 3, 3) < 1
    assert live_game_prob(0.64, 3, 0) > live_game_prob(0.64, 0, 3)


def test_set_and_match_symmetry():
    assert abs(set_prob(0.64, 0.64) - 0.5) < 0.02
    assert abs(match_prob(0.5, best_of=3) - 0.5) < 1e-9
    assert match_prob(0.6, best_of=5) > match_prob(0.6, best_of=3) > 0.6


def test_predict_favourite():
    players = load_players()
    pred = predict(find_player(players, "Swiatek"), find_player(players, "Sabalenka"),
                   surface="Clay", best_of=3)
    assert pred.match_prob_a > 0.5  # Swiatek's clay Elo edge should make her favourite
    assert 0 < pred.set_prob_a < 1
