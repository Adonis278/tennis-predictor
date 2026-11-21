"""FastAPI service. Run: uvicorn tennis_predictor.app:app --reload

GET  /players
POST /predict   {"player_a": "Sinner", "player_b": "Alcaraz", "surface": "Clay", "best_of": 5}
POST /live-hold {"p": 0.64, "server_points": 2, "returner_points": 1}
"""
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise SystemExit("FastAPI is optional: pip install -r requirements.txt") from exc

from .data import load_players, find_player
from .model import predict
from .markov import live_game_prob

app = FastAPI(title="Tennis Predictor", version="0.1.0")
PLAYERS = load_players()


class PredictIn(BaseModel):
    player_a: str
    player_b: str
    surface: str = "Hard"
    best_of: int = 3


class LiveHoldIn(BaseModel):
    p: float
    server_points: int = 0
    returner_points: int = 0


@app.get("/players")
def players():
    return [{"name": p.name, "tour": p.tour, "elo": p.elo} for p in PLAYERS]


@app.post("/predict")
def predict_endpoint(body: PredictIn):
    a = find_player(PLAYERS, body.player_a)
    b = find_player(PLAYERS, body.player_b)
    pred = predict(a, b, surface=body.surface, best_of=body.best_of)
    return {
        "match_win_prob_a": round(pred.match_prob_a, 4),
        "match_win_prob_b": round(1 - pred.match_prob_a, 4),
        "set_win_prob_a": round(pred.set_prob_a, 4),
        "hold_prob_a": round(pred.hold_a, 4),
        "hold_prob_b": round(pred.hold_b, 4),
        "elo_prob_a": round(pred.elo_prob_a, 4),
        "markov_match_prob_a": round(pred.markov_match_prob_a, 4),
        "better_chance": pred.player_a if pred.match_prob_a >= 0.5 else pred.player_b,
    }


@app.post("/live-hold")
def live_hold_endpoint(body: LiveHoldIn):
    return {"hold_probability": round(live_game_prob(body.p, body.server_points, body.returner_points), 4)}
