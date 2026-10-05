from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque, Dict, List, Tuple
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.optim import Adam


def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Bounds:
    speed_min: float = 2.0
    speed_max: float = 7.0
    zone_temp_min: float = 560.0
    zone_temp_max: float = 720.0
    target_min: float = 430.0
    target_max: float = 540.0


@dataclass
class Config:
    seed: int = 7
    n_surrogate_samples: int = 5000
    surrogate_epochs: int = 180
    surrogate_lr: float = 2e-3
    dqn_episodes: int = 140
    dqn_batch_size: int = 64
    dqn_lr: float = 8e-4
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.988
    target_sync: int = 20
    replay_capacity: int = 15000
    max_steps: int = 35
    speed_step: float = 0.25
    temp_step: float = 5.0
    tolerance: float = 3.0


BOUNDS = Bounds()


def synthetic_process_outlet(speed: np.ndarray, zone_temp: np.ndarray, noise_std: float = 0.0, rng=None) -> np.ndarray:
    """Synthetic glass-heating process.

    This is intentionally NOT the proprietary/experimental furnace model from the paper.
    It is a transparent surrogate data generator used to demonstrate the DQN + ANN workflow.
    """
    ambient = 25.0
    residence_factor = 1.0 - np.exp(-8.0 / np.maximum(speed, 1e-3))
    nonlinear_gain = 0.82 + 0.035 * np.sin((zone_temp - 560.0) / 35.0)
    outlet = ambient + (zone_temp - ambient) * residence_factor * nonlinear_gain
    outlet -= 1.2 * (speed - 4.5) ** 2
    if noise_std > 0:
        if rng is None:
            rng = np.random.default_rng()
        outlet = outlet + rng.normal(0.0, noise_std, size=np.shape(outlet))
    return outlet


def normalize_inputs(speed: np.ndarray, temp: np.ndarray) -> np.ndarray:
    s = (speed - BOUNDS.speed_min) / (BOUNDS.speed_max - BOUNDS.speed_min)
    t = (temp - BOUNDS.zone_temp_min) / (BOUNDS.zone_temp_max - BOUNDS.zone_temp_min)
    return np.column_stack([s, t]).astype(np.float32)


def normalize_target(y: np.ndarray) -> np.ndarray:
    lo, hi = 350.0, 680.0
    return ((y - lo) / (hi - lo)).astype(np.float32)


def denormalize_target(y: np.ndarray) -> np.ndarray:
    lo, hi = 350.0, 680.0
    return y * (hi - lo) + lo


class SurrogateANN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_surrogate(cfg: Config, device: str = "cpu") -> Tuple[SurrogateANN, Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)
    speed = rng.uniform(BOUNDS.speed_min, BOUNDS.speed_max, cfg.n_surrogate_samples)
    temp = rng.uniform(BOUNDS.zone_temp_min, BOUNDS.zone_temp_max, cfg.n_surrogate_samples)
    y = synthetic_process_outlet(speed, temp, noise_std=1.0, rng=rng)
    X = normalize_inputs(speed, temp)
    yn = normalize_target(y)[:, None]

    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]

    Xtr = torch.tensor(X[tr], device=device)
    ytr = torch.tensor(yn[tr], device=device)
    Xte = torch.tensor(X[te], device=device)
    yte = torch.tensor(yn[te], device=device)

    model = SurrogateANN().to(device)
    opt = Adam(model.parameters(), lr=cfg.surrogate_lr)
    loss_fn = nn.MSELoss()

    for _ in range(cfg.surrogate_epochs):
        model.train()
        pred = model(Xtr)
        loss = loss_fn(pred, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_te = denormalize_target(model(Xte).cpu().numpy().ravel())
        y_te = denormalize_target(yte.cpu().numpy().ravel())
    mae = float(np.mean(np.abs(pred_te - y_te)))
    rmse = float(np.sqrt(np.mean((pred_te - y_te) ** 2)))
    return model, {"mae_C": mae, "rmse_C": rmse}


class HeatingEnv:
    ACTIONS = {
        0: (-1, 0),
        1: (+1, 0),
        2: (0, -1),
        3: (0, +1),
        4: (0, 0),
    }

    def __init__(self, surrogate: SurrogateANN, cfg: Config, device: str = "cpu") -> None:
        self.surrogate = surrogate
        self.cfg = cfg
        self.device = device
        self.rng = np.random.default_rng(cfg.seed + 101)
        self.speed = 4.5
        self.zone_temp = 640.0
        self.target = 500.0
        self.steps = 0

    def _predict(self, speed: float, temp: float) -> float:
        x = torch.tensor(normalize_inputs(np.array([speed]), np.array([temp])), device=self.device)
        self.surrogate.eval()
        with torch.no_grad():
            yn = self.surrogate(x).cpu().numpy().ravel()[0]
        return float(denormalize_target(np.array([yn]))[0])

    def _state(self) -> np.ndarray:
        pred = self._predict(self.speed, self.zone_temp)
        return np.array([
            (self.target - BOUNDS.target_min) / (BOUNDS.target_max - BOUNDS.target_min),
            (pred - 350.0) / 330.0,
            (self.speed - BOUNDS.speed_min) / (BOUNDS.speed_max - BOUNDS.speed_min),
            (self.zone_temp - BOUNDS.zone_temp_min) / (BOUNDS.zone_temp_max - BOUNDS.zone_temp_min),
            np.clip((self.target - pred) / 150.0, -1.0, 1.0),
        ], dtype=np.float32)

    def reset(self, target: float | None = None) -> np.ndarray:
        self.target = float(target if target is not None else self.rng.uniform(BOUNDS.target_min, BOUNDS.target_max))
        self.speed = float(self.rng.uniform(3.0, 6.0))
        self.zone_temp = float(self.rng.uniform(590.0, 690.0))
        self.steps = 0
        return self._state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        prev_pred = self._predict(self.speed, self.zone_temp)
        prev_error = abs(self.target - prev_pred)
        ds, dt = self.ACTIONS[int(action)]
        self.speed = float(np.clip(self.speed + ds * self.cfg.speed_step, BOUNDS.speed_min, BOUNDS.speed_max))
        self.zone_temp = float(np.clip(self.zone_temp + dt * self.cfg.temp_step, BOUNDS.zone_temp_min, BOUNDS.zone_temp_max))
        self.steps += 1
        pred = self._predict(self.speed, self.zone_temp)
        error = abs(self.target - pred)
        improvement = prev_error - error
        reward = improvement / 4.0 - error / 40.0 - 0.01
        done = error <= self.cfg.tolerance or self.steps >= self.cfg.max_steps
        if error <= self.cfg.tolerance:
            reward += 5.0
        info = {
            "target": self.target,
            "predicted_outlet": pred,
            "abs_error": error,
            "speed": self.speed,
            "zone_temp": self.zone_temp,
        }
        return self._state(), float(reward), bool(done), info


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 5, n_actions: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Replay:
    def __init__(self, capacity: int) -> None:
        self.buf: Deque = deque(maxlen=capacity)

    def add(self, item) -> None:
        self.buf.append(item)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(r, dtype=np.float32), np.array(ns), np.array(d, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.buf)


def train_dqn(env: HeatingEnv, cfg: Config, device: str = "cpu") -> Tuple[QNetwork, Dict[str, List[float]]]:
    q = QNetwork().to(device)
    target = QNetwork().to(device)
    target.load_state_dict(q.state_dict())
    opt = Adam(q.parameters(), lr=cfg.dqn_lr)
    replay = Replay(cfg.replay_capacity)
    eps = cfg.epsilon_start
    rewards, final_errors = [], []

    for ep in range(cfg.dqn_episodes):
        state = env.reset()
        total = 0.0
        last_info = {"abs_error": 999.0}
        for _ in range(cfg.max_steps):
            if random.random() < eps:
                action = random.randrange(5)
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                    action = int(qs.argmax(dim=1).item())
            next_state, reward, done, info = env.step(action)
            replay.add((state, action, reward, next_state, done))
            state = next_state
            total += reward
            last_info = info

            if len(replay) >= cfg.dqn_batch_size:
                s, a, r, ns, d = replay.sample(cfg.dqn_batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
                r_t = torch.tensor(r, dtype=torch.float32, device=device)
                ns_t = torch.tensor(ns, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device)

                qsa = q(s_t).gather(1, a_t).squeeze(1)
                with torch.no_grad():
                    next_q = target(ns_t).max(dim=1).values
                    y = r_t + cfg.gamma * (1.0 - d_t) * next_q
                loss = nn.functional.smooth_l1_loss(qsa, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

            if done:
                break
        rewards.append(total)
        final_errors.append(float(last_info["abs_error"]))
        eps = max(cfg.epsilon_end, eps * cfg.epsilon_decay)
        if (ep + 1) % cfg.target_sync == 0:
            target.load_state_dict(q.state_dict())

    return q, {"episode_reward": rewards, "final_abs_error_C": final_errors}


def evaluate(env: HeatingEnv, q: QNetwork, targets: List[float], device: str = "cpu") -> List[Dict[str, float]]:
    results = []
    for t in targets:
        state = env.reset(target=t)
        info = None
        for _ in range(env.cfg.max_steps):
            with torch.no_grad():
                action = int(q(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).argmax(dim=1).item())
            state, _, done, info = env.step(action)
            if done:
                break
        assert info is not None
        results.append(info)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Open conceptual reproduction of the DRL heating-process decision workflow")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.episodes is not None:
        cfg.dqn_episodes = args.episodes
    set_seed(cfg.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    surrogate, metrics = train_surrogate(cfg, device)
    env = HeatingEnv(surrogate, cfg, device)
    q, hist = train_dqn(env, cfg, device)
    targets = [440.0, 460.0, 480.0, 500.0, 520.0, 535.0]
    eval_results = evaluate(env, q, targets, device)

    torch.save(surrogate.state_dict(), args.out / "surrogate_ann.pt")
    torch.save(q.state_dict(), args.out / "dqn_agent.pt")
    payload = {
        "config": asdict(cfg),
        "surrogate_metrics": metrics,
        "evaluation": eval_results,
        "last_50_mean_abs_error_C": float(np.mean(hist["final_abs_error_C"][-50:])),
        "last_50_success_rate": float(np.mean(np.array(hist["final_abs_error_C"][-50:]) <= cfg.tolerance)),
        "note": "Synthetic conceptual reproduction; not the original industrial dataset or author-supplied code from the paper.",
    }
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
