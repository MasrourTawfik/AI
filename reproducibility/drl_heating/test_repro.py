import numpy as np
from drl_heating_repro import Config, HeatingEnv, synthetic_process_outlet, train_surrogate, set_seed


def test_process_monotonicity():
    speed = np.array([3.0, 6.0])
    temp = np.array([650.0, 650.0])
    y = synthetic_process_outlet(speed, temp)
    assert y[0] > y[1], "Slower synthetic transfer should heat more in the transparent demo model"


def test_surrogate_quality():
    cfg = Config(n_surrogate_samples=1200, surrogate_epochs=120)
    set_seed(cfg.seed)
    model, metrics = train_surrogate(cfg)
    assert metrics["mae_C"] < 8.0
    env = HeatingEnv(model, cfg)
    state = env.reset(target=500.0)
    assert state.shape == (5,)
    assert np.isfinite(state).all()
