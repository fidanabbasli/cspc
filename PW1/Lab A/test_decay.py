import numpy as np
import pytest
from decay import simulate, simulate_loop


def test_starts_at_N0():
    assert simulate(1000, 0.4)[0] == 1000


def test_rejects_negative_rate():
    with pytest.raises(ValueError):
        simulate(1000, -0.4)


def test_matches_law():
    N0 = 1000
    lam = 0.4
    dt = 0.05
    n_simulations = 200

    # seed=None yazırıq ki, hər dəfə həqiqətən fərqli təsadüfi simulyasiya yaransın
    results = [simulate(N0, lam, dt=dt, seed=None) for _ in range(n_simulations)]
    mean_decay = np.mean(results, axis=0)

    t = np.arange(len(mean_decay)) * dt
    expected = N0 * np.exp(-lam * t)

    assert mean_decay == pytest.approx(expected, rel=0.05)