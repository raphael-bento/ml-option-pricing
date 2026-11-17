import numpy as np
import pytest

from option_pricing import bs_call_price, bs_put_price, scaled_call_price


def test_bs_call_price_textbook_value():
    # Hull, Options Futures and Other Derivatives: S=42, K=40, T=0.5, r=10%, sigma=20% -> 4.76
    assert bs_call_price(42, 40, 0.5, 0.10, 0.20) == pytest.approx(4.7594, abs=1e-3)


def test_bs_put_price_textbook_value():
    assert bs_put_price(42, 40, 0.5, 0.10, 0.20) == pytest.approx(0.8086, abs=1e-3)


@pytest.mark.parametrize("s", [20.0, 40.0, 60.0])
def test_put_call_parity_holds(s):
    k, tau, r, sigma = 40.0, 0.75, 0.05, 0.3
    lhs = bs_call_price(s, k, tau, r, sigma) - bs_put_price(s, k, tau, r, sigma)
    assert lhs == pytest.approx(s - k * np.exp(-r * tau))


def test_scaled_call_price_is_homogeneous_in_strike():
    m, tau, r, sigma = 1.2, 0.5, 0.03, 0.25
    assert scaled_call_price(m, tau, r, sigma) == pytest.approx(
        bs_call_price(120.0, 100.0, tau, r, sigma) / 100.0
    )


def test_call_price_respects_no_arbitrage_bounds():
    rng = np.random.default_rng(42)
    m = rng.uniform(0.4, 1.6, 1_000)
    tau = rng.uniform(0.2, 1.1, 1_000)
    r = rng.uniform(0.02, 0.10, 1_000)
    sigma = rng.uniform(0.01, 1.00, 1_000)
    pi = scaled_call_price(m, tau, r, sigma)
    lower = np.maximum(m - np.exp(-r * tau), 0.0)
    assert np.all(pi >= lower - 1e-12)
    assert np.all(pi <= m + 1e-12)


@pytest.mark.parametrize(
    "args", [(0, 1, 1, 0.05, 0.2), (1, 0, 1, 0.05, 0.2), (1, 1, 0, 0.05, 0.2), (1, 1, 1, 0.05, 0)]
)
def test_invalid_inputs_raise(args):
    with pytest.raises(ValueError):
        bs_call_price(*args)
