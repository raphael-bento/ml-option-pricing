"""Neural networks that learn the Black-Scholes pricing function."""

from option_pricing.black_scholes import bs_call_price, bs_put_price, scaled_call_price

__all__ = ["bs_call_price", "bs_put_price", "scaled_call_price"]
