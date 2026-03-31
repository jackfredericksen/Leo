"""
Kelly Criterion position sizing.

For a bet with:
  p = estimated true probability of YES resolving
  b = net odds (payout per $1 staked, i.e. 1/price - 1)

Full Kelly fraction = (p*b - (1-p)) / b
  = (p - (1-p)/b)
  = edge / odds

We use fractional Kelly (default 25%) to reduce variance while
preserving most of the long-run growth rate advantage.

Usage:
  sizer = KellySizer(bankroll=1000, fraction=0.25)
  size = sizer.size(true_prob=0.65, market_price=0.50, max_size=100)
"""

import logging

logger = logging.getLogger(__name__)


class KellySizer:
    def __init__(self, bankroll: float, fraction: float = 0.25, max_pct_of_bankroll: float = 0.10):
        """
        Args:
            bankroll: Total capital available for betting.
            fraction: Kelly fraction (1.0 = full Kelly, 0.25 = quarter Kelly).
            max_pct_of_bankroll: Hard cap — never bet more than this % of bankroll.
        """
        self.bankroll = bankroll
        self.fraction = fraction
        self.max_pct_of_bankroll = max_pct_of_bankroll

    def size(self, true_prob: float, market_price: float, max_size: float = float("inf")) -> float:
        """
        Calculate optimal bet size in USD.

        Args:
            true_prob: Your estimate of the true probability of YES resolving.
            market_price: Current YES price (0–1).
            max_size: External cap on position size.

        Returns:
            Recommended bet size in USD, or 0 if no edge.
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0

        # Net odds: if you pay `price` and win, you receive 1.0, net = (1-price)/price
        b = (1.0 - market_price) / market_price
        q = 1.0 - true_prob

        kelly_fraction = (true_prob * b - q) / b

        if kelly_fraction <= 0:
            logger.debug(f"No edge: true_prob={true_prob:.3f} price={market_price:.3f}")
            return 0.0

        fractional = kelly_fraction * self.fraction
        hard_cap = self.bankroll * self.max_pct_of_bankroll
        recommended = self.bankroll * fractional

        size = min(recommended, hard_cap, max_size)
        logger.debug(
            f"Kelly size: true_prob={true_prob:.3f} price={market_price:.3f} "
            f"edge={kelly_fraction:.3f} fractional={fractional:.3f} "
            f"→ ${size:.2f}"
        )
        return round(size, 2)

    def update_bankroll(self, new_bankroll: float):
        self.bankroll = new_bankroll


def implied_edge(true_prob: float, market_price: float, fee_pct: float = 0.0) -> float:
    """
    Return the expected value edge as a fraction of stake.
    Positive = bet has positive expected value.
    """
    net_payout = (1.0 - market_price) / market_price
    ev = true_prob * net_payout - (1.0 - true_prob) - fee_pct
    return ev / 1.0  # normalised to stake
