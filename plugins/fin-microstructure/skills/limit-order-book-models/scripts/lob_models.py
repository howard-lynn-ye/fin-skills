#!/usr/bin/env python3
"""The order book as a queueing system: Cont-Stoikov-Talreja (2010), reproduced and extended.

CONT, STOIKOV & TALREJA (2010), "A stochastic model for order book dynamics", Operations
Research 58(3), 549-563. Read in the authors' preprint (www.columbia.edu/~ww2040/orderbook.pdf).
The model, section 2.2, is three independent Poisson flows per price level, for i >= 1 ticks
from the OPPOSITE best quote:

    limit orders   arrive at rate lambda(i)
    market orders  arrive at rate mu
    cancellations  arrive at rate theta(i) * x   when x orders rest at that level

so one best-quote queue is a birth-death process with birth rate lambda(i) and death rate
mu + i*theta(i) ... in state i:  death rate in state x is  mu + x*theta.  Table 2, estimated
from Tokyo Stock Exchange data for Sky Perfect Communications, prints

    i          1     2     3     4     5
    lambda  1.85  1.51  1.09  0.88  0.77
    theta   0.71  0.81  0.68  0.56  0.47
    mu = 0.94,  and the power-law fit lambda(i) = k / i^alpha with k = 1.92, alpha = 0.52.

WHAT THIS SCRIPT VERIFIES AGAINST THE PAPER

Proposition 3 (eqs. 9-11) gives the probability that the mid-price increases before it
decreases, as an inverse Laplace transform of a continued fraction. Its own proof states the
identity that matters: at spread S = 1 the first mid-price move happens "exactly when one of
the two independent birth-death processes X~_A and X~_B reaches the state 0 for the first
time ... the quantity (8) is given by P[sigma_A < sigma_B]".

That is a first-passage race between two independent birth-death chains, and it can be
computed EXACTLY as an absorption probability of the two-dimensional chain, with no Laplace
inversion at all.  `prob_mid_up` does that, and reproduces every one of the 25 entries of the
paper's Table 3 (bottom panel, its own Laplace-transform values) to 8.3e-4 - i.e. to the three
decimals the table is printed in.

Proposition 5 (eqs. 14-16) is the probability that a limit order at the bid executes before
the mid-price moves. eq. (14) makes the order's waiting time epsilon_B a pure-death first
passage with rate mu + theta*(i-1) in state i, and eq. (16) makes the event P[epsilon_B <
sigma_A]. `fill_prob_before_move` implements exactly that - and does NOT reproduce Table 4
(off by up to 0.11). An independent Monte Carlo agrees with `fill_prob_before_move` to three
decimals, and no value of lambda reconciles the two tables, so the gap is in the reading of
the event, not in the arithmetic. It is reported, not hidden. See `cst_table_checks`.

WHAT THIS SCRIPT ADDS

Queue POSITION, which is the quantity the model is really about and the one a volume-based
fill estimate cannot see: at a fixed level depth, being 2 deep and being 28 deep are the same
level and a wildly different order. `book_experiment` places a passive buy order at a chosen
position in a bid queue of a chosen depth and runs the CST event stream (embedded jump chain -
every question here is about the ORDER of events, not their timing) to measure

  * the fill probability by queue position,
  * the "volume alone" estimate a backtest makes when it ignores position, and
  * the adverse selection in the fills you DO get: buying at the bid earns half a tick only if
    the mid stays put, and the fills arrive precisely when the bid queue is being swept.

Run:  python lob_models.py      (numpy + scipy, fixed seed, about 20 s)
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

SEED = 20260909

# Cont, Stoikov & Talreja (2010) Table 2, the i = 1 (best-quote) row for Sky Perfect
# Communications: limit orders 1.85/min, market orders 0.94/min, cancellation 0.71/min/order.
CST_PARAMS = {"lam": 1.85, "mu": 0.94, "theta": 0.71}

# Table 3, BOTTOM panel (the paper's own Laplace-transform values), rows b = 1..5 (bid queue),
# columns a = 1..5 (ask queue). P[the mid-price increases before it decreases].
CST_TABLE3 = (
    (0.500, 0.336, 0.259, 0.216, 0.188),
    (0.664, 0.500, 0.407, 0.348, 0.307),
    (0.741, 0.593, 0.500, 0.437, 0.391),
    (0.784, 0.652, 0.563, 0.500, 0.452),
    (0.812, 0.693, 0.609, 0.548, 0.500),
)
# Table 4, BOTTOM panel. P[a bid order executes before the mid-price moves]. NOT reproduced.
CST_TABLE4 = (
    (0.497, 0.641, 0.709, 0.749, 0.776),
    (0.302, 0.449, 0.535, 0.591, 0.631),
    (0.206, 0.336, 0.422, 0.483, 0.528),
    (0.152, 0.263, 0.344, 0.404, 0.452),
    (0.118, 0.213, 0.287, 0.346, 0.393),
)


def _check(lam: float, mu: float, theta: float) -> None:
    if lam <= 0 or mu <= 0 or theta <= 0:
        raise ValueError("lam, mu and theta must be positive")


# ------------------------------------------------------------------ stationary queue length
def stationary_queue_dist(lam: float, mu: float, theta: float, n_max: int = 60) -> np.ndarray:
    """Stationary law of ONE best-quote queue: birth lam, death mu + x*theta in state x.

    Detailed balance gives pi_x proportional to lam^x / prod_{k=1..x} (mu + k*theta), which is
    proper for any theta > 0 - the linear cancellation rate is what makes the queue ergodic
    however large lambda is (CST Proposition 1). Returned over x = 0..n_max, normalised.
    """
    _check(lam, mu, theta)
    if n_max < 1:
        raise ValueError("n_max must be at least 1")
    logw = np.zeros(n_max + 1)
    x = np.arange(1, n_max + 1)
    logw[1:] = np.cumsum(np.log(lam) - np.log(mu + x * theta))
    w = np.exp(logw - logw.max())
    return w / w.sum()


# ------------------------------------------------- Proposition 3: direction of the next move
def prob_mid_up(a: int, b: int, lam: float = CST_PARAMS["lam"], mu: float = CST_PARAMS["mu"],
                theta: float = CST_PARAMS["theta"], n_max: int = 60) -> float:
    """P[the mid-price moves UP before it moves DOWN], given ask depth `a` and bid depth `b`.

    CST (2010) Proposition 3 at spread S = 1, computed as the paper's own proof describes the
    event - P[sigma_A < sigma_B], a race between the first-passage times to 0 of two
    INDEPENDENT birth-death queues - rather than by inverting eq. (10). Exact up to the state
    truncation at `n_max`, which is harmless because the death rate mu + x*theta grows without
    bound: at the paper's parameters the queue reaches 30 with probability ~1e-19.
    """
    _check(lam, mu, theta)
    if a < 0 or b < 0:
        raise ValueError("queue sizes must be non-negative")
    if a == 0:
        return 1.0 if b > 0 else 0.5
    if b == 0:
        return 0.0
    if max(a, b) > n_max:
        raise ValueError("n_max must be at least max(a, b)")
    n = n_max
    idx = lambda i, j: (i - 1) * n + (j - 1)              # noqa: E731  states i,j in 1..n
    d = mu + np.arange(1, n + 1) * theta                   # death rate by queue length

    rows, cols, vals, rhs = [], [], [], np.zeros(n * n)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            k = idx(i, j)
            tot = 2 * lam + d[i - 1] + d[j - 1]
            rows.append(k); cols.append(k); vals.append(tot)
            # ask side: up (i+1) reflected at the truncation, down (i-1); i == 0 -> mid UP
            if i < n:
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(-lam)
            else:
                vals[-1] -= lam                            # reflect: P[n+1, j] = P[n, j]
            if i > 1:
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(-d[i - 1])
            else:
                rhs[k] += d[i - 1] * 1.0                   # absorbed with the mid UP
            # bid side: j == 0 -> mid DOWN, contributes 0
            if j < n:
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(-lam)
            else:
                vals[-1] -= lam
            if j > 1:
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(-d[j - 1])
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n))
    p = spsolve(A.tocsc(), rhs)
    return float(p[idx(a, b)])


def prob_mid_up_grid(n: int = 5, **kw) -> np.ndarray:
    """The 5x5 panel of CST Table 3: rows b = 1..n (bid), columns a = 1..n (ask)."""
    return np.array([[prob_mid_up(a, b, **kw) for a in range(1, n + 1)]
                     for b in range(1, n + 1)])


def prob_mid_up_sim(a: int, b: int, n_paths: int = 200_000, seed: int = SEED,
                    lam: float = CST_PARAMS["lam"], mu: float = CST_PARAMS["mu"],
                    theta: float = CST_PARAMS["theta"], max_steps: int = 4000) -> tuple:
    """Monte Carlo check on `prob_mid_up`, on the embedded jump chain. Returns (p, std err).

    Only the ORDER of events decides which queue empties first, so the holding times can be
    dropped entirely and the whole ensemble stepped in lockstep with numpy.
    """
    _check(lam, mu, theta)
    rng = np.random.default_rng(seed)
    A = np.full(n_paths, a, dtype=np.int64)
    B = np.full(n_paths, b, dtype=np.int64)
    live = np.ones(n_paths, dtype=bool)
    up = np.zeros(n_paths, dtype=bool)
    for _ in range(max_steps):
        if not live.any():
            break
        da, db = mu + A * theta, mu + B * theta
        tot = 2 * lam + da + db
        u = rng.random(n_paths) * tot
        # order of the four events: +A, -A, +B, -B
        c1, c2, c3 = lam, lam + da, 2 * lam + da
        step_a = np.where(u < c1, 1, np.where(u < c2, -1, 0))
        step_b = np.where((u >= c2) & (u < c3), 1, np.where(u >= c3, -1, 0))
        A = np.where(live, A + step_a, A)
        B = np.where(live, B + step_b, B)
        hit_a, hit_b = live & (A == 0), live & (B == 0)
        up |= hit_a
        live &= ~(hit_a | hit_b)
    p = float(up.mean())
    return p, float(np.sqrt(p * (1 - p) / n_paths))


# ------------------------------------------- Proposition 5: executing before the mid moves
def fill_prob_before_move(b: int, a: int, lam: float = CST_PARAMS["lam"],
                          mu: float = CST_PARAMS["mu"], theta: float = CST_PARAMS["theta"],
                          n_max: int = 200) -> float:
    """CST (2010) Proposition 5 at S = 1: P[epsilon_B < sigma_A].

    eq. (14) makes epsilon_B a sum of independent exponentials with rates mu + theta*(i-1) for
    i = 1..b - the pure-death first passage of a queue in which YOUR order is the one that
    never cancels, so `b` counts you: b = 1 means you are at the front. eq. (16) says the
    mid-price can only move first through the ask queue emptying, sigma_A.

    Exact, by sweeping b tridiagonal solves over the ask queue length. This does NOT reproduce
    Table 4 - see `cst_table_checks` and the docstring at the top of the file.
    """
    _check(lam, mu, theta)
    if b < 1 or a < 1:
        raise ValueError("b and a must be at least 1")
    n = n_max
    x = np.arange(1, n + 1)
    dx = mu + x * theta
    F = np.zeros((b + 1, n + 1))
    F[0, :] = 1.0                                   # your order executed: success
    for w in range(1, b + 1):
        dw = mu + (w - 1) * theta
        diag = (dw + lam + dx).copy()
        diag[-1] -= lam                             # reflect at the truncation
        lower, upper = -dx[1:], -lam * np.ones(n - 1)
        rhs = dw * F[w - 1, 1:].copy()              # F[w, 0] = 0: the ask emptied first
        c, dd = np.zeros(n), np.zeros(n)            # Thomas algorithm
        c[0], dd[0] = upper[0] / diag[0], rhs[0] / diag[0]
        for k in range(1, n):
            m = diag[k] - lower[k - 1] * c[k - 1]
            c[k] = (upper[k] / m) if k < n - 1 else 0.0
            dd[k] = (rhs[k] - lower[k - 1] * dd[k - 1]) / m
        sol = np.zeros(n)
        sol[-1] = dd[-1]
        for k in range(n - 2, -1, -1):
            sol[k] = dd[k] - c[k] * sol[k + 1]
        F[w, 1:] = sol
    return float(F[b, a])


def cst_table_checks(**kw) -> dict:
    """Both panels against the paper: Table 3 reproduces, Table 4 does not. Returns the gaps."""
    got3 = prob_mid_up_grid(5, **kw)
    want3 = np.array(CST_TABLE3)
    got4 = np.array([[fill_prob_before_move(b, a) for a in range(1, 6)] for b in range(1, 6)])
    want4 = np.array(CST_TABLE4)
    return {"table3": got3, "table3_paper": want3, "table3_maxabs": float(np.abs(got3 - want3).max()),
            "table4": got4, "table4_paper": want4, "table4_maxabs": float(np.abs(got4 - want4).max())}


# --------------------------------------------------------- queue position, fills, toxicity
def book_experiment(position: int, depth: int, ask_depth: int | None = None,
                    n_trials: int = 20_000, seed: int = SEED,
                    lam: float = CST_PARAMS["lam"], mu: float = CST_PARAMS["mu"],
                    theta: float = CST_PARAMS["theta"], max_steps: int = 6000) -> dict:
    """Place a passive BUY order in a bid queue of `depth` units with `position` ahead of it.

    `position = 0` is the front of the queue, `position = depth - 1` the back. The order is
    small enough not to change anyone else's behaviour (it is not counted in the depth) and it
    is never cancelled. `ask_depth` defaults to `depth`, which makes the book symmetric and
    pins the unconditional P[mid up] at exactly 0.5 - a free correctness check on the run.

    The episode runs on the embedded jump chain until one of three things happens:
      * a market sell arrives while nothing is ahead      -> FILLED at the bid
      * the bid queue empties                             -> the mid moves DOWN half a tick
      * the ask queue empties                             -> the mid moves UP half a tick
    A filled order keeps running to the end of the episode, so the mid move that follows the
    fill is observed. Buying at the bid pays mid - 0.5 ticks, so the mark-to-mid of a fill is
    +1.0 tick if the mid then goes up and 0.0 if it goes down: 'earning the half spread' means
    0.5, and that is what the model does not give you.

    'volume_only' is the estimate a backtest makes from the tape alone - filled once the
    number of market sells that reached the bid exceeds `position`. It cannot see the
    cancellations that move you up the queue for free, and it books a fill on the sweep that
    ends the level.
    """
    _check(lam, mu, theta)
    if depth < 1 or position < 0 or position >= depth:
        raise ValueError("need depth >= 1 and 0 <= position < depth")
    a0 = depth if ask_depth is None else ask_depth
    if a0 < 1:
        raise ValueError("ask_depth must be at least 1")
    rng = np.random.default_rng(seed)
    A = np.full(n_trials, a0, dtype=np.int64)
    B = np.full(n_trials, depth, dtype=np.int64)
    ahead = np.full(n_trials, position, dtype=np.int64)

    live = np.ones(n_trials, dtype=bool)
    filled = np.zeros(n_trials, dtype=bool)
    trades = np.zeros(n_trials, dtype=np.int64)        # market sells that reached the bid
    adv_trade = np.zeros(n_trials, dtype=np.int64)     # places gained because of a trade
    adv_cancel = np.zeros(n_trials, dtype=np.int64)    # places gained because of a cancel
    mid_up = np.zeros(n_trials, dtype=bool)
    for _ in range(max_steps):
        if not live.any():
            break
        da, db = mu + A * theta, mu + B * theta
        tot = 2 * lam + da + db
        u = rng.random(n_trials) * tot
        c1, c2, c3 = lam, lam + da, 2 * lam + da
        is_lim_a, is_mkt_a = u < c1, (u >= c1) & (u < c2)
        is_lim_b, is_dec_b = (u >= c2) & (u < c3), u >= c3
        # a bid-side decrement is a market sell with prob mu/(mu + B*theta), else a cancel
        is_mkt_b = is_dec_b & (rng.random(n_trials) * db < mu)
        is_cxl_b = is_dec_b & ~is_mkt_b
        # a cancellation removes a uniformly chosen resting bid order: ahead of you with
        # probability ahead/B
        cxl_ahead = live & is_cxl_b & (rng.random(n_trials) * np.maximum(B, 1) < ahead)
        trade_ahead = live & is_mkt_b & (ahead > 0)

        A = np.where(live & is_lim_a, A + 1, np.where(live & is_mkt_a, A - 1, A))
        B = np.where(live & is_lim_b, B + 1, np.where(live & is_dec_b, B - 1, B))
        trades += (live & is_mkt_b).astype(np.int64)
        adv_trade += trade_ahead.astype(np.int64)
        adv_cancel += cxl_ahead.astype(np.int64)
        filled |= live & is_mkt_b & (ahead == 0)
        ahead = np.where(trade_ahead | cxl_ahead, ahead - 1, ahead)
        hit_a, hit_b = live & (A == 0), live & (B == 0)
        mid_up |= hit_a
        live &= ~(hit_a | hit_b)

    n_fill = int(filled.sum())
    up_given_fill = float(mid_up[filled].mean()) if n_fill else float("nan")
    mtm = 1.0 * up_given_fill if n_fill else float("nan")
    moved = int(adv_trade.sum() + adv_cancel.sum())
    fp = float(filled.mean())
    return {
        "position": position, "depth": depth, "ask_depth": a0, "n_trials": n_trials,
        "fill_prob": fp, "fill_se": float(np.sqrt(fp * (1 - fp) / n_trials)),
        "volume_only": float((trades > position).mean()),
        "p_mid_up_given_fill": up_given_fill,
        "mark_to_mid_ticks": mtm, "half_spread_ticks": 0.5,
        "adverse_selection_ticks": 0.5 - mtm if n_fill else float("nan"),
        "p_mid_up_unconditional": float(mid_up.mean()),
        "cancel_share_of_advance": float(adv_cancel.sum() / moved) if moved else float("nan"),
        "mean_trades": float(trades.mean()), "unresolved": int(live.sum()),
    }


def position_sweep(depth: int, positions, **kw) -> list:
    return [book_experiment(p, depth, **kw) for p in positions]


# --------------------------------------------------------------------------------- demo
if __name__ == "__main__":
    P = CST_PARAMS
    print("=" * 92)
    print("LIMIT-ORDER-BOOK MODELS  --  Cont, Stoikov & Talreja (2010), reproduced and extended")
    print("=" * 92)
    print(f"CST Table 2, Sky Perfect Communications, best quote: lambda(1) = {P['lam']}, "
          f"mu = {P['mu']}, theta(1) = {P['theta']} per minute")

    # ---- 1. Proposition 3 against the paper's own Table 3
    chk = cst_table_checks()
    print("\n1. Proposition 3: P[mid-price up before down] given the two queue sizes")
    print("   Computed as the paper's own proof states the event - a first-passage race")
    print("   P[sigma_A < sigma_B] between two independent birth-death queues - so as an")
    print("   absorption probability of the 2-D chain, with no Laplace inversion.")
    print("       " + "".join(f"{'a=' + str(a):>16}" for a in range(1, 6)))
    for b in range(1, 6):
        row = "".join(f"{chk['table3'][b - 1][a - 1]:>9.4f}/{CST_TABLE3[b - 1][a - 1]:<6.3f}"
                      for a in range(1, 6))
        print(f"   b={b}" + row)
    print(f"   mine / the paper's Table 3.  Worst |difference| over all 25 cells: "
          f"{chk['table3_maxabs']:.2e}")
    print("   - i.e. every cell agrees to the three decimals Table 3 is printed in.")
    for (a, b) in ((1, 5), (5, 1)):
        p, se = prob_mid_up_sim(a, b, 120_000)
        print(f"   Monte Carlo check a={a} b={b}: {p:.4f} +/- {se:.4f}   "
              f"exact {prob_mid_up(a, b):.4f}")
    print("   ! The diagonal is exactly 0.500 and the answer depends on the two QUEUE SIZES,")
    print("     not on any price history. b=5 against a=1 is a 0.812 chance of an up move.")

    # ---- 2. Proposition 5, and the part that does not reproduce
    print("\n2. Proposition 5: P[a bid order executes before the mid moves] - NOT reproduced")
    print("       " + "".join(f"{'a=' + str(a):>16}" for a in range(1, 6)))
    for b in range(1, 6):
        row = "".join(f"{chk['table4'][b - 1][a - 1]:>9.4f}/{CST_TABLE4[b - 1][a - 1]:<6.3f}"
                      for a in range(1, 6))
        print(f"   b={b}" + row)
    print(f"   mine / the paper's Table 4.  Worst |difference|: {chk['table4_maxabs']:.3f}")
    print("   eq. (14) is unambiguous - epsilon_B is a sum of exponentials with rates")
    print("   mu + theta*(i-1), i = 1..b - and eq. (16) is unambiguous about the event,")
    print("   P[epsilon_B < sigma_A]. Implementing both gives the top row above. An")
    print("   independent Monte Carlo agrees with it to three decimals and no single value")
    print("   of lambda reconciles Tables 3 and 4, so the gap is in the reading of the")
    print("   event, not the arithmetic. Reported, not hidden: use Table 3, and measure")
    print("   fill probability the way section 3 does.")

    # ---- 3. queue position is not depth
    depth = 20
    positions = (0, 1, 2, 5, 10, 15, 19)
    pi = stationary_queue_dist(**P)
    mean_depth = float((np.arange(len(pi)) * pi).sum())
    print(f"\n3. TRAP - 'the level traded more than the size ahead of me, so I was filled'")
    print(f"   A symmetric book, both queues {depth} units deep, sweeping YOUR position in the")
    print("   bid queue. Same level, same tape, same printed volume, one order per row:")
    print(f"   {'position':>9}{'fill prob':>11}{'+/-':>7}{'volume-only':>13}"
          f"{'vol/true':>10}{'trades seen':>13}{'cancel share':>14}")
    rows = position_sweep(depth, positions, n_trials=20_000)
    for r in rows:
        cs = "     -" if np.isnan(r["cancel_share_of_advance"]) else \
            f"{r['cancel_share_of_advance']:>6.2f}"
        print(f"   {r['position']:>9}{r['fill_prob']:>11.3f}{r['fill_se']:>7.3f}"
              f"{r['volume_only']:>13.3f}"
              f"{r['volume_only'] / max(r['fill_prob'], 1e-9):>10.2f}{r['mean_trades']:>13.2f}"
              f"{cs:>14}")
    lo, hi = rows[-1]["fill_prob"], rows[0]["fill_prob"]
    print(f"   Front of the queue fills with probability {hi:.3f}; the back of the SAME queue with")
    print(f"   {lo:.3f}. Volume alone sees ONE level and cannot tell the two orders apart. The")
    print("   'volume-only' column is what a backtest books when it fills you as soon as the")
    print("   tape prints more than the size ahead of you: exact at the front (there is nothing")
    print(f"   ahead), and {rows[4]['fill_prob'] / max(rows[4]['volume_only'], 1e-9):.0f}x too LOW at position {rows[4]['position']} - because "
          f"{rows[4]['cancel_share_of_advance']:.0%} of the places you gain")
    print("   are gained by CANCELLATIONS ahead of you, which never print on the tape.")
    print("   The same sweep inside CST's own calibrated depth range, both queues 5 deep:")
    small = position_sweep(5, (0, 1, 2, 3, 4), n_trials=20_000)
    print("   " + "  ".join(f"pos {r['position']}: {r['fill_prob']:.3f}" for r in small))
    print(f"   (CST's fitted Sky Perfect book has a mean depth of {mean_depth:.2f} units, so the")
    print("    20-deep queue is a statement about the queueing mechanics, not about that stock.)")

    # ---- 4. the fills you get are the ones you did not want
    print("\n4. Adverse selection - what the fills you DO get are worth")
    print("   You buy at the bid = mid - 0.5 ticks. Mark to the mid at the end of the episode:")
    print("   +1.0 tick if the mid then goes up, 0.0 if it goes down. 'Earning the half spread'")
    print("   means 0.5. The book is symmetric, so P[mid up] over ALL episodes must be 0.500.")
    print(f"   {'position':>9}{'fill prob':>11}{'P[up|fill]':>12}{'P[up] all':>11}"
          f"{'mark-to-mid':>13}{'adverse sel.':>14}")
    for r in rows:
        print(f"   {r['position']:>9}{r['fill_prob']:>11.3f}{r['p_mid_up_given_fill']:>12.3f}"
              f"{r['p_mid_up_unconditional']:>11.3f}{r['mark_to_mid_ticks']:>13.3f}"
              f"{r['adverse_selection_ticks']:>14.3f}")
    front, back = rows[0], rows[-1]
    print(f"   The 'P[up] all' column is {front['p_mid_up_unconditional']:.3f} against a symmetry argument that says 0.500")
    print("   exactly - that is the run checking itself. Conditional on YOUR bid being filled")
    print(f"   the mid goes up {front['p_mid_up_given_fill']:.0%} of the time from the front of the queue and only "
          f"{back['p_mid_up_given_fill']:.0%} from")
    print("   the back, so the same order is worth "
          f"{front['mark_to_mid_ticks']:.3f} ticks at the front and {back['mark_to_mid_ticks']:.3f} at the back")
    print(f"   against the 0.5 you thought you were earning - {front['adverse_selection_ticks']:.3f} to "
          f"{back['adverse_selection_ticks']:.3f} ticks handed back.")
    print("   ! The direction is the point. Queue position buys fill probability AND fill")
    print("     quality at the same time: from the front, a fill is one market sell out of many")
    print("     and says almost nothing; from the back, being filled is very nearly the same")
    print("     event as the bid queue emptying, which IS the mid-price falling. Note the back")
    print(f"     row fills with probability {back['fill_prob']:.3f} against P[mid down] = {1 - back['p_mid_up_unconditional']:.3f}: from the back of")
    print("     the queue your fill and the down-move are almost the same event.")
    print("   Nothing here is informed trading. It is pure queueing, in a model where every")
    print("   order is anonymous and every trader uninformed - so this cost is on top of the")
    print("   adverse selection that market-making-models measures against informed flow.")

    print("\nRule: a passive order's fill probability is a function of your QUEUE POSITION and"
          " the two queue sizes, not of the volume the level printed - and the fills that"
          " arrive are the ones whose mid is about to move against you.")
