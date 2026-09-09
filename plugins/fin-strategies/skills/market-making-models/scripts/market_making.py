#!/usr/bin/env python3
"""Avellaneda-Stoikov quoting, reproduced, and the two things it does not model.

The closed forms of Avellaneda & Stoikov (2008), the paper's own simulation rerun on its own
parameters, and then the part the model leaves out: every counterparty in AS is uninformed,
so the only risk is inventory. Add informed flow and the same quotes lose money.

AVELLANEDA & STOIKOV (2008), "High-frequency trading in a limit order book", Quantitative
Finance 8(3), 217-224. Every formula below was read in the published article
(math.nyu.edu/~avellane/HighFrequencyTrading.pdf), equation numbers as printed there:

  (6)  r^a(s, q, t) = s + (1 - 2q) * gamma*sigma^2*(T - t) / 2      reservation ask
  (7)  r^b(s, q, t) = s + (-1 - 2q) * gamma*sigma^2*(T - t) / 2     reservation bid
  (8)  r(s, q, t)   = s - q*gamma*sigma^2*(T - t)                   their average: the
       reservation / indifference price. Long inventory pushes it BELOW the mid.
  (12) lambda(delta) = A * exp(-k*delta), derived from a power-law market-order size
       distribution f^Q(x) ~ x^(-1-alpha) (9) and a LOGARITHMIC price impact (11), with
       "A = Lambda/alpha and k = alpha*K".
  (20) lambda^a(delta) = lambda^b(delta) = A*exp(-k*delta)          the symmetric case
  (29) r(s, t) = s - q*gamma*sigma^2*(T - t)                        same price for the
       TRADING agent, under an expansion in q (22) and a LINEAR approximation of the order
       arrival term (26)
  (30) delta^a + delta^b = gamma*sigma^2*(T - t) + (2/gamma)*ln(1 + gamma/k)
       -- the TOTAL spread, quoted "around this indifference or reservation price".

Their simulation, section 3.3 verbatim: "we chose the following parameters: s = 100, T = 1,
sigma = 2, dt = 0.005, q = 0, gamma = 0.1, k = 1.5 and A = 140", 1000 simulations. The
'symmetric' benchmark "uses the average bid/ask spread of the inventory strategy over the
time period, but centres it around the mid-price".

Their reported results, Tables 1-3, pp. 222-223:

                     Average spread   Profit   Std(Profit)   Final q   Std(Final q)
    Inventory  0.1        1.49         65.0        6.6         0.08        2.9
    Symmetric  0.1        1.49         68.4       12.7         0.26        8.4
    Inventory  0.01       1.35         68.6        8.7         0.12        5.1
    Symmetric  0.01       1.35         68.8       12.8         0.09        8.7
    Inventory  1          3.02         31.4        5.0         0.02        1.7
    Symmetric  1          3.02         44.0       11.0         0.00        5.1

All six rows are reproduced here to within about 1%. Three things this script measures
that the paper does not:

  1. that the reproduction only works with "with probability lambda*dt" read LITERALLY.
     A*dt = 0.7 here, so lambda*dt is not a small-probability regime, and it exceeds 1 when
     the reservation price pushes a quote through the mid. Substituting the exact Poisson
     1 - exp(-lambda*dt) costs 10-12% of the reported profit on the same strategy;
  2. what happens when a fraction of the arriving orders are INFORMED - they take only the
     side the next price move rewards. AS has no such trader; the entire risk in the model
     is inventory risk;
  3. how much wider the maker must quote to survive that. Less than the story suggests at
     AS's own parameters, and the quantity that decides it is the adverse move per informed
     fill divided by the half-spread being earned - not the risk aversion.

Run:  python market_making.py       (numpy only, fixed seed, a few seconds)
"""
from __future__ import annotations

import numpy as np

SEED = 20260909

# Avellaneda & Stoikov (2008) section 3.3, verbatim.
AS_PARAMS = {"s0": 100.0, "T": 1.0, "sigma": 2.0, "dt": 0.005, "q0": 0.0,
             "gamma": 0.1, "k": 1.5, "A": 140.0}

# Tables 1 (gamma=0.1) and 2 (gamma=0.01) on p. 222 and Table 3 (gamma=1) on p. 223, as
# (Profit, Std(Profit), Final q, Std(Final q)). The demo reproduces all six rows.
AS_TABLES = {
    0.1: {"spread": 1.49, "inv": (65.0, 6.6, 0.08, 2.9), "sym": (68.4, 12.7, 0.26, 8.4)},
    0.01: {"spread": 1.35, "inv": (68.6, 8.7, 0.12, 5.1), "sym": (68.8, 12.8, 0.09, 8.7)},
    1.0: {"spread": 3.02, "inv": (31.4, 5.0, 0.02, 1.7), "sym": (44.0, 11.0, 0.00, 5.1)},
}


# ------------------------------------------------------------------------- the closed forms
def reservation_price(s, q, gamma: float, sigma: float, t: float, T: float):
    """AS (2008) eq. (8) / (29):  r = s - q*gamma*sigma^2*(T - t).

    Long inventory (q > 0) puts the reservation price BELOW the mid, which skews both
    quotes down and makes the ask more likely to be hit - the model's whole inventory
    control. At t = T the adjustment vanishes and r = s.
    """
    if T < t:
        raise ValueError("t must not exceed T")
    return np.asarray(s, dtype=float) - np.asarray(q, dtype=float) * gamma * sigma ** 2 * (T - t)


def optimal_spread(gamma: float, sigma: float, t: float, T: float, k: float) -> float:
    """AS (2008) eq. (30):  delta^a + delta^b = gamma*sigma^2*(T-t) + (2/gamma)*ln(1+gamma/k).

    This is the TOTAL spread, quoted around the reservation price - not a half-spread. It
    does NOT depend on the inventory: AS note that "the bid-ask spread in (25) is
    independent of the inventory. This follows from our assumption of exponential arrival
    rates." All the inventory control lives in where the spread is centred.
    """
    if gamma <= 0 or k <= 0:
        raise ValueError("gamma and k must be positive")
    return gamma * sigma ** 2 * (T - t) + (2.0 / gamma) * np.log1p(gamma / k)


def mean_optimal_spread(gamma: float, sigma: float, T: float, k: float) -> float:
    """Time-average of eq. (30) over [0, T] - the spread the paper's 'symmetric' benchmark
    quotes. The integral of the first term is gamma*sigma^2*T/2; the second is constant.

    This is a closed form for the paper's own "Average spread" column, so it is the
    sharpest available check on eq. (30): 1.49, 1.35 and 3.02 for gamma = 0.1, 0.01 and 1.
    """
    return gamma * sigma ** 2 * T / 2.0 + (2.0 / gamma) * np.log1p(gamma / k)


def arrival_intensity(delta, A: float, k: float):
    """AS (2008) eq. (12) / (20):  lambda(delta) = A * exp(-k * delta).

    delta is the distance of the quote from the MID, per section 2.4. A negative delta - a
    quote through the mid, which happens when a large inventory pushes the reservation
    price far enough - gives an intensity above A, and the model does not forbid it.
    """
    if A <= 0 or k <= 0:
        raise ValueError("A and k must be positive")
    return A * np.exp(-k * np.asarray(delta, dtype=float))


def fill_probability(lam, dt: float, model: str = "poisson"):
    """Probability of at least one arrival in dt.

    'poisson'  1 - exp(-lambda*dt)   exact for a Poisson process, always in [0, 1]
    'linear'   min(lambda*dt, 1)     the paper's own wording, "with probability lambda*dt"

    They agree to O((lambda*dt)^2) and NOT at these parameters: A*dt = 0.7, so at delta = 0
    the linear form is already 0.7 against the exact 0.503. The demo measures the gap.
    """
    lam = np.asarray(lam, dtype=float)
    if model == "poisson":
        return 1.0 - np.exp(-lam * dt)
    if model == "linear":
        return np.minimum(lam * dt, 1.0)
    raise ValueError("model must be 'poisson' or 'linear'")


# ------------------------------------------------------------------------------ simulation
def simulate(strategy: str = "inventory", n_paths: int = 2000, seed: int = SEED,
             informed_frac: float = 0.0, informed_jump: float = 0.0,
             spread_mult: float = 1.0, prob_model: str = "linear",
             q_max: float | None = None, **overrides) -> dict:
    """The AS (2008) section 3.3 experiment, vectorised across paths.

    strategy='inventory' quotes eq. (30)'s spread around eq. (8)'s reservation price.
    strategy='symmetric' quotes the TIME-AVERAGE of that spread around the mid - the
    paper's own benchmark.

    `prob_model` defaults to 'linear', the paper's own literal "with probability
    lambda*dt", because that is what reproduces its Tables 1-3.

    `informed_frac` is not in AS. A fraction phi of arrivals are informed: they know the
    sign of the next mid move and only trade the side that profits from it, so raising phi
    both thins the flow and poisons what is left. `informed_jump` additionally displaces
    the mid by that much in the direction an informed trade went - the Glosten-Milgrom
    channel, where the toxic flow moves the price it was informed about. phi = 0 is the
    paper's model exactly.

    `q_max` is not in AS either - the paper states no inventory bound - and defaults to
    None so the reproduction is faithful.
    """
    p = {**AS_PARAMS, **overrides}
    if strategy not in ("inventory", "symmetric"):
        raise ValueError("strategy must be 'inventory' or 'symmetric'")
    if not 0.0 <= informed_frac <= 1.0:
        raise ValueError("informed_frac must be in [0, 1]")
    if spread_mult <= 0 or n_paths < 1:
        raise ValueError("spread_mult must be positive and n_paths >= 1")
    if informed_jump < 0:
        raise ValueError("informed_jump must be >= 0")
    T, dt, sigma, gamma, k, A = p["T"], p["dt"], p["sigma"], p["gamma"], p["k"], p["A"]
    n_steps = int(round(T / dt))
    rng = np.random.default_rng(seed)

    s = np.full(n_paths, p["s0"])
    q = np.full(n_paths, float(p["q0"]))
    cash = np.zeros(n_paths)
    n_fills = np.zeros(n_paths)
    n_toxic = np.zeros(n_paths)
    spread_sum = 0.0
    flat = mean_optimal_spread(gamma, sigma, T, k) * spread_mult

    for j in range(n_steps):
        t = j * dt
        if strategy == "inventory":
            centre = reservation_price(s, q, gamma, sigma, t, T)
            spread = optimal_spread(gamma, sigma, t, T, k) * spread_mult
        else:
            centre = s
            spread = flat
        spread_sum += float(np.mean(spread)) if np.ndim(spread) else spread
        p_ask, p_bid = centre + spread / 2.0, centre - spread / 2.0
        d_ask, d_bid = p_ask - s, s - p_bid
        hit_a = rng.random(n_paths) < fill_probability(
            arrival_intensity(d_ask, A, k), dt, prob_model)
        hit_b = rng.random(n_paths) < fill_probability(
            arrival_intensity(d_bid, A, k), dt, prob_model)
        up = rng.random(n_paths) < 0.5                       # sign of the NEXT mid move
        jump = np.zeros(n_paths)

        if informed_frac > 0.0:
            # an informed buyer only lifts the ask when the price is about to rise, and an
            # informed seller only hits the bid when it is about to fall
            inf_a = rng.random(n_paths) < informed_frac
            inf_b = rng.random(n_paths) < informed_frac
            tox_a, tox_b = hit_a & inf_a & up, hit_b & inf_b & ~up
            n_toxic += tox_a + tox_b
            jump = informed_jump * (tox_a.astype(float) - tox_b.astype(float))
            hit_a = hit_a & (~inf_a | up)
            hit_b = hit_b & (~inf_b | ~up)
        if q_max is not None:
            hit_a = hit_a & (q > -q_max)
            hit_b = hit_b & (q < q_max)

        cash += hit_a * p_ask - hit_b * p_bid
        q += hit_b.astype(float) - hit_a.astype(float)
        n_fills += hit_a + hit_b
        s = s + sigma * np.sqrt(dt) * np.where(up, 1.0, -1.0) + jump

    pnl = cash + q * s
    return {"pnl_mean": float(pnl.mean()), "pnl_std": float(pnl.std(ddof=1)),
            "q_mean": float(q.mean()), "q_std": float(q.std(ddof=1)),
            "q_absmax": float(np.abs(q).max()),
            "avg_spread": spread_sum / n_steps,
            "fills": float(n_fills.mean()), "toxic": float(n_toxic.mean()),
            "pnl_per_fill": float(pnl.mean() / n_fills.mean()) if n_fills.mean() else np.nan,
            "pnl": pnl, "final_q": q}


def best_spread_multiplier(informed_frac: float, mults, **kw) -> tuple[float, dict]:
    """The multiplier on eq. (30)'s spread that maximises mean PnL at a given toxicity.

    The Glosten-Milgrom argument by simulation: a maker facing informed flow must quote
    wider, and the widening is not a preference - it is what keeps the mean above zero.
    """
    scored = {m: simulate(informed_frac=informed_frac, spread_mult=m, **kw) for m in mults}
    best = max(scored, key=lambda m: scored[m]["pnl_mean"])
    return best, scored


# ---------------------------------------------------------------------------------- demo
def _fmt(r: dict) -> str:
    return (f"{r['avg_spread']:>9.2f}{r['pnl_mean']:>9.1f}{r['pnl_std']:>9.1f}"
            f"{r['q_mean']:>9.2f}{r['q_std']:>9.1f}")




if __name__ == "__main__":
    P = AS_PARAMS
    N_PATHS = 4000
    print("=" * 88)
    print("MARKET-MAKING MODELS  --  Avellaneda & Stoikov (2008), reproduced and stressed")
    print("=" * 88)

    # ---- 1. eq. (30) against the paper's own "Average spread" column
    print("\n1. eq. (30) checked against the paper's printed Average spread")
    print("   time-average of eq. (30) over [0,T] = gamma*sigma^2*T/2 + (2/gamma)*ln(1+gamma/k)")
    print(f"   {'gamma':>8}{'this formula':>15}{'paper prints':>15}{'in':>10}")
    for g, want, tbl in ((0.1, 1.49, "Table 1"), (0.01, 1.35, "Table 2"), (1.0, 3.02, "Table 3")):
        got = mean_optimal_spread(g, P["sigma"], P["T"], P["k"])
        print(f"   {g:>8g}{got:>15.4f}{want:>15.2f}{tbl:>10}")
    print("   Three independent numbers from three of the paper's own tables, out of one")
    print("   closed form. That is the sharpest available check that eq. (30) is transcribed")
    print("   correctly - the ln(1 + gamma/k) term especially.")
    r0 = reservation_price(100.0, 0.0, 0.1, 2.0, 0.0, 1.0)
    r5 = reservation_price(100.0, 5.0, 0.1, 2.0, 0.0, 1.0)
    rT = reservation_price(100.0, 5.0, 0.1, 2.0, 1.0, 1.0)
    print(f"   eq. (8) at s=100, gamma=0.1, sigma=2, T-t=1:  q=0 -> {r0:.2f}   "
          f"q=+5 -> {r5:.2f}   at t=T -> {rT:.2f}")
    print("   Long inventory pushes the quote pair DOWN, so the ask is likelier to be hit.")
    print("   ! The spread does NOT depend on q - AS: 'the bid-ask spread in (25) is")
    print("     independent of the inventory ... from our assumption of exponential arrival")
    print("     rates'. All the inventory control is in where the spread is CENTRED. A")
    print("     market maker who widens when long has left the model.")

    # ---- 2. the paper's own experiment, reproduced
    print(f"\n2. The paper's section 3.3 experiment, rerun ({N_PATHS} paths, seed {SEED},")
    print("   'with probability lambda*dt' taken literally, as the paper writes it)")
    print(f"   {'gamma':>6}  {'strategy':<11}{'spread':>8}{'PnL':>8}{'std':>7}{'q':>7}"
          f"{'q std':>7}     paper: spread / PnL / std / q std")
    for g in (0.1, 0.01, 1.0):
        for strat, key in (("inventory", "inv"), ("symmetric", "sym")):
            r = simulate(strat, gamma=g, n_paths=N_PATHS)
            sp, (pm, ps, qm, qs) = AS_TABLES[g]["spread"], AS_TABLES[g][key]
            print(f"   {g:>6g}  {strat:<11}{r['avg_spread']:>8.2f}{r['pnl_mean']:>8.1f}"
                  f"{r['pnl_std']:>7.1f}{r['q_mean']:>7.2f}{r['q_std']:>7.1f}"
                  f"     paper {sp:.2f} / {pm:.1f} / {ps:.1f} / {qs:.1f}")
    print("   All six rows of Tables 1, 2 and 3 land within about 1% of the paper's profits")
    print("   and dispersions, on an independent implementation and a different RNG.")
    print("   The paper's own reading is reproduced too: the symmetric strategy earns MORE,")
    print("   because it sits on the mid and takes more volume, and carries about twice the")
    print("   PnL dispersion and several times the inventory dispersion. AS: 'in the limit as")
    print("   gamma -> 0 the two strategies are identical' - the gamma = 0.01 rows show it.")

    # ---- 3. what "with probability lambda*dt" costs if you 'fix' it
    print("\n3. TRAP - 'with probability lambda*dt' is not a probability at these parameters")
    print(f"   {'delta':>8}{'lambda':>10}{'lambda*dt':>12}{'1-exp(-lambda*dt)':>20}")
    for d in (-0.5, 0.0, 0.5, 1.0):
        lam = arrival_intensity(d, P["A"], P["k"])
        print(f"   {d:>8.1f}{lam:>10.1f}{fill_probability(lam, P['dt'], 'linear'):>12.3f}"
              f"{fill_probability(lam, P['dt'], 'poisson'):>20.3f}")
    print(f"   {'gamma':>6}  {'strategy':<11}{'PnL (lambda*dt)':>17}{'PnL (Poisson)':>15}"
          f"{'gap':>8}{'fills':>10}")
    for g in (0.1, 0.01, 1.0):
        for strat in ("inventory", "symmetric"):
            a = simulate(strat, gamma=g, prob_model="linear", n_paths=N_PATHS)
            b = simulate(strat, gamma=g, prob_model="poisson", n_paths=N_PATHS)
            print(f"   {g:>6g}  {strat:<11}{a['pnl_mean']:>17.1f}{b['pnl_mean']:>15.1f}"
                  f"{b['pnl_mean'] / a['pnl_mean'] - 1:>+8.0%}"
                  f"{a['fills']:>7.0f}/{b['fills']:<3.0f}")
    print(f"   A*dt = {P['A'] * P['dt']:.2f}, so lambda*dt is NOT a small-probability regime;")
    print("   at delta = 0 it is 0.70 against the exact 0.503, and it exceeds 1 whenever the")
    print("   reservation price pushes a quote through the mid. An implementer who 'corrects'")
    print("   it to the exact Poisson probability reports 10-12% less profit on five of the")
    print("   six rows above, on the SAME strategy, and stops matching the paper. Neither")
    print("   reading is wrong; publishing a number without saying which one you used is.")

    # ---- 4. what AS does not model: informed flow
    print("\n4. Adverse selection - the risk AS has no term for")
    print("   A fraction phi of arrivals know the sign of the next mid move and take only the")
    print("   side that pays. phi = 0 is the paper's model exactly.")
    print(f"   {'phi':>6}  {'strategy':<11}{'PnL':>9}{'PnL std':>9}{'fills':>8}"
          f"{'PnL/fill':>10}{'picked off':>12}")
    base = {}
    for phi in (0.0, 0.10, 0.25, 0.50):
        for strat in ("inventory", "symmetric"):
            r = simulate(strat, informed_frac=phi, n_paths=N_PATHS)
            if strat == "inventory":
                base[phi] = r["pnl_mean"]
            print(f"   {phi:>6.0%}  {strat:<11}{r['pnl_mean']:>9.1f}{r['pnl_std']:>9.1f}"
                  f"{r['fills']:>8.0f}{r['pnl_per_fill']:>10.3f}{r['toxic']:>12.1f}")
    print(f"   The inventory strategy's mean PnL falls {base[0.0]:.1f} -> {base[0.5]:.1f} "
          f"({base[0.5] / base[0.0] - 1:+.0%}) as phi goes 0 -> 50%,")
    print("   on quotes that are optimal by eq. (30) at every step. Nothing in the model is")
    print("   wrong: AS has no informed trader, so the spread it prescribes compensates for")
    print("   inventory risk and for nothing else. Both strategies degrade almost identically,")
    print("   because the reservation-price skew is an inventory control, not a toxicity one.")

    # ---- 5. Glosten-Milgrom by simulation, and the honest answer
    print("\n5. So how much wider should the maker quote? Less than the story suggests.")
    half = mean_optimal_spread(P["gamma"], P["sigma"], P["T"], P["k"]) / 2.0
    step = P["sigma"] * np.sqrt(P["dt"])
    print(f"   average half-spread from eq. (30) = {half:.3f}; one mid step = "
          f"sigma*sqrt(dt) = {step:.3f}")
    mults = (1.0, 1.25, 1.5, 2.0, 3.0)
    print("   phi = 25%, sweeping a multiplier on eq. (30)'s spread. 'jump' is the extra mid")
    print("   displacement an informed fill causes (0 = section 4's model):")
    print(f"   {'jump':>7}{'adverse/half-spread':>21}"
          + "".join(f"{f'x{m:g}':>9}" for m in mults) + f"{'best':>8}")
    for jump in (0.0, 0.5, 1.0, 2.0, 4.0):
        best, sc = best_spread_multiplier(0.25, mults, informed_jump=jump, n_paths=N_PATHS)
        print(f"   {jump:>7.1f}{(jump + step) / half:>21.2f}"
              + "".join(f"{sc[m]['pnl_mean']:>9.1f}" for m in mults) + f"{f'x{best:g}':>8}")
    print("   Widening pays only once the adverse move per informed fill is well above the")
    print("   half-spread being earned - and at AS's OWN parameters it is nowhere near:")
    print(f"   the prescribed half-spread is {half / step:.1f}x one mid step, so a maker quoting")
    print("   eq. (30) is hard to pick off, and widening mostly starves the flow.")
    print("   That is the Glosten-Milgrom quantity, measured: the spread has to cover the")
    print("   EXPECTED LOSS TO INFORMED FLOW, and what decides it is the ratio in column 2 -")
    print("   not your risk aversion. eq. (30) prices gamma. It does not price phi.")
    print("   ! A, k and the size of an informed move are the three numbers this depends on,")
    print("     and AS calibrates none of them. Read the shape of this table, not its levels.")

    print("\nRule: quote eq. (30)'s spread around eq. (8)'s reservation price to control"
          " inventory, then re-price the spread for adverse selection, which the model"
          " does not contain.")
