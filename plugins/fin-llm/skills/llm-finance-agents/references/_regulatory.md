# Regulatory constraints on an LLM trading agent

The US trading rules themselves — short-sale restrictions, margin, settlement, PDT, wash sales,
market-data licensing, and how backtested performance may be presented — now live in their own
skill, because they constrain every strategy and not only agent-shaped ones:

**`../../../fin-core/skills/us-market-rules/SKILL.md`**

They used to live here, which meant `broker-execution-apis`, `backtesting-engines` and
`research-integrity-guards` had no path to them. Read that skill; this file keeps only what is
specific to an LLM acting as the agent.

## 🚨 Personalized investment advice is regulated

Under the **Investment Advisers Act**, recommending specific securities or allocations as *suitable
for a particular person* is regulated activity.

An agent may:

- explain mechanics
- analyze data
- implement a strategy the user has stated

It must not present a recommendation as personal suitability advice. **Say plainly that you are not
a licensed adviser and stay on the mechanics.** This is a design constraint on the agent's output,
not a disclaimer to append after the fact — an agent whose whole output shape is "you should buy X"
does not become compliant by adding a footer.

## The reporting obligation applies to the agent's output too

An LLM agent emits backtest numbers constantly, often into a chat window the user will screenshot
and send to someone else. The Marketing Rule's disclosure requirements
(`../../../fin-core/skills/us-market-rules/SKILL.md` §6) attach at the moment the number leaves the
machine, which for an agent is every message.

Emit the result card from
`../../../fin-core/skills/research-integrity-guards/scripts/result_manifest.py` rather than a bare
Sharpe ratio.

## Autonomy raises the stakes on §2 of us-market-rules

A human running a mean-reversion short discovers Reg SHO Rule 201 the first time a fill is rejected.
An unattended agent discovers it as a growing divergence between expected and realized P&L, which is
the failure mode that runs longest before anyone notices. Model the SSR trigger explicitly, or gate
the short leg off.
