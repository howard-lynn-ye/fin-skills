# Regulatory sources — verbatim, from hashed copies

The rules an options backtest hard-codes, quoted from the primary documents rather than described.
Every document below was fetched on **2026-09-05** and hashed; the OCC site returns HTTP 403 to
Python's `urllib` and 200 to `curl`, so fetch with `curl` and read the file from Python.

| Document | URL | Size | Last-Modified | sha256 |
|---|---|---|---|---|
| OCC Rules | `theocc.com/getmedia/9d3854cd-b782-450f-bcf7-33169b0576ce/occ_rules.pdf` | 1,492,449 B | 2026-08-26 20:50:55 GMT | `facaed0569bc45f3a7c1872a0be33f6365a69daf094107c720a8b91f53043a9e` |
| OCC By-Laws | `theocc.com/getmedia/3309eceb-56cf-48fc-b3b3-498669a24572/occ_bylaws.pdf` | 1,658,160 B | 2026-04-24 15:41:11 GMT | `812f5e4c43de81ccda1c165b43115757e4c320f088da7720616dacdca0548c62` |
| Cboe RG08-073 | `cdn.cboe.com/resources/regulation/circulars/regulatory/RG08-073.pdf` | 24,066 B | 2019-04-23 (server) | `939825b597f0943ab814ba1cfa4e1e6bff38c5db95430241a165ca56c22f918a` |

🚨 **The OCC PDFs are live-updated at a stable URL.** A citation without the hash cannot be
reproduced; the Rules file was last modified ten days before this fetch. Cite the hash.

## 1. Exercise by exception — equity options

**OCC Rules, Rule 805(d)(2), p. 91.** A Clearing Member is deemed to have tendered an exercise notice
for:

> *"every option contract of each series listed in the Clearing Member's Expiration Exercise Report
> that has an exercise price below (in the case of a call) or above (in the case of a put) the closing
> price of the underlying security by $0.01 or more, unless the Clearing Member shall have duly
> instructed the Corporation, in accordance with subparagraph (b), to exercise none, or fewer than all,
> of the option contracts of such series carried in such account, provided that in the case of options
> with an exercise price expressed as a multiple of the per-unit price, in making the above
> calculations such multiple shall be applied to the closing price."*

The last clause is the `× multiplier` in the skill's §7b title, in the rule's own words.

**Interpretation .02, p. 92** — the caveat that makes broker thresholds legitimate:

> *"The exercise thresholds provided for in Rule 805(d) and elsewhere in the rules are part of the
> administrative procedures established by the Corporation to expedite its processing of exercises of
> expiring options by Clearing Members, and are not intended to dictate to Clearing Members which
> positions in customers' accounts …"*

**Interpretation .01, p. 93** — accelerated cash-deliverable contracts:

> *"When option contracts are adjusted to require delivery of a fixed amount of cash and the expiration
> date is accelerated, the "exercise by exception" threshold for such contracts for purposes of Rule
> 805(d)(2) shall be $.01 per share."*

Amendment trail on the same page: *Amended January 18, 2007 (SR-OCC-2006-20); July 18, 2012
(SR-OCC-2012-08); December 31, 2025 (SR-OCC-2025-017).* The rule moved **nine months** before this
fetch.

## 2. Exercise by exception — index and other cash-settled options

**OCC Rules, Rule 1804(c), pp. 171–172.** For expiring index option contracts:

> *"(1) for cash settled option contracts with a multiplier other than one, each option contract that
> has an exercise settlement value of $1.00 or more per contract, or such other amount as the
> Corporation may from time to time establish on not less than 30 days prior notice to all Index
> Clearing Members …"*

> *"(2) for cash settled option contracts with a multiplier of one, each option contract that has an
> exercise settlement amount of $0.01 or more per contract or such other amount as the Corporation may
> from time to time establish on not less than 30 days prior notice …"*

$1.00 per standard contract is $0.01 × 100. **The "$0.01 per contract" that circulates is the
multiplier-one figure quoted as if it were universal.** Both amounts are "such other amount as the
Corporation may … establish", so they are policy with 30 days' notice, not statute.

## 3. Where the $0.01 came from — Cboe RG08-073

**Cboe Regulatory Circular RG08-073, dated June 13, 2008.** The threshold moved

> *"… from $.05 to $.01 in a clearing member's customer, firm, and market maker account."*

> *"For example, if a clearing member has an equity option position in the customer account, which is
> in the money by $.01 or more, the position will be automatically exercised."*

> *"This change is effective for the June 2008 expiration, which is Saturday, June 21st."*

So: **$0.01 for all three account types since June 2008**; "$0.05" is the October-2006 value. Note the
Saturday — standard expirations were legally the Saturday after the third Friday until 2015-02-01,
and here is one in a primary document.

## 4. Which cash dividends adjust the contract

**OCC By-Laws, Article VI, Section 3A(a)(3), p. 129:**

> *"It shall be the general rule that there will be no adjustment to reflect (x) ordinary cash
> dividends or distributions or ordinary stock dividends or distributions (collectively, "ordinary
> distributions") by the issuer of the underlying equity security or (y) any cash dividend or
> distribution by the issuer of the underlying equity security if such dividend or distribution is
> less than $.125 per unit of trading."*

**Interpretation .01, p. 130:**

> *"Cash dividends or distributions (regardless of size) by the issuer of the underlying equity
> security which the Corporation believes to have been declared pursuant to a policy or practice of
> paying such dividends or distributions on a quarterly or other regular basis or which the
> Corporation believes represent an acceleration or deferral of such payments will, as a general rule,
> be deemed to be "ordinary cash dividends or distributions" within the meaning of Section 3A(a)(3)."*

**Interpretation .02, p. 131** (fund shares): *"… no adjustment shall be made for any such
distribution where the amount of the adjustment would be less than $.125 per fund share."*

**So the test is regularity first, then a $0.125-per-unit floor** — $12.50 on a standard 100-share
contract. The "10% of market value" rule in older material is not in the current by-laws.

⚠️ The research behind the skill cited Article VI **Section 11 / 11A** for adjustments. In this copy
Section 11 concerns the voting procedure of adjustment panels (pp. 133, 156); the substantive
dividend rule is **Section 3A**. Section 11 was not read in full.

## 5. Still secondhand

The OSI field layout, the adjusted-root suffix conventions (`MSFT1`, recycling), the mini-option `7`
suffix, and the 2015-02-01 Saturday→Friday expiration change were retrieved through search, not read
from OCC or Cboe documents. `_reverify.md` tracks them.
