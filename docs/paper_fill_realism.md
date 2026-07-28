# Paper maker fills are optimistic, and that is why nothing graduates

Audited 2026-07-28.

## What the accrual shows

| strategy | filled | countable | taker rate |
|---|---|---|---|
| llm | 14 | 2 | 14% |
| long_horizon | 23 | 0 | 0% |
| weather_temp | 23 | 0 | 0% |
| agent_trader_* (6 arms) | 45 | 0 | 0% |
| bias_harvest | — | 0 | 0% |

Every decision that captured a book tells the same story:

| strategy | price | bid | ask | | evidence |
|---|---|---|---|---|---|
| llm | 0.68 | 0.67 | **0.68** | at ask | `book_cross` |
| llm | 0.31 | 0.30 | **0.31** | at ask | `book_cross` |
| long_horizon | 0.82 | **0.82** | 0.83 | at bid | `synthetic` |
| weather_temp | 0.58 | **0.58** | 0.59 | at bid | `synthetic` |
| bias_harvest | 0.90 | **0.90** | 0.94 | at bid | `synthetic` |

`llm` crosses. Everything else rests at the bid, so `price >= best_ask` is
false and the fill is stamped `synthetic` — which is not in
`credible_fill_evidence`. **A resting paper strategy accrues zero countable
evidence, forever.**

## The mechanism, and the part that is already built

`PaperTrader` has two paths:

* `execute(order)` — fills immediately, at the order price, unconditionally.
* `submit_limit_order(order)` → `check_fills(prices)` — queues, and fills only
  on strict trade-through. Its docstring is explicit: *"Merely touching the
  limit is not evidence that our queue position executed, so no random fill
  credit is awarded."*

`check_fills` is already wired into the order monitor
(`bot_order_monitor.py:82`). But `client.py:368` routes **every** paper order
through `execute`, and nothing ever calls `submit_limit_order` outside the
market maker. So the honest path exists, is wired, and is never fed.

Two consequences:

1. **Paper P&L flatters maker strategies.** A resting bid fills instantly at
   the bid — a fill live would not have granted. Roughly 85-100% of paper fills
   are non-marketable, so this is not an edge case.
2. **The evidence system compensates by refusing to count them**, which is
   correct given the optimistic fill, but leaves maker strategies permanently
   ungraduatable from paper.

`trade_through` is already in `credible_fill_evidence` and **nothing emits it**
— the evidence type was designed for exactly this case and has no producer.

## The fix

Route non-marketable paper orders through `submit_limit_order` so they fill
only on trade-through, and stamp those fills `trade_through`. That
simultaneously makes paper P&L honest for maker strategies and unblocks their
graduation path with evidence that means something.

It is a real change to what "filled" means for most paper orders, and it will
reduce paper fill counts substantially — which is the point. It wants its own
pass rather than being appended to other work.

## Until then

Only strategies trading LIVE accrue countable evidence, because live fills are
stamped `venue_fill` unconditionally. That makes the ladder satisfiable only by
strategies an operator has already promoted — a chicken-and-egg the promotions
(`llm`, `llm_kalshi`, `agent_trader_opus`, `term_structure`) currently paper
over.
