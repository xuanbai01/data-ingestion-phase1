# Review Analysis Report — Walmart
*Groups app reviews into recurring issues and tracks them over time — surfacing what's getting worse, improving, new, or resolved.*

10,000 reviews · generated 2026-05-01 · run `20260501T004318Z`

> Top issue: Payment and Account Verification Issues — 680 reviews, avg 2.1★, priority 0.63. No significant changes since prior run.
>
> **10,000** reviews · **0** escalating · **0** new · **0** resolved

## Key takeaways
- **Fix Payment and Account Verification Issues first**: This issue affects 680 reviews (6.8% of corpus) with a critically low average rating of 2.1, and spans versions 26.8, 26.11, and 26.6—users report sadness and anger primarily around PayPal, EBT, and Google integrations.
- **Address Intrusive Design and Navigation Friction urgently**: The second-ranked issue impacts 1,893 reviews (18.9% of corpus) with an average rating of 2.6, appearing consistently across versions 26.8, 26.11, and 26.10, particularly affecting EBT and Walmart experiences.
- **Preserve and expand price and shopping value proposition**: Positive sentiment around "price, shopping, store" generates 1,168 high-rated reviews (avg 4.8), and "price, job, thank" yields 1,010 reviews at 4.9 rating—this core strength should remain a product focus.
- **Investigate and mitigate scam and error perceptions**: Although lower in volume, "scam" (27 mentions, polarity -0.97) and "error" (29 mentions, polarity -0.92) represent the most negatively polarized aspects and risk reputational damage despite small counts.
- **Monitor address validation workflows**: Address-related issues show strong negative polarity (-0.77, 69 mentions) and appear in both top payment and navigation friction issues, suggesting a cross-functional UX/backend problem requiring investigation.

## Run Delta — stable
_No significant changes vs prior run `20260430T182313Z`._

## Priority Issues
Negative-leaning clusters ranked by composite priority — weighted by volume, severity (avg rating), urgency, and intense-emotion share (anger / disgust / fear).

| # | Issue | Reviews | Avg rating | Bug / Complaint | Priority |
|---:|---|---:|---:|---|---:|
| 1 | Payment and Account Verification Issues | 680 | 2.1 | 65% / 35% | 0.63 |
| 2 | Intrusive Design and Navigation Friction | 1893 | 2.6 | 51% / 49% | 0.62 |

### Issue 1 — Payment and Account Verification Issues  (priority 0.63)
*card, account, option, order*
**680 reviews** · avg rating 2.1 · avg urgency 0.50

| Side | Share | Top aspects |
|---|---:|---|
| Objective (bugs) | 65% | card, account, option, order |
| Subjective (complaints) | 35% | card, account, payment, option |

**Top emotions:** sadness 36%, neutral 34%, anger 13%
**Mentioned entities:** PayPal (16), EBT (11), Google (10), Wal-Mart (3)

**Trend (2026-02-16 → 2026-04-29):** `▆▄▇▇█▃▂▇█▄▃▁` (12 buckets, peak 71 mentions)

**By app version:** 26.8 (103), 26.11 (77), 26.6 (73), 26.5.1 (56), 26.10 (55)

Representative reviews:
- It has been a nightmare trying to order. used 3 valid cards , talked to a fake voice , talked to a real person who said it would be fixed in
- After not using this service in a while, my phone and the app I had a hard time reconnecting and verifying my account. I wasn't getting the 
- wouldn't allow a membership offer at the end of the checkout apparently the card I used which was still good enough to take the payment wasn

### Issue 2 — Intrusive Design and Navigation Friction  (priority 0.62)
*order, item, store, delivery*
**1,893 reviews** · avg rating 2.6 · avg urgency 0.43

| Side | Share | Top aspects |
|---|---:|---|
| Objective (bugs) | 51% | order, item, store, delivery |
| Subjective (complaints) | 49% | order, item, store, delivery |

**Top emotions:** neutral 33%, sadness 22%, joy 12%
**Mentioned entities:** EBT (15), Wal-Mart (12), Google (10), Instagram (5), ALWAYS (4)

**Trend (2026-02-16 → 2026-04-29):** `▄▄▅█▆▃▄▅▅▆▄▁` (12 buckets, peak 215 mentions)

**By app version:** 26.8 (280), 26.11 (194), 26.10 (189), 26.6 (183), 26.13.1 (160)

Representative reviews:
- My biggest gripe is the FULL-PAGE popup that tries to trick us into hitting the single-button acceptance for joining Walmart+. This intrusiv
- The only thing I had an issue with was trying to add an item during the option for replacement. I didn't want to go back and undo what I had
- it's really nice and convenient to use, especially for same-day pickups. it's it's nice that you can add your WIC card and shop straight fro

---

## Detailed analysis

## What are users happy about?
**Price and shopping feedback** dominates positive reviews with 1,168 mentions at a 4.82 rating, suggesting this is the strongest area to preserve and monitor for competitive advantage.

| Theme | Top aspects | Reviews | Avg rating |
|---|---|---:|---:|
| 5 | price, shopping, store | 1168 | 4.8 |
| 11 | delivery, store, service | 1163 | 4.5 |
| 6 | price, job, thank | 1010 | 4.9 |
| 9 | delivery, service, grocery | 772 | 4.5 |
| 7 | awesome | 526 | 4.9 |

## Which features are loved vs hated?
This section shows what aspects of the app customers love and hate most. **"Love" sentiment peaks with the "love" aspect at 0.94 polarity (97 mentions), while "scam" generates the strongest negative reaction at -0.97 polarity (27 mentions)—watch the scam complaints closely and investigate the account issues that appear most frequently in negative feedback (115 mentions).**

### Top Loved Features
| Aspect | Avg polarity | Mentions |
|---|---:|---:|
| love | +0.94 | 97 |
| deal | +0.89 | 110 |
| service | +0.60 | 739 |
| convenience | +0.86 | 87 |
| price | +0.61 | 501 |
| thank | +0.81 | 85 |
| ordering | +0.77 | 107 |
| ease | +0.94 | 38 |

### Top Hated Features
| Aspect | Avg polarity | Mentions |
|---|---:|---:|
| fee | -0.81 | 75 |
| address | -0.77 | 69 |
| scam | -0.97 | 27 |
| error | -0.92 | 29 |
| code | -0.78 | 53 |
| refund | -0.72 | 66 |
| account | -0.63 | 115 |
| screen | -0.89 | 27 |

## By the numbers
10,000 reviews · 3.97★ avg · +0.30 polarity · 10% negative (4% bug-shaped, 6% emotional) · 2,605 aspects · 12 topics · 2 priority issues
