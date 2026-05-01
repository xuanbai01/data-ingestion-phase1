# Review Issue Tracker — Project Summary

A one-page overview for non-engineering readers. Code, design rationale, and run instructions are in [README.md](README.md).

---

## The problem

App reviews are the most direct channel users have for telling a product team what's broken. The volume makes them unreadable. Amazon's Google Play page accumulates ~10,000 new reviews a quarter; nobody on the product team reads more than a handful, and the ones they do read are not a representative sample. Recurring complaints get rediscovered every few weeks; emerging issues only get noticed once they're loud enough to spike the rating average.

## The approach

Group reviews into **issues**, score each issue's severity, and track how each issue moves between runs. Three layers, in order of how synthesized the output is:

1. **Per-review features** — sentiment, per-aspect polarity (DeBERTa ABSA), emotion (DistilRoBERTa), urgency, named entities, and a 768-dim embedding from `all-mpnet-base-v2`.
2. **Cluster discovery and matching** — KMeans over the embeddings with adaptive K via silhouette score; clusters from the current run are matched to the prior run's clusters via aspect-set Jaccard with a centroid-cosine fallback. This is what turns a one-shot snapshot into a monitoring tool.
3. **Synthesis** — a 4-stat ribbon, a one-sentence run narrative, and an LLM-synthesized 3–5 bullet "Key Takeaways" section sit on top of everything else. Anyone reading the report top-down gets the answer in the first 200 words.

## The output

Every run writes two reports next to each other in `reports/{app}_{run_id}.{md,html}`:

- A **markdown** report — greppable, diffable, paste-into-chat friendly.
- A **self-contained HTML** report — Pico CSS and Chart.js inlined (~300 KB), opens offline, works in any browser. Sortable tables, interactive sparklines, dark/light theme.

Sample reports for all three currently-supported apps live in [`reports/showcase/`](reports/showcase/) and are linked from the README's "See the output" section.

## What it found on Amazon (10,000 reviews, run `20260429T051411Z`)

The top three priority issues account for ~34% of all reviews and almost all of the negative volume:

| # | Issue | Reviews | Avg ★ |
|---|---|---:|---:|
| 1 | App Crashes and Freezing Issues | 1,350 | 1.6 |
| 2 | Unreliable Delivery and Poor Customer Service | 1,086 | 1.5 |
| 3 | Payment Method Issues and Rejections | 1,004 | 1.4 |

Two non-obvious findings the synthesis layer surfaced that wouldn't have come out of raw rating averages:

- **Tablet support was silently dropped on Samsung and Lenovo devices.** 503 reviews flag the issue with an avg rating of 1.3 — the lowest of any issue. The dominant emotion is *surprise*, not anger; users are discovering this on app open. Strong candidate for a roadmap announcement even if the engineering position is "tablets are deprecated."
- **Account / payment friction concentrates on Google Pay and OTP rejections**, not stored card failures. The mentioned-entities pill on the Payment issue card lists Google (31), Walmart (11), OTP (9), PayPal (3) — the comparative volume on Google suggests an integration regression rather than a generic auth bug.

The positive cluster (1,867 reviews, avg ★ 4.6) is dominated by *shopping*, *price*, and *service* — those are the differentiators worth preserving, and they're not where the engineering effort needs to go.

## Limitations (honest)

- The cross-run delta needs at least one prior snapshot to be meaningful. The first run for a new app shows a "no prior run on file" placeholder by design.
- LLM-generated titles and key-takeaway bullets degrade silently to keyword labels / no-section if no Anthropic key is configured. The pipeline never crashes on LLM failure — but the report is meaningfully less readable without a key.
- Cluster IDs are not stable across runs (KMeans labels are arbitrary). The cross-run matching layer compensates with aspect Jaccard + centroid cosine, which has worked well on Amazon but hasn't been stress-tested across many runs of eBay or Walmart yet.
- Per-aspect ABSA dominates pipeline runtime on CPU (~25 min on 10k reviews vs ~3 min on GPU). Production deployment would either pin GPU instances or accept the slower CPU path.

## What's not in scope

Multi-app comparison views, scheduled CI runs, multi-prior-run trend lines, and HTML snapshot tests were considered but deliberately deferred — they're net-new analytical surface, not finishing work, and the current ten-phase scope already delivered what the pipeline was meant to deliver.
