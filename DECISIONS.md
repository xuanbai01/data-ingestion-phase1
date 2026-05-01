# Engineering Decisions Log

A phase-by-phase retrospective on what was tried, what was learned, and what changed direction. The README documents *what the system does today*; this document records *how it got there and why*. Useful for the next person picking up the codebase, and for me as a record of judgment calls I'd want to revisit.

The pipeline shipped in ten phases. The first three built the analytical machinery. Phase IV added the time dimension that turned a snapshot into a monitoring tool. Phases V and VI were polish on top of that. **Phases VII through X were a course-correction** — the diagnosis was that the system was producing data, not analysis, and the fix was structural editing rather than another feature.

---

## Phase I — Data ingestion

**What I tried first.** The original plan was Trustpilot — a flat HTML scrape per company, ~20 lines of `requests` + `beautifulsoup`. It got blocked by sign-in walls within the first 200 reviews on every brand I tried.

**What I changed.** Switched to Google Play Store via `google-play-scraper` (a maintained Python package wrapping Google's internal review endpoint). This came with a real schema — `review_id`, `app_version`, `thumbs_up`, `reviewer_name` — and pagination that survives 10,000-row pulls. The Trustpilot scraper survived in the repo as dead code through Phase IV before I deleted it.

**What I'd do differently.** Delete experimental code at the moment I stop using it, not three phases later.

---

## Phase II — Feature engineering

**The decision.** I needed per-review structured signals for downstream clustering. The choice was between rolling many features into one big model (an LLM with a structured-output prompt) versus assembling small, single-purpose tools that each do one thing. I picked the latter — TextBlob for polarity, spaCy for entities and noun-chunk extraction, sentence-transformers for embeddings, a heuristic for urgency. Each feature has a measurable failure mode and a separate cache slot.

**Why.** Cost predictability and debuggability. When the urgency score is wrong on a specific review, I can read the heuristic and find out why. When an LLM decides "urgency: 0.4" inside a single big prompt, I can't.

**Schema versioning.** Added `FEATURE_SCHEMA_VERSION` to the cache table early (a constant in `feature_engineering.py`). Bumping it on any semantic change means the next run recomputes; not bumping it would silently mix old and new feature semantics in the same report. The Phase III ABSA change forced a bump — that was the validation that the discipline was worth it.

---

## Phase III — ABSA

**The motivation.** The Phase II `aspects` field was a flat list of noun-chunk strings, with the *review-level* polarity attached implicitly. That was wrong on mixed reviews — "loved the price, hated the delivery" read as net-positive at the review level, so both `price` and `delivery` got tagged positive in aggregation.

**What changed.** Replaced the field shape from `list[str]` to `list[{aspect, polarity, confidence}]` and ran a DeBERTa ABSA model (`yangheng/deberta-v3-base-absa-v1.1`) per aspect. Bumped `FEATURE_SCHEMA_VERSION` from 2 to 3 to force recompute.

**Cost.** ABSA runs ~25 minutes on 10k reviews on CPU vs ~3 minutes on GPU. That's the dominant runtime cost in the pipeline. I added a `tqdm` progress bar specifically because CPU runs needed it to feel alive.

**Open question.** Whether to cache ABSA outputs separately at a finer grain (per-aspect rather than per-review) so adding new aspects to an existing review wouldn't re-run the model on the existing aspects. Decided no — the noun-chunk extraction step is stable enough that the set of aspects per review doesn't change between runs.

---

## Phase IV — Time dimension

**The blocker I almost missed.** The cleaner was silently dropping `app_version`, `review_id`, and `thumbs_up` from incoming records. The Google Play scraper was producing them, the database schema had columns for them, but the cleaning step's return dict didn't include them. I noticed this only when I started writing the cross-run matcher and discovered every review had `app_version=None`. Fixing the cleaner was a 5-line change; finding the bug took an hour.

**Cross-run matching design.** Two signals tried in order:

1. **Aspect-set Jaccard (≥ 0.4)** — `|A ∩ B| / |A ∪ B|` over each cluster's top distinctive aspects. Robust to wording shifts, naturally handles cluster growth/shrinkage.
2. **Centroid cosine (≥ 0.7)** — fallback when Jaccard is below threshold. Catches "same theme, different vocabulary" cases.

**What I tried first.** Pure centroid cosine. It worked but produced false-positive matches between unrelated negative-sentiment clusters whose centroids happened to land near each other in embedding space. Aspect Jaccard as the primary signal solved that — it's content-based, not just direction-in-space-based.

**The 20-review floor on Run Delta buckets.** Without it, KMeans label assignments shuffling between runs would produce dozens of phantom "new" / "resolved" entries on tiny clusters. The threshold is a heuristic; in practice it cleanly separates real movement from clustering noise.

---

## Phase V — LLM cluster labels & polish

**The seam decision.** `pipeline/llm.py` is intentionally a single function (`generate_cluster_label`) that talks to one provider. The motivation was to keep the LLM as a *swappable component*, not an architectural commitment. To switch from Claude Haiku to DeepSeek (which honors the Anthropic message format), only `_call_llm` needs to change.

**Caching by content, not by cluster ID.** The cache key is `md5(sorted(review_hashes) + sorted(aspect_set))` — a content hash. Same cluster contents → same key → no API call on a re-run. This means cache hits transfer across cluster-ID renumbering, across reviews.db migrations, and across multiple apps that happen to have a similar issue.

**Failure semantics first.** Decided very early that the pipeline must never crash on LLM failure. Missing API key, network errors, empty responses, and parsing failures all degrade silently to the keyword-style label. A WARNING goes to the run log; the report itself stays clean. Same pattern was reused for Phases VIII and IX.

---

## Phase VI — HTML report

**The architectural unlock.** Refactored the markdown renderer into two layers: `build_report_data()` returns a single nested dict (the "data model"), and `render_markdown(data)` / `render_html(data)` consume it. The original `_xxx_section()` markdown functions survived as thin wrappers for back-compat, but new code consumes the dict directly.

**Why this mattered.** Without the data-dict seam, adding the HTML report would have meant duplicating every section's logic. With it, the HTML renderer was almost entirely a Jinja template — the data was already shaped right.

**The self-containment constraint.** The HTML report had to work as a single file someone could drop into Slack or attach to an email. That meant inlining Pico CSS and Chart.js (~300 KB total). Decided that 300 KB was a reasonable price for "double-click and it works offline" — and, importantly, for not having to host any static assets anywhere.

---

## Phase VII — Editorial pass (the inflection point)

**The diagnosis.** First-pass user (a friend, then me re-reading after some distance) said the report was unreadable for non-engineers. They couldn't tell the difference between "this is the takeaway" and "this is the supporting evidence." The structure was data-driven (all sentiment metrics together, all cluster info together, all delta info together), not reader-driven.

**The reframe.** The report wasn't an analysis output — it was a data dump with a header. Real analyses lead with conclusions and demote evidence; this one was doing the opposite.

**What changed structurally**, in priority order:

1. **Elevator pitch** under the H1, written in plain English. No jargon. No mention of techniques. (`ELEVATOR_PITCH` constant — there's a comment in the source explicitly forbidding future me from adding "uses sentence embeddings" to the tagline.)
2. **Four-stat ribbon** answering "what happened this run?" in four numbers.
3. **Run-summary narrative** — one auto-generated sentence picking from four cases (escalations dominate / improvements dominate / mixed / quiet). Written templatically, not LLM-generated, so it always renders.
4. **Run Delta promoted** to the top, ahead of Priority Issues. The "what changed" answer is more time-sensitive than the "what is" answer.
5. **Top 3 issue cards expanded; the rest collapsed** behind a `<details>` element.
6. **"Detailed analysis" divider** signaling readers can stop here if they got their answer.

**The rule of thumb that came out of this.** *Reader effort, not data hierarchy, drives section order.* Every later phase used this as a tiebreaker.

---

## Phase VIII — Key Takeaways

**Why a new layer.** Even after the Phase VII restructure, the report still required the reader to *do the synthesis themselves* — read three cards, notice that two are escalating, infer the implication. The Key Takeaways section moves that synthesis up the stack: 3–5 LLM-generated bullets that cite specific issues and numbers, in priority order.

**What I tried first.** Generated takeaways from raw issue data (just the cluster labels and counts). Output was generic — "review the top issue first" type bullets that didn't reference the actual content.

**What changed.** Built a synthesis input that includes top issues *with their distinctive aspects*, the run-delta buckets, the loved/hated features, and the dominant entities. Output got concrete fast — "Investigate Payment Method Issues with Google Pay and OTP rejections" instead of "address the top issue."

**Failure mode to watch for.** The takeaways are only as honest as the synthesis input. If the input misrepresents the data (e.g. the wrong issue at the top), the takeaways will confidently misrepresent it too. Mitigation: every fact in the takeaways traces back to a number visible elsewhere in the report. A reader can verify in 30 seconds.

---

## Phase IX — Detail-section cull

**The rule I broke for too long.** Sections that were tables-without-insight had been accumulating since Phase II. Most Urgent Reviews, Emotion Distribution, Mentioned Entities, Aspect Index. Each one was easy to render because the data was already there; each one made the report longer and harder to skim.

**The cut.** Dropped all four. The signal each carried was already surfaced more usefully elsewhere:

- *Urgency* feeds the priority score, so urgent reviews surface inside the top issue cards.
- *Emotion* and *entities* are per-issue pills inside cards, where the framing is actionable rather than aggregate.
- *Aspects* are the cluster labels themselves.

**The principle.** *I'd been adding narrative on top of data sections instead of asking whether they earned their place in the first place.* Reframed the question on every surviving section: "If a reader skipped this section, what would they miss?" If the answer was "nothing the cards above didn't already tell them," the section came out.

**What replaced them.** One-sentence LLM narrative leads on the survivors (Top Positives and ABSA), with question-shaped headings ("What are users happy about?", "Which features are loved vs hated?") instead of category labels. The headings tell the reader why they'd read the section; the lead sentence delivers the headline finding.

---

## Phase X — Polish

A pass of small edits, none structural. None of these would matter on their own; together they remove the last ~10% of "this is a draft" feel.

- **Render-time entity noise filter.** spaCy's NER mistags corpus-domain words (`Rufus`, `Alexa`, `Newest`, `Tablet`) as ORG. Filtered at render time rather than re-running ABSA on a bumped schema version. Implementation: a small set in `summarizer.py`, comparison lowercases first.
- **Word-boundary review truncation.** Representative reviews used to slice mid-word at the character budget. `_truncate_at_word(text, max_len)` cuts at the last word boundary inside the budget, with the ellipsis counted against the budget so output never exceeds `max_len`.
- **Compact stable Run Delta state.** Empty buckets used to render four empty headers. Replaced with a single green-tinted line when nothing qualifies as movement.
- **Trend arrows on the leaderboard.** `_attach_trends_to_issues(data)` tags each top issue with its Run Delta classification (escalating / improving / new / stable). Renders as ▲ / ▼ / ✦ / — in the leaderboard column.
- **Theme-button visibility fix.** Pico CSS redefines `--pico-color` inside its own button scope to `--pico-primary-inverse` (white) — which made the theme toggle invisible on a white background in light mode. The fix uses `color: inherit` to escape the button-scope variable plus `border-color: currentColor` on hover. Cost me two iterations to find — the first attempt (`color: var(--pico-color)`) didn't work because Pico had already redefined that variable inside the button scope.

---

## Things I would do differently with another month

- **Snapshot-test the rendered HTML.** The Phase X theme-button bug shipped because the HTML renderer is only test-covered indirectly via `build_report_data`. A small Playwright or BeautifulSoup-based test that diffs the rendered HTML against a committed fixture would have caught it.
- **Bound the LLM caches.** `issue_labels`, `takeaways_cache`, and `section_narratives_cache` are content-addressed but not size-bounded. A long-running install will accumulate dead rows. Cheap to fix (TTL or LRU eviction on insert) but not pressing.
- **Multi-prior-run delta.** Run Delta currently compares N vs N-1. Tracking N vs N-2, N-3 would let the report show "this issue has been escalating for three runs in a row" — much higher signal than a single-run delta. The schema already supports it (`issue_snapshots` is keyed by `(app_slug, run_id)`); the matcher just needs to walk further back.
- **Earlier reader feedback.** The Phase VII–IX restructure was mostly correcting a structural mistake I'd made in Phase II by not asking "would a non-engineer read this?" Every phase past II would have been smaller if I'd gotten that feedback during Phase II rather than Phase VI.

---

## What I learned about pipeline design

Three things that I'd carry into the next project:

1. **Cache invalidation is a feature, not a chore.** Every cache layer in this pipeline (feature cache, issue labels, takeaways, narratives) has a deliberate cache-key design that survives the changes I expected. Picking the key shape early (content hash, not row id; schema version field; defensive load on missing tables) saved several reruns of expensive features when later phases added new fields.
2. **Make the data model the contract, not the renderer.** The `build_report_data` → `render_markdown` / `render_html` split was the single most useful refactor in the project. Every later phase composed cleanly on top of it; the pre-Phase-VI code had every section's markdown formatting tangled with its data extraction.
3. **Reader effort beats data hierarchy.** A report that's organized by what the data is (sentiment, then clusters, then deltas) is wrong by default. Organize by how much synthesis the reader needs to do — most-synthesized first, raw-evidence last — and a 100% factual report becomes a 10x more useful one.
