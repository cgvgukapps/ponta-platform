# Review backlog B: before Phase 2

> **Status: in progress (started 27 November 2026).** Backlog B of the independent review (`ponta-blueprint/work-review.md`, section 5) is the work before the AI layer. This report records each step as it lands: what changed, what it fixes and how it was checked. Measured figures are not expected to move in steps 1 to 3; any that do are reported here.

## 1. A synthetic engine golden in CI (SE-01)

**The gap.** The only value-level check of Plaiq's metrics was the prototype equivalence test, which is opt-in and whose goldens live outside the repository. A change that shifted every decision score passed CI.

**What changed.** The licence-free synthetic match (fictional players, 10 s, no sample data) moved to `tests/support/synthetic_match.py`. `tests/golden/synthetic/engine.json` (38 kB, tracked) holds its headline values: the decision values and their alternatives, movement value, pass completion, Choice, Gain version 2, 1v1 pressure at five frames, and the control and value surfaces at those frames (summary statistics and every 97th cell). `tests/cross_engine/test_synthetic_golden.py` compares them on every push, and checks the scales: pressure 1 to 10, Gain 0 to 1.5, Choice 0 to 1, each band agreeing with its score. Recapture with `python tools/synthetic_golden.py --capture`, only for an intended change.

**Checked.** Moving one constant in `pass_option_value.py` (the carry pressure discount, 0.30 to 0.31) fails the test; restored, it passes, also with `PONTA_SAMPLE_DATA` pointing nowhere (CI mode).

## 2. Identity and pseudonyms (AI-04, DE-03, DE-15, SE-11, AI-11, GC-04, SE-15)

**The gaps.** Provider ids sat in every AI-facing row (`match_id` `skillcorner_1886347`, `competition` `skillcorner:61`), so a match id and a shirt number resolved to a named player by lookup. `event_id` was the row position in the loader's table, so any mapping change shifted later ids and could turn a stored citation into a different action. The team-key secret defaulted to a public constant, keys carried no version, a name in an event feed would have become a player ref, and STATSports ids used one public salt for every tenant.

**What changed (semantic schema 2.0, profiles 0.3.0).**

- `plaiq_engine/teams.py`: `TeamRegistry` needs its secret (`TeamRegistry.for_reference()` for tools and tests). The reference secret is refused when `PONTA_ENV` is `production`; a tenant secret needs 16 characters or more. Every key carries its key version (`t1_`, `m1_`, `c1_`, `s1_`). `identity(match)` returns the match's opaque keys.
- The semantic builder needs the registry. `match_id` is the match key on every event, episode and profile row; competition and season are opaque keys; `state_id` and `possession_chain_id` are built on the match key.
- `event_id` comes from the event's content: the match key, period, start frame, type, actor and its order among identical events. Adding an event elsewhere leaves every other id unchanged (tested).
- The provider's ids move to `provenance()` (`event_provenance.parquet`): event id, provider, provider match id and the provider's event id where the feed has one. It is not AI-facing; in the product it sits with Postgres.
- A player ref must be a shirt number; a name in the feed fails the build (`PseudonymError`). The StatsBomb event-only path still keys players by name inside its own identity code; it does not reach the builder, and StatsBomb is "Coming soon". Mapping StatsBomb's `player_id` to a shirt number is a condition of building that loader.
- `nested_record` gains a `team` block, and a test checks every flat column reaches the nested view (the loader's row position, `event_idx`, is the one named exclusion).
- STATSports player ids take a per-tenant `id_salt` (an HMAC under it); the public salt is refused in production.

**Checked.** A value-level test builds the synthetic match as SkillCorner match 1886347 with real-looking team, competition and season ids and finds none of them in any value of the semantic events or episodes; the provenance table holds them. Values are unchanged: the synthetic engine golden and the context goldens pass.

**Still to do in the product.** Keep tenant secrets in Supabase Vault; treat the registry's merges file as tenant-confidential; the API never returns `event_provenance` to a model.
