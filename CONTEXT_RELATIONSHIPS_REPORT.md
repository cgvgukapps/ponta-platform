# Context Relationships, Movement and Space — Report (Stage C)

Stage C of `ponta-blueprint/context-packages.md`, built to `ponta-blueprint/context-relationships-task.md`. Written deliverable by deliverable; each section records what was built, how it was measured and what the founder decides.

## 1. In contest from logged events (P1 0.3.0, 20 November 2026)

**Built** (`plaiq_engine/context/possession.py`):
- **`p1_state`**, FIFA's possession states beside `p1_possession` (unchanged): `in_possession_home`, `in_possession_away`, `in_contest`, `out_of_play`, `unknown`, with `p1_state_source`.
- **In contest** where both teams are at the ball (P1's `contested`) and, with a logged feed, from a logged duel (`CHALLENGE`), clearance, block or deflection until a team has held control for `p1_min_switch_s` (0.4 s), for at most `p1_contest_max_s` (proposed 4 s). Interceptions are not contests: they win the ball.
- **IDSSE's defensive clearances** (`OtherBallAction` with `DefensiveClearance`) now reach P1 as `CLEARANCE` events (`tools/idsse.py`). That type is not one the event-anchored possession reads, so `p1_possession` is unchanged there too.
- **Chains** gain `interruptions` (the other team's logged contest touches inside the chain: FIFA's "interrupted", touched by the opponent with no turnover) and `out_of_play` (`retained` or `lost` by which team restarts, for a chain ending in a dead ball).
- **Data check:** a `possession_state` entry, Full with logged contest events, Degraded without (in contest only where both teams are at the ball).
- **Thresholds 0.13.0** (`p1_contest_max_s`, proposed). P1 0.3.0. Golden outputs recaptured: on the four regression matches only P1's new fields, its chain table and version stamps change; P2 to P4 are untouched.

**Measured** (`tools/context_contest.py`) against the founder's 50 blind possession calls on the German league's matches (18 November 2026: home, away or nobody yet):

| | `p1_possession` (unchanged) | `p1_state`, cap 2 s or 4 s | `p1_state`, cap 6 s |
|---|---|---|---|
| "Nobody" moments called in contest | 0 of 14 | 2 | 3 |
| Team moments kept | 11 of 26 | 10 | 10 |
| Controls kept | 10 of 10 | 9 | 9 |
| Right in total | 21 of 50 | 21 | 22 |

**In-play time in contest** (cap 4 s): 11% to 16% on the seven IDSSE matches (8.5% to 11.4% from logged events), 9% to 11% on Metrica Games 1 to 3 (5% to 8% from events), 2.7% on SkillCorner 1886347 (no logged feed: both teams at the ball only).

**Why it does not find the founder's "nobody" moments.** Of the 14:
- **2** follow a logged duel, and the state catches them;
- **about half** have the league logging the duel or the ball loss 0.3 to 0.8 s *after* the moment, when a player wins the ball; the founder saw it loose before;
- **the rest** are balls travelling after a misplaced pass, with no duel or clearance logged.

So FIFA's "in contest" (from the event log) and the founder's "nobody has it yet" (the ball in flight or loose) are different readings. The event log cannot place the second.

**Correction to the spec.** It said in contest would fix two phase review cases where a long ball was intercepted (Metrica 1 grid point 9197, Metrica 2 grid point 4206). The founder's notes accept `direct` at a moment just before the interception: that is a P4 phase question, not possession, and a pass in flight is in possession under FIFA's definitions too. Not fixed by this deliverable.

**For the founder:**
- **Keep `p1_state`** as FIFA's vocabulary, estimated: it is FIFA's own method (states from the event sequence), it changes no existing output, and it gives chains their "interrupted" and out-of-play outcomes.
- **Agree `p1_contest_max_s` at 4 s?** 2 s and 4 s give the same calls; 6 s catches one more loose ball.
- **Recommended next, separately:** an observed **ball-state** field (controlled by home, controlled by away, in flight, loose) from ball speed and the nearest player's distance, measured against the founder's calls with a larger judged sample. That is the reading behind "nobody has it yet", and the possession round already proposed it.

**Decided by the founder on 20 November 2026:** `p1_state` kept as FIFA's vocabulary (estimated), and `p1_contest_max_s` agreed at 4 s (thresholds 0.14.0, no value changed). The ball-state field is built next (section 2), before P5.
