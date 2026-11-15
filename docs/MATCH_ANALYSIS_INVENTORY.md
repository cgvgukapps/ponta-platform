# MatchAnalysis facade: inventory of what `app.py` imports

Step 2.4 of the extraction spec asks for an inventory before the `MatchAnalysis` facade is written. Source: the prototype's `app.py` at tag `prototype-validation-final`, import statements only (AST). It is **58 names, not the roughly 30 the spec assumed**. The grouping below is by name and purpose and is **provisional**: it has not yet been checked against how `app.py` calls each name. The verified call order for the full-tracking path is in `tools/golden/capture_prototype_golden.py`.

## Core analysis (facade candidates, full-tracking path)

| Module | Names |
|---|---|
| `analytics.hex_analytics` | `assign_all_players`, `compute_pressure`, `infer_possession`, `tag_possession` |
| `analytics.hex_value` | `xT_surface`, `compute_value`, `compute_movement_value`, `compute_movement_delta_v`, `tag_events_delta_v` |
| `analytics.pass_option_value` | `compute_event_pov`, `compute_frame_pov`, `_is_failed_pass_event` (private) |
| `analytics.pitch_control` | `build_pitch_hex_grid`, `compute_frame_control`, `compute_freeze_frame_control`, `DEFAULT_PROFILE` |
| `analytics.event_stitch` | `build_pass_receipt_lookup`, `patch_pass_receipt_receiver` |

Verified order (golden capture): `build_pitch_hex_grid` → `tag_events_delta_v` → `assign_all_players` → `compute_pressure` → `infer_possession` → `tag_possession` → `compute_movement_delta_v` → `compute_movement_value` → `compute_event_pov` (returns POV and alternatives frames); per-frame surfaces via `compute_frame_control` and `compute_value`.

## Event-only path (StatsBomb)

`ingestion.statsbomb_io`: `SB_PITCH`, `list_sb_competitions`, `list_sb_matches`, `load_sb_match`. `statsbomb.identity`: `identity_summary`, `resolve_identities`. `statsbomb.sb_dq`: `sb_carry_dq`, `sb_event_player_rows`, `sb_pass_dq`. `statsbomb.sb_pov_adapter`: `sb_all_teammates`, `_nearest_hex` (private). `statsbomb.stitch`: `build_stitch_graph`, `end_proxy_ff`.

## Rendering helpers (probably not facade; reference for the web pitch canvas)

`build_render_grid`, `control_to_hex_traces`, `value_to_hex_traces`, `player_zone_outlines`, `corridor_traces_for_frame`, `build_grid_to_hex_mapping`. These produce Plotly traces for the Streamlit view. Not verified; confirm before leaving them out.

## Unclassified (check usage in `app.py`)

`smooth_control`, `pressure_field`, `pitch_config.DEFAULT_PITCH`.

## Roster and player lookups (`roster.player_db`, `roster.build_roster`)

`get_player`, `get_player_by_id`, `get_profiles`, `list_players`, `get_team_name`, `get_sb_player`, `get_sb_player_by_name`, `get_sb_team_name`, `list_sb_players`, `load_sb_roster`, `get_sc_player`, `get_sc_player_by_id`, `get_sc_team_name`, `list_sc_players`, `load_sc_roster`, `format_spells_compact`, `sub_events`, `write_statsbomb_roster`. These return **names**. The semantic layer uses pseudonymous `player_ref`, so the facade must not expose name lookups to the AI path.

## Decisions needed before writing the facade

1. Whether the facade takes already-normalised Matrix output or the provider-specific frames the prototype used (affects the equivalence test, step 4).
2. Whether rendering helpers stay out of the engine package.
3. Private names (`_is_failed_pass_event`, `_nearest_hex`) get public wrappers, or the facade absorbs the logic.
