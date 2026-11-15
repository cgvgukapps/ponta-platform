"""
Shuffled player IDs are recovered from trajectories alone.

Named ``test_identity_matching`` rather than ``test_identity`` only because
``tests/statsbomb/test_identity.py`` already exists and pytest requires unique
test module basenames when test directories are not packages. Existing tests
were not to be touched, so this file yielded the name.

Accuracy is measured and asserted against a stated bar rather than assumed, and
the measured value is printed into the failure message when it falls short.
"""

import pytest
from matrix_engine.identity import (
    apply_mapping,
    match_identities,
    split_sources,
)
from matrix_engine.sync import synchronize
from matrix_engine.testing import DegradationParams, degrade

#: Accuracy the matcher is expected to clear under the standard degradation.
ACCURACY_BAR = 1.0


def _match(truth, synthetic, **kwargs):
    synced = synchronize(
        {"metrica": truth, synthetic.data["source"].iloc[0]: synthetic.data},
        reference="metrica",
    )
    ref_slice, other_slice = split_sources(
        synced.data, "metrica", synthetic.data["source"].iloc[0]
    )
    return match_identities(ref_slice, other_slice, **kwargs)


# ---------------------------------------------------------------------------
# The headline claim
# ---------------------------------------------------------------------------

def test_every_shuffled_id_is_recovered(drifted):
    result = drifted["result"].identity["synthetic_gps"]
    truth = drifted["synthetic"].identity_map
    accuracy = result.accuracy_against(truth)
    assert accuracy >= ACCURACY_BAR, (
        f"identity accuracy {accuracy:.1%} below the {ACCURACY_BAR:.0%} bar "
        f"({len(result.mapping)} matched, {len(result.unmatched)} unmatched)"
    )


def test_no_player_is_left_unmatched(drifted):
    assert drifted["result"].identity["synthetic_gps"].unmatched == []


def test_mapping_is_one_to_one(drifted):
    mapping = drifted["result"].identity["synthetic_gps"].mapping
    assert len(set(mapping.values())) == len(mapping)


def test_accuracy_counts_unmatched_players_as_wrong(drifted):
    """Declining to answer must not be scored as a correct answer."""
    result = drifted["result"].identity["synthetic_gps"]
    truth = dict(drifted["synthetic"].identity_map)
    truth["gps_999"] = "home_99"  # a pair the matcher never saw
    assert result.accuracy_against(truth) < 1.0


# ---------------------------------------------------------------------------
# Why it works: trajectories, not single frames
# ---------------------------------------------------------------------------

def test_correct_pairs_are_far_clearer_than_the_runner_up(drifted):
    """
    The margin between the chosen match and the next best candidate is what
    makes this robust. A margin of centimetres would mean the right answer was
    luck.
    """
    costs = drifted["result"].identity["synthetic_gps"].costs
    assert (costs["runner_up_margin_m"] > 1.0).all()
    assert costs["runner_up_margin_m"].median() > 5.0


def test_matched_distance_reflects_only_the_injected_noise(drifted):
    costs = drifted["result"].identity["synthetic_gps"].costs
    sigma = drifted["synthetic"].params.position_noise_m
    # Mean distance between two 2-D points differing by per-axis noise sigma is
    # about 1.25*sigma; allow generous slack for dropout and interpolation.
    assert costs["distance_m"].max() < 3.0 * sigma


def test_players_are_only_matched_within_their_own_team(drifted):
    mapping = drifted["result"].identity["synthetic_gps"].mapping
    truth_teams = (
        drifted["truth"].drop_duplicates("player_id").set_index("player_id")["team"]
    )
    synthetic = drifted["synthetic"].data
    synthetic_teams = (
        synthetic.drop_duplicates("player_id").set_index("player_id")["team"]
    )
    for synthetic_id, reference_id in mapping.items():
        assert synthetic_teams[synthetic_id] == truth_teams[reference_id]


# ---------------------------------------------------------------------------
# Behaviour under stress
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.3, 1.0, 2.0])
def test_accuracy_holds_across_realistic_noise_levels(metrica_home, sigma):
    synthetic = degrade(
        metrica_home,
        DegradationParams(position_noise_m=sigma, dropout_rate=0.1,
                          n_dropout_bursts=2, dropout_burst_s=4.0,
                          time_offset_s=2.0, seed=13),
    )
    result = _match(metrica_home, synthetic)
    accuracy = result.accuracy_against(synthetic.identity_map)
    assert accuracy >= ACCURACY_BAR, f"sigma={sigma} gave {accuracy:.1%}"


def test_independent_noise_is_averaged_away_over_a_long_window(metrica_home):
    """
    Documents why this synthetic test is *easier* than the real problem.

    Costs are mean distances over ~1800 frames, and independent zero-mean noise
    is precisely the error model averaging destroys. Accuracy therefore survives
    noise far larger than any real sensor's — the rejection threshold has to be
    lifted for the match to even be offered. Cross-vendor identity failures in
    practice come from *correlated* errors (a mirrored axis, sustained tracking
    swaps between nearby players, long absences), none of which this generator
    produces. See matrix_engine/README.md.
    """
    synthetic = degrade(
        metrica_home,
        DegradationParams(position_noise_m=25.0, dropout_rate=0.1,
                          n_dropout_bursts=1, time_offset_s=0.5, seed=17),
    )
    result = _match(metrica_home, synthetic, max_distance_m=1e6)
    assert result.accuracy_against(synthetic.identity_map) == 1.0
    assert result.costs["distance_m"].median() > 20.0


def test_the_runner_up_margin_shrinks_as_noise_grows(metrica_home):
    """
    Accuracy stays pinned at 100%, so the margin is the only visible sign that
    the problem is getting harder. It is the quantity to watch on real data.
    """
    margins = []
    for sigma in (1.0, 8.0, 25.0):
        synthetic = degrade(
            metrica_home,
            DegradationParams(position_noise_m=sigma, dropout_rate=0.1,
                              n_dropout_bursts=1, time_offset_s=0.5, seed=17),
        )
        result = _match(metrica_home, synthetic, max_distance_m=1e6)
        margins.append(float(result.costs["runner_up_margin_m"].median()))
    assert margins[0] > margins[1] > margins[2]


def test_implausible_matches_are_refused_rather_than_forced(metrica_home):
    """
    Hungarian assignment will pair everything up if allowed to. With a tight
    distance ceiling, wrong pairings should be dropped, not accepted.
    """
    synthetic = degrade(
        metrica_home,
        DegradationParams(position_noise_m=0.5, time_offset_s=2.0, seed=13),
    )
    result = _match(metrica_home, synthetic, max_distance_m=0.05)
    assert result.mapping == {}
    assert len(result.unmatched) == synthetic.data["player_id"].nunique()


def test_low_overlap_pairs_are_refused(metrica_home):
    synthetic = degrade(
        metrica_home,
        DegradationParams(position_noise_m=0.5, time_offset_s=2.0, seed=13),
    )
    result = _match(metrica_home, synthetic, min_overlap=10**9)
    assert result.mapping == {}


# ---------------------------------------------------------------------------
# Applying a mapping
# ---------------------------------------------------------------------------

def test_apply_mapping_relabels_into_the_reference_id_space(drifted):
    result = drifted["result"]
    synthetic = result.sync.data[result.sync.data["source"] == "synthetic_gps"]
    relabelled = apply_mapping(synthetic, result.identity["synthetic_gps"].mapping)
    reference_ids = set(drifted["truth"]["player_id"].unique())
    assert set(relabelled["player_id"].unique()) <= reference_ids


def test_apply_mapping_drops_unmapped_players_by_default(drifted):
    result = drifted["result"]
    synthetic = result.sync.data[result.sync.data["source"] == "synthetic_gps"]
    partial = dict(list(result.identity["synthetic_gps"].mapping.items())[:3])
    relabelled = apply_mapping(synthetic, partial)
    assert relabelled["player_id"].nunique() == 3


def test_apply_mapping_can_keep_unmapped_players(drifted):
    result = drifted["result"]
    synthetic = result.sync.data[result.sync.data["source"] == "synthetic_gps"]
    partial = dict(list(result.identity["synthetic_gps"].mapping.items())[:3])
    kept = apply_mapping(synthetic, partial, drop_unmatched=False)
    assert kept["player_id"].nunique() == synthetic["player_id"].nunique()


def test_empty_input_yields_no_matches(drifted):
    result = drifted["result"]
    synthetic = result.sync.data[result.sync.data["source"] == "synthetic_gps"]
    empty = synthetic.iloc[0:0]
    assert match_identities(empty, synthetic).mapping == {}
