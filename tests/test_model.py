import pandas as pd

from qdtemp.model import FEATURE_COLUMNS, group_split, prepare_modeling_table


def test_prepare_modeling_table_groups_rows():
    df = pd.DataFrame({
        "qd_id": [1, 1, 2, 2],
        "temperature_K": [10, 10, 20, 20],
        "peak_eV": [2.4, 2.5, 2.3, 2.4],
        "fwhm_nm": [0.4, 0.5, 0.6, 0.7],
        "tau_moment_ns_raw": [5, 6, 7, 8],
        "tau_ampw_ns_raw": [1, 2, 3, 4],
        "tau_intw_ns_raw": [10, 11, 12, 13],
    })
    d_model, features = prepare_modeling_table(df, FEATURE_COLUMNS)
    assert d_model.shape[0] == 2
    assert set(features) == set(FEATURE_COLUMNS)


def test_group_split_no_overlap():
    df = pd.DataFrame({
        "qd_id": list(range(10)),
        "temperature_K": [10, 20, 40, 60, 80, 100, 110, 30, 50, 70],
        "peak_eV": [2.4] * 10,
        "fwhm_nm": [0.5] * 10,
        "tau_moment_ns_raw": [6] * 10,
        "tau_ampw_ns_raw": [2] * 10,
        "tau_intw_ns_raw": [14] * 10,
    })
    split = group_split(df, FEATURE_COLUMNS, test_size=0.3, random_state=1)
    train_groups = set(split.groups_train)
    test_groups = set(split.groups_test)
    assert train_groups.isdisjoint(test_groups)
