"""Spec 03 acceptance tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from triagem_fuzzy.eda import EdaReport, ExploratoryAnalysis


def test_describe_returns_expected_columns(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    eda = ExploratoryAnalysis(tmp_path)
    summary = eda.describe(sample_valid_df)
    for col in ("mean", "std", "min", "25%", "50%", "75%", "max"):
        assert col in summary.columns


def test_class_balance_raw_has_four_levels(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    eda = ExploratoryAnalysis(tmp_path)
    s = eda.class_balance(sample_valid_df, remap=False)
    assert set(s.index) <= {0, 1, 2, 3}


def test_class_balance_remapped_has_three_levels(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    eda = ExploratoryAnalysis(tmp_path)
    s = eda.class_balance(sample_valid_df, remap=True)
    assert list(s.index) == ["normal", "atencao", "risco"]
    assert s.sum() == len(sample_valid_df)


def test_run_writes_all_expected_files(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    eda = ExploratoryAnalysis(tmp_path)
    report: EdaReport = eda.run(sample_valid_df)

    for name in (
        "summary_statistics.csv",
        "class_balance_raw.csv",
        "class_balance_remapped.csv",
        "correlation_matrix.csv",
    ):
        assert (report.artifacts_dir / name).exists()

    # 8 numeric features → 8 histograms + 8 boxplots.
    pngs = list(report.artifacts_dir.glob("*.png"))
    assert sum(1 for p in pngs if p.name.startswith("dist_")) == 8
    assert sum(1 for p in pngs if p.name.startswith("box_")) == 8


def test_run_does_not_mutate_input(
    tmp_path: Path, sample_valid_df: pd.DataFrame
) -> None:
    snapshot = sample_valid_df.copy(deep=True)
    ExploratoryAnalysis(tmp_path).run(sample_valid_df)
    pd.testing.assert_frame_equal(sample_valid_df, snapshot)
