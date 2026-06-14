"""CLI orchestrator — runs the full triage pipeline end to end."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make `src/` importable when executed as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from triagem_fuzzy import config  # noqa: E402
from triagem_fuzzy.articulation.comparator import TriageComparator  # noqa: E402
from triagem_fuzzy.articulation.integrator import TriageIntegrator  # noqa: E402
from triagem_fuzzy.eda import ExploratoryAnalysis  # noqa: E402
from triagem_fuzzy.fuzzy.factory import (  # noqa: E402
    build_integrated_engine,
    build_standalone_engine,
)
from triagem_fuzzy.ingestion import DataIngestion  # noqa: E402
from triagem_fuzzy.ml.evaluator import ModelEvaluator  # noqa: E402
from triagem_fuzzy.ml.random_forest import TriageRandomForest  # noqa: E402
from triagem_fuzzy.preprocessing import DataPreprocessor  # noqa: E402
from triagem_fuzzy.reporting import ResultReporter  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Triagem Fuzzy pipeline runner")
    p.add_argument(
        "--pipeline",
        choices=("full", "eda", "ml", "fuzzy", "articulation"),
        default="full",
    )
    p.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    p.add_argument(
        "--articulation-sample",
        type=int,
        default=2000,
        help="Cap test rows fed to fuzzy comparator/integrator (speed).",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = DataIngestion().load_validated()
    print(f"[ingestion] loaded {len(df)} rows, {len(df.columns)} cols")

    if args.pipeline in ("full", "eda"):
        eda_report = ExploratoryAnalysis(args.output_dir).run(df)
        print(f"[eda] artifacts → {eda_report.artifacts_dir}")
        if args.pipeline == "eda":
            return 0
    else:
        eda_report = ExploratoryAnalysis(args.output_dir).run(df)

    prepared = DataPreprocessor().run(df)
    print(
        f"[prep] train={len(prepared.X_train)} test={len(prepared.X_test)} "
        f"features={len(prepared.feature_names)}"
    )

    model = TriageRandomForest().fit(prepared.X_train, prepared.y_train)
    model_path = args.output_dir / "models" / "random_forest.joblib"
    model.save(model_path)

    eval_report = ModelEvaluator(prepared.label_encoder).evaluate(
        model, prepared.X_test, prepared.y_test
    )
    print(
        f"[ml] accuracy={eval_report.accuracy:.3f} "
        f"macro_f1={eval_report.macro_f1:.3f} "
        f"weighted_f1={eval_report.weighted_f1:.3f}"
    )

    if args.pipeline == "ml":
        return 0

    X_test = prepared.X_test
    y_test = prepared.y_test
    if args.articulation_sample and len(X_test) > args.articulation_sample:
        sampled = X_test.sample(
            n=args.articulation_sample, random_state=config.RANDOM_STATE
        )
        X_test = sampled
        y_test = y_test.loc[sampled.index]
        print(f"[articulation] sampled {len(X_test)} rows for fuzzy passes")

    cmp_report = TriageComparator(
        model, build_standalone_engine(), prepared.label_encoder
    ).run(X_test, y_test)
    print(
        f"[A] agreement={cmp_report.agreement_rate:.3f} "
        f"ml_acc={cmp_report.ml_accuracy:.3f} "
        f"fuzzy_acc={cmp_report.fuzzy_accuracy:.3f}"
    )

    int_report = TriageIntegrator(
        model, build_integrated_engine(), prepared.label_encoder
    ).run(X_test, y_test)
    print(
        f"[B] acc_ml={int_report.accuracy_ml:.3f} "
        f"acc_int={int_report.accuracy_integrated:.3f} "
        f"f1_ml={int_report.macro_f1_ml:.3f} "
        f"f1_int={int_report.macro_f1_integrated:.3f}"
    )

    if args.pipeline in ("full",):
        report_path = ResultReporter(args.output_dir).assemble(
            eda_report, eval_report, cmp_report, int_report
        )
        print(f"[report] {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
