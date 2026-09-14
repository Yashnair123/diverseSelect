from pathlib import Path

import numpy as np
import pandas as pd


ALPHAS = [0.05, 0.2, 0.35]
N_SIZES = np.linspace(10, 100, 10).astype(int)
M_SIZES = np.linspace(50, 500, 10).astype(int)
SKIP_SIZES = np.linspace(5, 50, 10).astype(int)
NUM_JOBS = 250
NUM_SETTINGS = 2
NUM_ALPHAS = len(ALPHAS)


def build_variants():
    variants = []
    for job in range(NUM_JOBS):
        for setting in range(NUM_SETTINGS):
            for alpha_ind in range(NUM_ALPHAS):
                variants.append((job, setting, alpha_ind))
    return variants


def read_latest_row(path, expected_len):
    if not path.exists():
        return None

    rows = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)

    for row in rows[::-1]:
        row = row[np.isfinite(row)]
        if len(row) == expected_len:
            return row

    return None


def main():
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "sharpe_scale_results"
    output_path = base_dir / "scale_sharpe.csv"
    variants = build_variants()

    records = []
    missing = 0
    incomplete = 0

    for variant, (job, setting, alpha_ind) in enumerate(variants):
        path = results_dir / f"times_v{variant}.csv"
        if not path.exists():
            missing += 1
            continue

        total_times = read_latest_row(path, len(N_SIZES))
        if total_times is None:
            incomplete += 1
            continue

        for size_ind, total_time in enumerate(total_times):
            records.append(
                {
                    "variant": variant,
                    "job": job,
                    "setting_ind": setting,
                    "setting": f"Setting {setting + 1}",
                    "alpha_ind": alpha_ind,
                    "alpha": ALPHAS[alpha_ind],
                    "alpha_label": f"alpha={ALPHAS[alpha_ind]}",
                    "size_ind": size_ind,
                    "n": N_SIZES[size_ind],
                    "m": M_SIZES[size_ind],
                    "skip": SKIP_SIZES[size_ind],
                    "total_time_raw": total_time,
                }
            )

    raw = pd.DataFrame.from_records(records)
    if raw.empty:
        raise RuntimeError(f"No complete Sharpe scaling results found in {results_dir}")

    summary = (
        raw.groupby(
            ["setting_ind", "setting", "alpha_ind", "alpha", "alpha_label", "size_ind", "n", "m", "skip"],
            as_index=False,
        )
        .agg(
            total_time=("total_time_raw", "mean"),
            total_time_sd=("total_time_raw", "std"),
            num_jobs=("total_time_raw", "count"),
        )
        .sort_values(["alpha_ind", "setting_ind", "n"])
    )
    summary["total_time_se"] = summary["total_time_sd"] / np.sqrt(summary["num_jobs"])

    summary.to_csv(output_path, index=False)
    print(f"Wrote {len(summary)} rows to {output_path}")
    print(f"Missing files: {missing}; incomplete files: {incomplete}")


if __name__ == "__main__":
    main()
