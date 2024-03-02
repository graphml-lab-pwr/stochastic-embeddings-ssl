import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer

model_dict = {
    "rp1pns5t": "ssl_bayes_h; vicreg",
    "lsqfidza": "ssl_bayes_h; barlow",
    "jhk2cig6": "ssl;         barlow",
    "9d6810ku": "ssl_bayes_h; vicreg",
    "303yffuk": "ssl;         vicreg",
    "pbtq1sck": "ssl_bayes_h; barlow",
    "ugy9l6an": "ssl_bayes_h; vicreg",
    "cek669mh": "ssl_bayes_z; barlow",
    "tp1kecgw": "ssl_bayes_z; vicreg",
    "p449f0wf": "ssl_bayes_h; vicreg",
}


def main(
    root_dir: Path = typer.Option(...),
):
    results: list[dict[str, Any]] = []

    for metrics_file in root_dir.rglob("metrics.json"):
        exp_dir = metrics_file.parent
        *_, model_id, dataset, _, _ = exp_dir.parts

        with metrics_file.open() as file:
            metrics = json.load(file)

        ckpt_path = _get_best_checkpoint_path(exp_dir)
        best_epoch = int(ckpt_path.stem.split("=")[1].split("-")[0])

        res_item = {
            "dataset": dataset,
            "model_id": model_id,
            "model_name": model_dict.get(model_id, model_id),
            "epoch": best_epoch,
        }
        res_item |= metrics

        results.append(res_item)

    df = pd.DataFrame(results).sort_values(["dataset", "model_name"])

    report_dir = root_dir.parent / "reports"
    report_dir.mkdir(exist_ok=True)
    df.to_csv(
        report_dir / "linear_probing_report.csv",
        float_format="%.4f",
        index=False,
    )
    df.to_markdown(report_dir / "linear_probing_report.md", index=False)
    df.to_latex(
        report_dir / "linear_probing_report.tex",
        float_format="%.3f",
        index=False,
    )


def _get_best_checkpoint_path(exp_dir: Path) -> Path:
    ckpt_dir = exp_dir / "checkpoints"
    checkpoints = [
        ckpt_path
        for ckpt_path in ckpt_dir.glob("*.ckpt")
        if ckpt_path.stem.startswith("epoch")
    ]
    assert len(checkpoints) == 1
    ckpt_path, *_ = checkpoints
    return ckpt_path


if __name__ == "__main__":
    typer.run(main)
