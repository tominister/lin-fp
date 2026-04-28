from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_model_metric_plot(model_name: str, model_results: dict, output_path: Path):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    val_metrics = model_results["welfake_validation_20pct"]
    recovery_metrics = model_results["recovery_external_test"]

    x = np.arange(len(metric_names))
    width = 0.36

    val_values = [float(val_metrics[name]) for name in metric_names]
    recovery_values = [float(recovery_metrics[name]) for name in metric_names]

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, val_values, width=width, label="WELFake holdout")
    plt.bar(x + width / 2, recovery_values, width=width, label="Recovery external")
    plt.xticks(x, [name.upper() for name in metric_names])
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title(f"{model_name} Performance: Holdout vs External")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
