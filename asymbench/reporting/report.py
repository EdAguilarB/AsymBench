from datetime import datetime
import json
from pathlib import Path


def create_report(config, metrics, explanations):
    report = {
        "timestamp": str(datetime.now()),
        "config": config,
        "metrics": metrics,
        "explanations_summary": "SHAP computed",
    }

    Path("reports").mkdir(exist_ok=True)
    fname = f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(fname, "w") as f:
        json.dump(report, f, indent=2)
