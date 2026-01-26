# utils/logging.py
import os
import sys
from datetime import datetime
from utils.tee_logger import Tee


def redirect_output_per_run(
    repo_owner: str,
    repo_name: str,
    model_name: str,
    v_source: str,
    v_target: str,
    base_dir: str = "logs"
):
    safe_model = model_name.replace(":", "_").replace("/", "_")
    safe_range = f"{v_source}_to_{v_target}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = os.path.join(
        base_dir,
        repo_owner,
        repo_name,
        safe_model
    )
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(
        log_dir,
        f"{safe_range}_{timestamp}.txt"
    )

    log_file = open(log_path, "w", encoding="utf-8")

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    return log_path
