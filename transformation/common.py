import sys
from os import makedirs, path

from stable_baselines3.common.logger import (
    CSVOutputFormat,
    HumanOutputFormat,
    Logger,
    WandbWriter,
)


MODEL_PATH = path.abspath(path.join(__file__, "..", "..", "output", "models"))


def make_logger(project: str, run_name: str | None) -> Logger:
    if run_name is None:
        return Logger(project, [])  # dummy logger

    dir_path = path.abspath(path.join(__file__, "..", "..", "output", project))
    makedirs(dir_path, exist_ok=True)

    csv_path = path.abspath(path.join(dir_path, f"{run_name}.csv"))

    return Logger(
        folder="../.logs",
        output_formats=[
            HumanOutputFormat(sys.stdout),
            WandbWriter(),
            CSVOutputFormat(csv_path),
        ],
    )


def model_weight_path(project: str, run_name: str) -> str:
    dir_path = path.abspath(
        path.join(MODEL_PATH, project)
    )
    makedirs(dir_path, exist_ok=True)

    return path.abspath(path.join(dir_path, f"{run_name}.zip"))
