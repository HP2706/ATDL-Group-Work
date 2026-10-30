import fire

from video3d.baselines.api import main
from video3d.baselines.analysis import analyze
from video3d.baselines.evaluate import evaluate

fire.Fire({"run": main, "evaluate": evaluate, "analyze": analyze})
