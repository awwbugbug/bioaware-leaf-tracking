import sys

sys.path.insert(0, "/root/autodl-tmp/LeafTrackNet-main/TrackEval")
import trackeval

eval_config = trackeval.Evaluator.get_default_eval_config()
eval_config["USE_PARALLEL"] = False
eval_config["PRINT_CONFIG"] = False
eval_config["PLOT_CURVES"] = False

dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dataset_config["GT_FOLDER"] = (
    "/root/autodl-tmp/LeafTrackNet-main/datasets/CanolaTrack/CanolaTrack/val"
)
dataset_config["TRACKERS_FOLDER"] = (
    "/root/autodl-tmp/LeafTrackNet-main/outputs/centroid"
)
dataset_config["TRACKERS_TO_EVAL"] = ["tracks"]
dataset_config["TRACKER_SUB_FOLDER"] = ""
dataset_config["SKIP_SPLIT_FOL"] = True
dataset_config["SPLIT_TO_EVAL"] = "val"
dataset_config["SEQMAP_FILE"] = "/root/autodl-tmp/LeafTrackNet-main/seqmap"
dataset_config["PRINT_CONFIG"] = False

metrics_list = [
    trackeval.metrics.HOTA(),
    trackeval.metrics.CLEAR(),
    trackeval.metrics.Identity(),
]

evaluator = trackeval.Evaluator(eval_config)
dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
evaluator.evaluate(dataset_list, metrics_list)
