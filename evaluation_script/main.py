from __future__ import annotations

import contextlib
import copy
import csv
import io
import os
import time
import sys
import zipfile
from pathlib import Path
import requests
from argparse import ArgumentParser

import numpy as np
import scalabel
from scalabel.eval.ins_seg import evaluate_ins_seg
from scalabel.label.io import load

sys.path.append(str(Path(__file__).parent.absolute()))

from .depth_eval import DepthEvaluator
from .det3d_eval import evaluate_det_3d

CONDITIONS = [
    "clear",
    "overcast",
    "rainy",
    "foggy",
    "cloudy",
    "daytime",
    "dawn/dusk",
    "night",
]
SEQ_INFO_PATH = os.path.join(
    str(Path(__file__).parent.absolute()), "val_front_images_seq.csv"
)
VAL_GT_PATH = os.path.join(str(Path(__file__).parent.absolute()), "val_gt.zip")
TEST_GT_PATH = os.path.join(str(Path(__file__).parent.absolute()), "test_gt.zip")

SCALABEL_CACHE = {}


# Load sequence info
with open(SEQ_INFO_PATH, "r") as f:
    reader = csv.DictReader(f)
    SEQ_INFO = [
        (row["video"], row["start_weather_coarse"], row["start_timeofday_coarse"])
        for row in reader
    ]


def add_score_to_frames(frames):
    for frame in frames:
        for label in frame.labels:
            if label.score is None:
                label.score = 1.0
    return frames


def get_used_seqs(seq_filter):
    if seq_filter is not None:
        used_seqs = []
        for seq in SEQ_INFO:
            if seq_filter in ["clear", "overcast", "rainy", "foggy", "cloudy"]:
                if seq[1] == seq_filter:
                    used_seqs.append(seq[0])
            elif seq_filter in ["daytime", "dawn/dusk", "night"]:
                if seq[2] == seq_filter:
                    used_seqs.append(seq[0])
    else:
        used_seqs = [seq[0] for seq in SEQ_INFO]
    return used_seqs


def load_scalabel(file_path, used_seqs=None):
    if file_path in SCALABEL_CACHE:
        data_ = copy.deepcopy(SCALABEL_CACHE[file_path])
    else:
        data = load(file_path, validate_frames=False)
        SCALABEL_CACHE[file_path] = data
        data_ = copy.deepcopy(data)
    if used_seqs is not None:
        data_.frames = [frame for frame in data_.frames if frame.videoName in used_seqs]
    return data_


def filter_scalabel(pred, target):
    used_frames = []
    for frame in target.frames:
        used_frames.append(frame.videoName + frame.name)
    used_frames = set(used_frames)
    pred.frames = [
        frame for frame in pred.frames if frame.videoName + frame.name in used_frames
    ]
    return pred


def evaluate_shift_multitask(
    test_annotation_dir, user_submission_dir, seq_filter=None, max_num_seqs=-1
):
    if seq_filter is not None:
        assert seq_filter in CONDITIONS, (
            "seq_filter must be one of clear, overcast, rainy, foggy, cloudy, daytime, dawn/dusk, night, or None, "
            "but got {}".format(seq_filter)
        )
    used_seqs = get_used_seqs(seq_filter)
    if max_num_seqs > 0 and max_num_seqs < len(used_seqs):
        used_seqs = used_seqs[:max_num_seqs]

    result_dict = {}

    # Instance segmentation
    if os.path.exists(os.path.join(user_submission_dir, "det_insseg_2d.json")):
        print(">> Evaluating instance segmentation...")
        ins_seg_pred = load_scalabel(
            os.path.join(user_submission_dir, "det_insseg_2d.json"), used_seqs
        )
        ins_seg_target = load_scalabel(
            os.path.join(test_annotation_dir, "det_insseg_2d.json"), used_seqs
        )
        ins_seg_pred = filter_scalabel(ins_seg_pred, ins_seg_target)
        print(len(ins_seg_pred.frames), len(ins_seg_target.frames))
        with contextlib.redirect_stdout(io.StringIO()):
            ins_seg_result = evaluate_ins_seg(
                ins_seg_target.frames,
                add_score_to_frames(ins_seg_pred.frames),
                ins_seg_target.config,
                nproc=1,
            )
        ins_seg_result = ins_seg_result.summary()
        result_dict["insseg/mAP"] = ins_seg_result["AP"]
        print(">> Instance segmentation results:\n", ins_seg_result)

    # Depth estimation
    if os.path.exists(os.path.join(user_submission_dir, "depth")):
        print(">> Evaluating depth estimation...")
        depth_eval = DepthEvaluator()
        depth_eval.process_from_folder(
            os.path.join(user_submission_dir, "depth"),
            os.path.join(test_annotation_dir, "depth"),
            max_num_seqs=max_num_seqs,
            used_seqs=used_seqs,
        )
        depth_result = depth_eval.evaluate()
        # result_dict["depth/AbsErr"] = depth_result["mae"]
        result_dict["depth/SILog"] = depth_result["silog"]
        print(">> Depth estimation results:\n", depth_result)

    # 3D detection
    if os.path.exists(os.path.join(user_submission_dir, "det_3d.json")):
        print(">> Evaluating 3D detection...")
        det_3d_pred = load_scalabel(
            os.path.join(user_submission_dir, "det_3d.json"), used_seqs
        )
        det_3d_target = load_scalabel(
            os.path.join(test_annotation_dir, "det_3d.json"), used_seqs
        )
        det_3d_pred = filter_scalabel(det_3d_pred, det_3d_target)
        with contextlib.redirect_stdout(io.StringIO()):
            det_3d_result = evaluate_det_3d(
                det_3d_target.frames,
                det_3d_pred.frames,
                det_3d_target.config,
            )
        result_dict["det3d/mAP"] = det_3d_result["mean_ap"] * 100
        result_dict["det3d/mTPS"] = (
            (
                det_3d_result["tp_scores"]["trans_err"]
                + det_3d_result["tp_scores"]["orient_err"]
                + det_3d_result["tp_scores"]["scale_err"]
            )
            * 100
            / 3
        )
        print(">> 3D detection results:\n", det_3d_result)

    # Multitask metrics
    if "insseg/mAP" in result_dict and "depth/SILog" in result_dict:
        result_dict["Overall"] = (
            result_dict["insseg/mAP"]
            + (result_dict["det3d/mAP"] + result_dict["det3d/mTPS"]) / 2.0
            + np.clip(50 - result_dict["depth/SILog"], 0, 50) * 2.0
        ) / 3.0
    return result_dict


def evaluate_shift(test_annotation_dir, user_submission_dir):
    result_dict = {}
    for seq_filter in CONDITIONS:
        print("> Evaluating for condition: {}".format(seq_filter))
        result_dict[seq_filter] = evaluate_shift_multitask(
            test_annotation_dir, user_submission_dir, seq_filter
        )

    # Overall metrics
    overall_dict = {}
    for seq_filter in CONDITIONS:
        for metric in result_dict[seq_filter]:
            if metric not in overall_dict:
                overall_dict[metric] = []
            overall_dict[metric].append(result_dict[seq_filter][metric])
    for metric in overall_dict:
        overall_dict[metric] = np.mean(overall_dict[metric])

    # Add VTD metrics of all conditions
    if "Overall" in overall_dict:
        for seq_filter in CONDITIONS:
            overall_dict["Overall/{}".format(seq_filter)] = result_dict[seq_filter][
                "Overall"
            ]
    return overall_dict


def unzip_nested(file_path):
    assert file_path.endswith(".zip")
    output_path = file_path[:-4]
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(output_path)
    # for nested zip files
    for file in os.listdir(output_path):
        if file.endswith(".zip"):
            unzip_nested(os.path.join(output_path, file))


def download(phase, filename):
    if phase == "dev":
        url = "https://dl.cv.ethz.ch/shift/challenge2023/multitask_robustness/SHIFT_challenge2023_multitask_minival_gt.zip"

    tic = time.time()
    req = requests.get(url, timeout=10)
    with open(filename, "wb") as output_file:
        output_file.write(req.content)
    time_used = time.time() - tic
    print(f"Downloading finished in {time_used:.1f} sec.")


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    print("Starting Evaluation.....")
    print("\nEvaluation environments:")
    print(" - Python version:", sys.version.split(" ")[0])
    print(" - NumPy version:", np.__version__)
    print(" - Scalabel version:", scalabel.__version__)

    assert (
        test_annotation_file[-4:] == ".zip"
    ), "Test annotation file should be a zip file"
    assert (
        user_submission_file[-4:] == ".zip"
    ), "User submission file should be a zip file"

    # Unzip the annotation files
    print("\nStart unzipping...")
    user_submission_dir = user_submission_file[:-4]
    test_annotation_dir = test_annotation_file[:-4]
    print("> ", user_submission_file)
    unzip_nested(user_submission_file)
    print("> ", test_annotation_file)
    unzip_nested(test_annotation_file)
    print("Unzipping completed.")

    output = {}
    if phase_codename == "dev":
        print("Evaluation phase: Dev")
        result_dict = evaluate_shift(test_annotation_dir, user_submission_dir)
        output["result"] = [{"val_split": result_dict}]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["val_split"]
        print("Completed evaluation for Dev Phase")
        print(result_dict)
    elif phase_codename == "test":
        print("Evaluation phase: Test")
        result_dict = evaluate_shift(test_annotation_dir, user_submission_dir)
        output["result"] = [{"val_split": result_dict}, {"test_split": result_dict}]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]
        print("Completed evaluation for Test Phase")
    return output


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate the submission.")
    parser.add_argument('--target', type=str, help='path to the ground truth file')
    parser.add_argument('--pred', type=str, help='path to the result zip file')
    args = parser.parse_args()

    # test
    evaluate(
        args.target,
        args.pred,
        "dev",
    )
