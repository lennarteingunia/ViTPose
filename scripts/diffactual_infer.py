"""diffactual inference shim for the ViTPose(++) fork.

Contract:

    python scripts/diffactual_infer.py (--video <FILE> | --frames <DIR>) --out <FILE.npz> \
        [--start <N>] [--count <M>] [--rotate {0,90ccw,90cw}] \
        [--config <YAML>] [--overrides <JSON>]

Runs an mmdet person detector + ViTPose top-down whole-body pose over a frame range and
writes one .npz holding *every* person detected in each frame, in a flat (CSR-style)
layout: `frame_det_counts` says how many detections belong to each frame, and the `det_*`
arrays are their concatenation. Subject selection and tracking are deliberately left to a
later stage, so they can be redone without re-running inference.

`--video` decodes frames directly, so a long recording needs no extracted PNGs on disk.
`--rotate` de-rotates in the same step, for footage shot with the camera turned on its
side; the keypoints are then in the *rotated* image frame, which `rotation` records.

Kept on the fork's `diffactual` branch so the fork stays reusable by other projects.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
from typing import Iterator, Sequence

import numpy as np

FORK_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))

DEFAULTS = {
    "pose_config": "configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/"
                   "coco-wholebody/ViTPose_huge_wholebody_256x192.py",
    "pose_checkpoint": "pretrained_models/vitpose_h_wholebody.pth",
    "det_config": "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
    "det_checkpoint": "pretrained_models/mmdet/"
                      "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    "device": "cuda:0",
    "bbox_thr": 0.3,
    "det_cat_id": 1,   # person class in the COCO detector
}

ROTATIONS = ("0", "90ccw", "90cw")


def _abs(path: str) -> str:
    return path if osp.isabs(path) else osp.join(FORK_ROOT, path)


def load_cfg(argv_config: str | None, overrides: dict) -> dict:
    cfg = dict(DEFAULTS)
    if argv_config:
        import yaml
        with open(argv_config) as fh:
            cfg.update({k: v for k, v in (yaml.safe_load(fh) or {}).items() if v is not None})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def list_frames(frames_dir: str) -> list[str]:
    frames = sorted(glob.glob(osp.join(frames_dir, "frame_*.png")))
    if not frames:
        frames = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg")
            for p in glob.glob(osp.join(frames_dir, ext))
        )
    if not frames:
        raise SystemExit(f"no frames found in {frames_dir}")
    return frames


def rotate_image(img: np.ndarray, rotation: str) -> np.ndarray:
    """Rotate a decoded BGR frame. `90ccw` matches `ffmpeg -vf transpose=2`."""
    if rotation == "0":
        return img
    import cv2
    if rotation == "90ccw":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "90cw":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    raise SystemExit(f"unknown rotation {rotation!r}, expected one of {ROTATIONS}")


def iter_video_frames(path: str, start: int, count: int | None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (source_frame_index, BGR image) for a frame range of a video.

    The index is read back from the decoder rather than counted, so it stays correct even
    when seeking lands off the requested frame.
    """
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {path}")
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        emitted = 0
        while count is None or emitted < count:
            index = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            emitted += 1
    finally:
        cap.release()


def iter_dir_frames(paths: Sequence[str], start: int, count: int | None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index into the full sorted listing, BGR image) for a frame range."""
    import cv2
    stop = len(paths) if count is None else min(len(paths), start + count)
    for index in range(start, stop):
        img = cv2.imread(paths[index], cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"cannot read frame {paths[index]}")
        yield index, img


def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--rotate", choices=ROTATIONS, default="0")
    ap.add_argument("--config", default=None)
    ap.add_argument("--overrides", default="{}")
    args = ap.parse_args()

    if args.start < 0:
        raise SystemExit("--start must be >= 0")
    if args.count is not None and args.count <= 0:
        raise SystemExit("--count must be > 0")

    cfg_d = load_cfg(args.config, json.loads(args.overrides))

    if args.video:
        frame_iter = iter_video_frames(args.video, args.start, args.count)
        source_name = args.video
    else:
        frame_iter = iter_dir_frames(list_frames(args.frames), args.start, args.count)
        source_name = args.frames

    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import (
        inference_top_down_pose_model,
        init_pose_model,
        process_mmdet_results,
    )
    from mmpose.datasets import DatasetInfo

    pose_model = init_pose_model(
        _abs(cfg_d["pose_config"]), _abs(cfg_d["pose_checkpoint"]), device=cfg_d["device"]
    )
    det_model = init_detector(
        _abs(cfg_d["det_config"]), _abs(cfg_d["det_checkpoint"]), device=cfg_d["device"]
    )

    dataset = pose_model.cfg.data["test"]["type"]
    dataset_info = DatasetInfo(pose_model.cfg.data["test"]["dataset_info"])
    n_kpt = pose_model.cfg.data_cfg["num_output_channels"]

    frame_indices: list[int] = []
    det_counts: list[int] = []
    det_bbox: list[np.ndarray] = []
    det_xy: list[np.ndarray] = []
    det_conf: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for index, frame in frame_iter:
        frame = rotate_image(frame, args.rotate)
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])

        det = inference_detector(det_model, frame)
        person_results = process_mmdet_results(det, cat_id=cfg_d["det_cat_id"])
        pose_results, _ = inference_top_down_pose_model(
            pose_model, frame, person_results,
            bbox_thr=cfg_d["bbox_thr"], format="xyxy",
            dataset=dataset, dataset_info=dataset_info,
            return_heatmap=False, outputs=None,
        )

        frame_indices.append(index)
        det_counts.append(len(pose_results))
        for res in pose_results:
            kp = res["keypoints"]           # (n_kpt, 3): x, y, score
            det_bbox.append(np.asarray(res["bbox"], dtype=np.float32))
            det_xy.append(kp[:, :2].astype(np.float32))
            det_conf.append(kp[:, 2].astype(np.float32))

    if not frame_indices:
        raise SystemExit(f"no frames read from {source_name} at --start {args.start}")

    def _stack(items: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
        return np.stack(items).astype(np.float32) if items else np.zeros((0, *shape), np.float32)

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        frame_det_counts=np.asarray(det_counts, dtype=np.int64),
        det_bbox=_stack(det_bbox, (5,)),
        det_xy=_stack(det_xy, (n_kpt, 2)),
        det_conf=_stack(det_conf, (n_kpt,)),
        layout=np.array(dataset_info.dataset_name),
        rotation=np.array(args.rotate),
        image_size=np.asarray(image_size, dtype=np.int64),
        source=np.array(source_name),
        pose_config=np.array(_abs(cfg_d["pose_config"])),
        pose_checkpoint=np.array(_abs(cfg_d["pose_checkpoint"])),
        det_config=np.array(_abs(cfg_d["det_config"])),
        det_checkpoint=np.array(_abs(cfg_d["det_checkpoint"])),
        bbox_thr=np.array(cfg_d["bbox_thr"], dtype=np.float32),
        det_cat_id=np.array(cfg_d["det_cat_id"], dtype=np.int64),
    )
    print(
        f"wrote {args.out}  ({len(frame_indices)} frames, {len(det_xy)} detections, "
        f"{n_kpt} keypoints, rotation={args.rotate})"
    )


if __name__ == "__main__":
    main()
