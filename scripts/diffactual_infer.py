"""diffactual inference shim for the ViTPose(++) fork.

Contract:

    python scripts/diffactual_infer.py (--video <FILE> | --frames <DIR>) --out <FILE.npz> \
        [--start <N>] [--count <M>] [--rotate {0,90ccw,90cw}] \
        [--batch <B>] [--no-prefetch] \
        [--config <YAML>] [--overrides <JSON>]

Runs an mmdet person detector + ViTPose top-down whole-body pose over a frame range and
writes one .npz holding *every* person detected in each frame, in a flat (CSR-style)
layout: `frame_det_counts` says how many detections belong to each frame, and the `det_*`
arrays are their concatenation. Subject selection and tracking are deliberately left to a
later stage, so they can be redone without re-running inference.

`--video` decodes frames directly, so a long recording needs no extracted PNGs on disk.
`--rotate` de-rotates in the same step, for footage shot with the camera turned on its
side; the keypoints are then in the *rotated* image frame, which `rotation` records.

The detector runs in **half precision** by default (`det_fp16`), which measured 36.3 -> 27.4 ms
on an A5000 while leaving the boxes alone: the same 3.031 detections per frame and median IoU
0.9994 against fp32, over two camera views. It is skipped with a note on a non-CUDA device.
Set `det_fp16: false` through `--overrides` to go back to fp32.

`--batch` processes B frames together: one detector forward pass for the B frames, then one
pose forward pass for all their detections. mmpose already batches the detections *within* a
frame (`mmpose/apis/inference.py:280`), so at 3.03 detections per frame that goes from batch
3 to batch 3B. Detection stays strictly per frame at any B — no box is ever reused between
frames, and `--batch` changes only how work is grouped, never what is computed. Decoding and
the detector's preprocessing run a batch ahead on a worker thread unless `--no-prefetch`; on
this footage that is ~17 ms of CPU per frame otherwise spent in front of the GPU rather than
beside it.

Kept on the fork's `diffactual` branch so the fork stays reusable by other projects.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import sys
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
    "det_fp16": True,   # task 20: half precision on the detector, ~25 % off its forward pass
    "bbox_thr": 0.3,
    "det_cat_id": 1,   # person class in the COCO detector
}

ROTATIONS = ("0", "90ccw", "90cw")

# Frames per detector forward pass. 8 was the fastest configuration measured on an A5000
# (8.73 fps against 8.56 at batch 1, with fp16); the margin over batch 4 is inside run-to-run
# noise, so 4 is equally defensible if the buffered frames ever cost too much RAM.
DEFAULT_BATCH = 8

# Batches held ready ahead of the GPU. Two is enough to keep it fed; more only costs RAM,
# and a prepared frame carries both the decoded image and its float32 detector tensor.
PREFETCH_DEPTH = 2


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


def apply_det_fp16(det_model, enabled: bool, device: str) -> bool:
    """Wrap the detector for half-precision inference. Returns whether it was applied.

    `wrap_fp16_model` casts the weights to half, which most CPU kernels do not implement,
    so on a non-CUDA device this is skipped with a note rather than raised: the CPU paths
    (`scripts/smoke_models.py`, the fixture runs) have to keep working. Measured on an
    A5000 over two views: detector 36.3 -> 27.4 ms, boxes unchanged to median IoU 0.9994
    with the same 3.031 detections per frame, see `notes/detector-selection.md`.
    """
    if not enabled:
        return False
    if not str(device).startswith("cuda"):
        print(f"det_fp16: skipped, half precision needs a CUDA device, got {device}",
              file=sys.stderr)
        return False
    from mmcv.runner import wrap_fp16_model
    wrap_fp16_model(det_model)
    return True


def build_det_pipeline(det_model):
    """The detector's own test pipeline, built once instead of once per frame.

    `mmdet.apis.inference_detector` rebuilds this on every call (`mmdet/apis/inference.py:122`)
    and then runs it inline, so the resize/normalise/pad sits in front of the GPU. Hoisting it
    out is what lets `iter_prepared` run it on the prefetch thread instead.
    """
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose

    cfg = det_model.cfg.copy()
    # The frames are decoded arrays, not paths, which is the loader mmdet swaps in for that.
    cfg.data.test.pipeline[0].type = "LoadImageFromWebcam"
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    return Compose(cfg.data.test.pipeline)


def iter_prepared(frame_iter, rotation: str, det_pipeline) -> Iterator[tuple]:
    """Yield `(index, frame, prepared)` — the whole CPU half of a frame's work.

    Everything here is numpy and OpenCV, which release the GIL, so this is what `prefetch`
    moves off the critical path. The frame is copied because it outlives this step: the pose
    model reads it after the detector has run, and `rotate_image` passes the decoder's buffer
    straight through when the rotation is `0`.
    """
    for index, frame in frame_iter:
        frame = rotate_image(frame, rotation).copy()
        yield index, frame, det_pipeline(dict(img=frame))


def iter_batches(items: Iterator, size: int) -> Iterator[list]:
    """Group an iterator into lists of at most `size`, the last one short."""
    if size < 1:
        raise SystemExit(f"batch size must be >= 1, got {size}")
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def prefetch(items: Iterator, depth: int) -> Iterator:
    """Yield from `items` in order, produced `depth` items ahead on a worker thread.

    The worker's exception is re-raised here rather than ending the run quietly, so a decode
    failure still stops the pass instead of silently truncating the output.
    """
    import queue
    import threading

    done = object()
    slots: queue.Queue = queue.Queue(maxsize=depth)

    def produce() -> None:
        try:
            for item in items:
                slots.put((item, None))
        except BaseException as exc:  # surfaced in the consumer, below
            slots.put((None, exc))
        finally:
            slots.put((done, None))

    # Daemon: if the consumer stops early the producer may block on a full queue forever,
    # and this is a batch script that otherwise runs the iterator dry.
    threading.Thread(target=produce, daemon=True).start()
    while True:
        item, exc = slots.get()
        if exc is not None:
            raise exc
        if item is done:
            return
        yield item


def detect_batch(det_model, prepared: list) -> list:
    """Run the detector over already-preprocessed frames, one forward pass for the batch.

    This is the tail of `mmdet.apis.inference_detector` (`mmdet/apis/inference.py:142-158`).
    That function cannot be used here because it insists on doing its own preprocessing,
    which is the part being moved onto the prefetch thread.
    """
    import torch
    from mmcv.parallel import collate, scatter

    data = collate(prepared, samples_per_gpu=len(prepared))
    data["img_metas"] = [metas.data[0] for metas in data["img_metas"]]
    data["img"] = [img.data[0] for img in data["img"]]
    device = next(det_model.parameters()).device
    if device.type != "cpu":
        data = scatter(data, [device])[0]
    with torch.no_grad():
        return det_model(return_loss=False, rescale=True, **data)


def build_pose_pipeline(pose_model):
    """The pose model's test pipeline, built once instead of once per frame.

    `_inference_single_pose_model` rebuilds it on every call
    (`mmpose/apis/inference.py:149`), the same waste `build_det_pipeline` removes.
    """
    from mmpose.datasets.pipelines import Compose

    return Compose(pose_model.cfg.test_pipeline)


def threshold_boxes(persons: list, bbox_thr: float | None):
    """Drop detections scoring at or below `bbox_thr`, keeping order.

    Returns `(kept persons, their xyxy+score array)`; the array is empty when nothing
    survives. Mirrors the filter in `inference_top_down_pose_model`
    (`mmpose/apis/inference.py:378-382`) — strictly greater than the threshold, so a
    detection exactly on it is dropped, as upstream does.
    """
    if not persons:
        return [], np.zeros((0, 5), dtype=np.float32)
    xyxy = np.array([person["bbox"] for person in persons])
    if bbox_thr is None:
        return list(persons), xyxy
    if xyxy.shape[1] != 5:
        raise SystemExit(
            f"--bbox_thr needs scored boxes, got {xyxy.shape[1]} values per box"
        )
    valid = np.where(xyxy[:, 4] > bbox_thr)[0]
    return [persons[i] for i in valid], xyxy[valid]


def pose_batch(pose_model, pose_pipeline, dataset_info, frames, boxes, bbox_thr: float):
    """Pose every detection of several frames in one forward pass.

    mmpose 0.24 has no batch-across-images entry point: `inference_top_down_pose_model`
    takes a single image and batches only that image's boxes
    (`mmpose/apis/inference.py:280`). At 3.03 detections per frame it is therefore already
    running batch 3 — this widens that to batch 3B, which is what is left of task 19 once
    the detector is batched.

    `frames` and `boxes` are parallel lists, `boxes[i]` being the mmdet person results for
    `frames[i]`. Returns one list per frame, in order, each holding the same dicts
    `inference_top_down_pose_model` returns: the box as `xyxy+score` and `keypoints` (K, 3).
    """
    import numpy as np
    import torch
    from mmcv.parallel import collate, scatter
    from mmpose.apis.inference import _box2cs, _xyxy2xywh

    cfg = pose_model.cfg
    batch_data, owners = [], []
    kept: list[list[dict]] = []
    for frame_i, (frame, persons) in enumerate(zip(frames, boxes)):
        persons, xyxy = threshold_boxes(persons, bbox_thr)
        kept.append([dict(person) for person in persons])
        if not len(xyxy):
            continue
        for box_i, (xywh, box_xyxy) in enumerate(zip(_xyxy2xywh(xyxy), xyxy)):
            center, scale = _box2cs(cfg, xywh)
            kept[frame_i][box_i]["bbox"] = box_xyxy
            batch_data.append(pose_pipeline({
                "img": frame,
                "center": center,
                "scale": scale,
                "bbox_score": xywh[4] if len(xywh) == 5 else 1,
                # Distinct per sample: mmpose's own version hardcodes 0 and notes that it
                # "need[s] to be assigned if batch_size > 1" (`inference.py:258`).
                "bbox_id": len(batch_data),
                "dataset": dataset_info.dataset_name,
                "joints_3d": np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
                "joints_3d_visible": np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
                "rotation": 0,
                "ann_info": {
                    "image_size": np.array(cfg.data_cfg["image_size"]),
                    "num_joints": cfg.data_cfg["num_joints"],
                    "flip_pairs": dataset_info.flip_pairs,
                },
            }))
            owners.append((frame_i, box_i))

    if not batch_data:
        return kept

    device = next(pose_model.parameters()).device
    data = collate(batch_data, samples_per_gpu=len(batch_data))
    data = scatter(data, [-1 if device.type == "cpu" else device])[0]
    with torch.no_grad():
        result = pose_model(img=data["img"], img_metas=data["img_metas"],
                            return_loss=False, return_heatmap=False)

    for (frame_i, box_i), pose in zip(owners, result["preds"]):
        kept[frame_i][box_i]["keypoints"] = pose
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--rotate", choices=ROTATIONS, default="0")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help="frames per detector forward pass (pose stays per frame); "
                         f"default {DEFAULT_BATCH}")
    ap.add_argument("--no-prefetch", dest="prefetch", action="store_false",
                    help="decode and preprocess inline instead of on a worker thread")
    ap.add_argument("--config", default=None)
    ap.add_argument("--overrides", default="{}")
    args = ap.parse_args()

    if args.start < 0:
        raise SystemExit("--start must be >= 0")
    if args.count is not None and args.count <= 0:
        raise SystemExit("--count must be > 0")
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")

    cfg_d = load_cfg(args.config, json.loads(args.overrides))

    if args.video:
        frame_iter = iter_video_frames(args.video, args.start, args.count)
        source_name = args.video
    else:
        frame_iter = iter_dir_frames(list_frames(args.frames), args.start, args.count)
        source_name = args.frames

    from mmdet.apis import init_detector
    from mmpose.apis import init_pose_model, process_mmdet_results
    from mmpose.datasets import DatasetInfo

    pose_model = init_pose_model(
        _abs(cfg_d["pose_config"]), _abs(cfg_d["pose_checkpoint"]), device=cfg_d["device"]
    )
    det_model = init_detector(
        _abs(cfg_d["det_config"]), _abs(cfg_d["det_checkpoint"]), device=cfg_d["device"]
    )
    det_fp16_applied = apply_det_fp16(det_model, cfg_d["det_fp16"], cfg_d["device"])

    dataset_info = DatasetInfo(pose_model.cfg.data["test"]["dataset_info"])
    pose_pipeline = build_pose_pipeline(pose_model)
    n_kpt = pose_model.cfg.data_cfg["num_output_channels"]

    frame_indices: list[int] = []
    det_counts: list[int] = []
    det_bbox: list[np.ndarray] = []
    det_xy: list[np.ndarray] = []
    det_conf: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    batches = iter_batches(
        iter_prepared(frame_iter, args.rotate, build_det_pipeline(det_model)), args.batch
    )
    if args.prefetch:
        batches = prefetch(batches, PREFETCH_DEPTH)

    for batch in batches:
        frames = [frame for _, frame, _ in batch]
        dets = detect_batch(det_model, [prepared for _, _, prepared in batch])
        boxes = [process_mmdet_results(det, cat_id=cfg_d["det_cat_id"]) for det in dets]
        batch_poses = pose_batch(pose_model, pose_pipeline, dataset_info, frames, boxes,
                                 bbox_thr=cfg_d["bbox_thr"])

        for (index, frame, _), pose_results in zip(batch, batch_poses):
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])

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
        # Both of these change the numbers below, so the file has to carry them: fp16 moves
        # the boxes by ~1e-4 px and `--batch > 1` by ~2e-4, either of which can flip a
        # heatmap-argmax tie. See `notes/detector-selection.md`.
        det_fp16=np.array(bool(det_fp16_applied)),
        batch=np.array(args.batch, dtype=np.int64),
    )
    print(
        f"wrote {args.out}  ({len(frame_indices)} frames, {len(det_xy)} detections, "
        f"{n_kpt} keypoints, rotation={args.rotate})"
    )


if __name__ == "__main__":
    main()
