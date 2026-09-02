"""diffactual inference shim for the ViTPose(++) fork.

Contract (see diffactual/external/README.md):

    python scripts/diffactual_infer.py --frames <DIR> --out <FILE.npz> \
        [--config <YAML>] [--overrides <JSON>]

Runs an mmdet person detector + ViTPose top-down whole-body pose per frame and writes
one .npz with COCO-WholeBody 2D keypoints, in the schema `diffactual/external/vitpose.md`
documents (xy (T,133,2), confidence (T,133), source_frame_indices (T,), layout).

Kept on the fork's `diffactual` branch so the fork stays reusable by other projects.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--overrides", default="{}")
    args = ap.parse_args()

    cfg_d = load_cfg(args.config, json.loads(args.overrides))
    frames = list_frames(args.frames)

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

    n_kpt = pose_model.cfg.data_cfg["num_output_channels"]  # 133 for wholebody
    xy = np.zeros((len(frames), n_kpt, 2), dtype=np.float32)
    conf = np.zeros((len(frames), n_kpt), dtype=np.float32)

    for i, fp in enumerate(frames):
        det = inference_detector(det_model, fp)
        person_results = process_mmdet_results(det, cat_id=cfg_d["det_cat_id"])

        pose_results, _ = inference_top_down_pose_model(
            pose_model, fp, person_results,
            bbox_thr=cfg_d["bbox_thr"], format="xyxy",
            dataset=dataset, dataset_info=dataset_info,
            return_heatmap=False, outputs=None,
        )
        if not pose_results:
            continue
        # single subject per view: keep the most confident body detection
        best = max(pose_results, key=lambda r: float(np.mean(r["keypoints"][:17, 2])))
        kp = best["keypoints"]  # (n_kpt, 3): x, y, score
        xy[i] = kp[:, :2]
        conf[i] = kp[:, 2]

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    np.savez(
        args.out,
        xy=xy,
        confidence=conf,
        source_frame_indices=np.arange(len(frames), dtype=np.int64),
        layout=np.array("coco_wholebody"),
    )
    print(f"wrote {args.out}  ({len(frames)} frames, {n_kpt} keypoints)")


if __name__ == "__main__":
    main()
