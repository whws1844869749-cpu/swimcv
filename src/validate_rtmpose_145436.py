"""Run the validated RTMPose preprocessing pipeline on VID_20260827_145436."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["TORCH_HOME"] = str(ROOT / "models" / "torch_cache")

import cv2
from mmpose.apis import MMPoseInferencer


SOURCE_VIDEO = ROOT / "video" / "VID_20260827_145436.mp4"
OUTPUT_ROOT = ROOT / "output" / "validation" / "test_145436"
PREPARED_VIDEO = OUTPUT_ROOT / "rotated_input.mp4"
RESULT_DIR = OUTPUT_ROOT / "rtmpose_results"

# Keep this consistent with the previous validated RTMPose experiment.
CROP_Y1 = 320
CROP_Y2 = 900


def create_rotated_clip() -> float:
    """Crop the pool region and rotate it for an upright-pose model."""
    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(f"找不到视频：{SOURCE_VIDEO}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(SOURCE_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{SOURCE_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    y1, y2 = max(0, CROP_Y1), min(source_height, CROP_Y2)

    writer = cv2.VideoWriter(
        str(PREPARED_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (y2 - y1, source_width),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("无法创建预处理测试视频")

    saved_count = 0
    for _ in range(frame_count):
        success, frame = cap.read()
        if not success:
            break
        crop = frame[y1:y2, 0:source_width]
        writer.write(cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE))
        saved_count += 1

    cap.release()
    writer.release()
    print(f"原视频：{SOURCE_VIDEO.name}")
    print(f"原视频帧率：{fps:.2f}")
    print(f"已生成旋转测试视频：{PREPARED_VIDEO}")
    print(f"测试帧数：{saved_count}")
    return fps


def run_rtmpose(fps: float) -> None:
    """Run RTMPose-m with whole-image detection on the prepared clip."""
    print("\n正在加载 RTMPose-m……")
    inferencer = MMPoseInferencer(
        pose2d="human",
        det_model="whole_image",
        device="cpu",
    )
    results = inferencer(
        str(PREPARED_VIDEO),
        out_dir=str(RESULT_DIR),
        show=False,
        kpt_thr=0.1,
        radius=5,
        thickness=3,
    )

    completed = 0
    report_interval = max(1, round(fps))
    for _ in results:
        completed += 1
        if completed % report_interval == 0:
            print(f"已处理约 {completed / fps:.1f} 秒")

    print(f"\n推理完成，共处理 {completed} 帧")
    print(f"结果目录：{RESULT_DIR}")


if __name__ == "__main__":
    run_rtmpose(create_rotated_clip())
