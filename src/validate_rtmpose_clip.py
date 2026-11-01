import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["TORCH_HOME"] = str(ROOT / "models" / "torch_cache")

import cv2
from mmpose.apis import MMPoseInferencer


SOURCE_VIDEO = (
    ROOT / "input" / "VID_20260827_005507.mp4"
)

OUTPUT_ROOT = (
    ROOT / "output" / "validation" / "first_half_test"
)

PREPARED_VIDEO = OUTPUT_ROOT / "rotated_first_half_input.mp4"
RESULT_DIR = OUTPUT_ROOT / "rtmpose_results"

START_SECOND = 0.0
DURATION_SECONDS = 16.0

CROP_Y1 = 320
CROP_Y2 = 900


def create_rotated_clip():
    """截取5秒视频，裁剪泳池区域并旋转。"""

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

    start_frame = int(START_SECOND * fps)
    frames_to_process = int(DURATION_SECONDS * fps)
    end_frame = min(start_frame + frames_to_process, frame_count)

    y1 = max(0, CROP_Y1)
    y2 = min(source_height, CROP_Y2)

    # 原图裁剪后尺寸：
    # 宽 = source_width
    # 高 = y2 - y1
    #
    # 旋转后尺寸对调
    output_width = y2 - y1
    output_height = source_width

    writer = cv2.VideoWriter(
        str(PREPARED_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("无法创建测试视频")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved_count = 0

    for frame_number in range(start_frame, end_frame):
        success, frame = cap.read()

        if not success:
            break

        crop = frame[y1:y2, 0:source_width]

        rotated = cv2.rotate(
            crop,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

        writer.write(rotated)
        saved_count += 1

    cap.release()
    writer.release()

    print(f"原视频帧率：{fps:.2f}")
    print(f"已生成测试视频：{PREPARED_VIDEO}")
    print(f"测试帧数：{saved_count}")

    return fps


def run_rtmpose(fps):
    """对5秒视频逐帧执行RTMPose。"""

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


def main():
    fps = create_rotated_clip()
    run_rtmpose(fps)


if __name__ == "__main__":
    main()