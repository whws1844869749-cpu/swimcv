import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 将下载的模型继续保存在D盘项目中
os.environ["TORCH_HOME"] = str(ROOT / "models" / "torch_cache")

import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer


RAW_DIR = ROOT / "output" / "validation" / "raw_frames"
CROP_DIR = ROOT / "output" / "validation" / "rotated_crops"
RESULT_DIR = ROOT / "output" / "validation" / "rtmpose_results"
CONTACT_SHEET = (
    ROOT / "output" / "validation" / "rtmpose_contact_sheet.jpg"
)

TILE_WIDTH = 180
TILE_HEIGHT = 600
COLUMNS = 5
ROWS = 2


def prepare_rotated_crops():
    """统一裁剪水面区域，并将横向游泳者旋转为直立方向。"""

    CROP_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(RAW_DIR.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(f"没有找到验证帧：{RAW_DIR}")

    saved_paths = []

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"无法读取：{image_path}")
            continue

        height, width = frame.shape[:2]

        # 保留整个水平方向，只裁掉无关的看台和池底
        y1 = max(0, 320)
        y2 = min(height, 900)
        crop = frame[y1:y2, 0:width]

        # 逆时针旋转，使手臂方向朝上、腿部方向朝下
        rotated = cv2.rotate(
            crop,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

        output_path = CROP_DIR / image_path.name
        cv2.imwrite(str(output_path), rotated)
        saved_paths.append(output_path)

        print(f"已准备：{output_path.name}")

    return saved_paths


def run_rtmpose():
    """批量运行RTMPose。"""

    print("\n正在加载 RTMPose-m……")

    inferencer = MMPoseInferencer(
        pose2d="human",
        det_model="whole_image",
        device="cpu",
    )

    result_generator = inferencer(
        str(CROP_DIR),
        out_dir=str(RESULT_DIR),
        show=False,
        kpt_thr=0.1,
        radius=5,
        thickness=3,
    )

    completed = 0

    for _ in result_generator:
        completed += 1
        print(f"已完成：{completed}")

    print(f"\nRTMPose 共处理 {completed} 张图片")


def create_contact_sheet():
    """把10张姿态结果合并成一张总览图。"""

    visualization_dir = RESULT_DIR / "visualizations"
    image_paths = sorted(visualization_dir.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(
            f"没有找到姿态可视化结果：{visualization_dir}"
        )

    canvas = np.zeros(
        (ROWS * TILE_HEIGHT, COLUMNS * TILE_WIDTH, 3),
        dtype=np.uint8,
    )

    for index, image_path in enumerate(image_paths[:10]):
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        tile = cv2.resize(
            image,
            (TILE_WIDTH, TILE_HEIGHT),
        )

        label = f"{index + 1:02d}"

        cv2.rectangle(
            tile,
            (0, 0),
            (TILE_WIDTH, 30),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            tile,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        row = index // COLUMNS
        column = index % COLUMNS

        y1 = row * TILE_HEIGHT
        y2 = y1 + TILE_HEIGHT
        x1 = column * TILE_WIDTH
        x2 = x1 + TILE_WIDTH

        canvas[y1:y2, x1:x2] = tile

    cv2.imwrite(str(CONTACT_SHEET), canvas)
    print(f"RTMPose 总览图：{CONTACT_SHEET}")


def main():
    prepared = prepare_rotated_crops()

    print(f"\n共生成 {len(prepared)} 张旋转裁剪图")

    run_rtmpose()
    create_contact_sheet()


if __name__ == "__main__":
    main()