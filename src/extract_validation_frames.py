from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

VIDEOS = [
    ROOT / "input" / "VID_20260827_005655.mp4",
    ROOT / "input" / "VID_20260827_005507.mp4",
]

OUTPUT_DIR = ROOT / "output" / "validation"
RAW_DIR = OUTPUT_DIR / "raw_frames"
CONTACT_SHEET = OUTPUT_DIR / "contact_sheet.jpg"

# 分别选取视频15%、32.5%、50%、67.5%、85%位置的画面
SAMPLE_RATIOS = [0.15, 0.325, 0.50, 0.675, 0.85]

TILE_WIDTH = 480
TILE_HEIGHT = 270


def extract_frames(video_path):
    if not video_path.exists():
        raise FileNotFoundError(f"找不到视频：{video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(
        f"\n视频：{video_path.name}\n"
        f"帧率：{fps:.2f} FPS\n"
        f"总帧数：{frame_count}\n"
        f"时长：{duration:.2f} 秒"
    )

    saved_frames = []

    for index, ratio in enumerate(SAMPLE_RATIOS, start=1):
        frame_number = min(int(frame_count * ratio), frame_count - 1)
        time_seconds = frame_number / fps

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if not success:
            print(f"读取第 {frame_number} 帧失败")
            continue

        output_name = (
            f"{video_path.stem}_"
            f"{index:02d}_"
            f"{time_seconds:.2f}s.jpg"
        )
        output_path = RAW_DIR / output_name

        cv2.imwrite(str(output_path), frame)
        saved_frames.append((output_path, time_seconds))

        print(f"已保存：{output_path.name}")

    cap.release()
    return saved_frames


def create_contact_sheet(frame_records):
    rows = 2
    columns = 5

    canvas = np.zeros(
        (rows * TILE_HEIGHT, columns * TILE_WIDTH, 3),
        dtype=np.uint8,
    )

    for index, (frame_path, time_seconds) in enumerate(frame_records):
        image = cv2.imread(str(frame_path))

        if image is None:
            continue

        tile = cv2.resize(image, (TILE_WIDTH, TILE_HEIGHT))

        label = f"{frame_path.stem} | {time_seconds:.2f}s"

        cv2.rectangle(
            tile,
            (0, 0),
            (TILE_WIDTH, 28),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            tile,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        row = index // columns
        column = index % columns

        y1 = row * TILE_HEIGHT
        y2 = y1 + TILE_HEIGHT
        x1 = column * TILE_WIDTH
        x2 = x1 + TILE_WIDTH

        canvas[y1:y2, x1:x2] = tile

    cv2.imwrite(str(CONTACT_SHEET), canvas)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = []

    for video_path in VIDEOS:
        all_frames.extend(extract_frames(video_path))

    create_contact_sheet(all_frames)

    print(f"\n共提取 {len(all_frames)} 帧")
    print(f"验证帧目录：{RAW_DIR}")
    print(f"总览图：{CONTACT_SHEET}")


if __name__ == "__main__":
    main()