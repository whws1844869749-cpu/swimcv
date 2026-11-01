import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]

# 原始完整视频
SOURCE_VIDEO = ROOT / "video" / "VID_20260827_145436.mp4"

# RTMPose 推理产生的关键点 JSON
JSON_FILE = (
    ROOT
    / "output"
    / "validation"
    / "test_145436"
    / "rtmpose_results"
    / "predictions"
    / "rotated_input.json"
)

# 最终输出：完整原视频画面 + 横向骨架
OUTPUT_VIDEO = (
    ROOT
    / "output"
    / "validation"
    / "test_145436"
    / "original_full_pose.mp4"
)

# 必须与 RTMPose 测试脚本一致
CROP_Y1 = 320
CONF_THRESHOLD = 0.30

# COCO 17 点骨架连接关系
SKELETON = [
    (5, 7), (7, 9),       # 左肩-左肘-左腕
    (6, 8), (8, 10),      # 右肩-右肘-右腕
    (5, 6),               # 双肩
    (5, 11), (6, 12),     # 肩到髋
    (11, 12),             # 双髋
    (11, 13), (13, 15),   # 左髋-左膝-左踝
    (12, 14), (14, 16),   # 右髋-右膝-右踝
]


def read_instances(frame_result):
    """兼容 MMPose 导出的常见 JSON 结构。"""
    if "instances" in frame_result:
        return frame_result["instances"]

    if "instance_info" in frame_result:
        return frame_result["instance_info"]

    return []


def rotated_to_original(x_rot, y_rot, source_width):
    """
    RTMPose 输入画面由原图 y=320:900 裁剪后逆时针旋转得到。

    反变换：
    原图 x = 原图宽度 - 1 - 旋转后 y
    原图 y = 裁剪起始 y + 旋转后 x
    """
    x_original = source_width - 1 - y_rot
    y_original = CROP_Y1 + x_rot

    return int(round(x_original)), int(round(y_original))


with open(JSON_FILE, "r", encoding="utf-8") as f:
    predictions = json.load(f)

cap = cv2.VideoCapture(str(SOURCE_VIDEO))

if not cap.isOpened():
    raise FileNotFoundError(f"无法读取原始视频：{SOURCE_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

for frame_id in range(frame_count):
    ok, frame = cap.read()

    if not ok:
        break

    points = [None] * 17

    if frame_id < len(predictions):
        instances = read_instances(predictions[frame_id])

        if instances:
            person = instances[0]
            keypoints = person["keypoints"]
            scores = person["keypoint_scores"]

            for joint_id, (point, score) in enumerate(zip(keypoints, scores)):
                if score < CONF_THRESHOLD:
                    continue

                x_rot, y_rot = point
                x, y = rotated_to_original(x_rot, y_rot, width)

                # 只绘制位于原始画面范围内的点
                if 0 <= x < width and 0 <= y < height:
                    points[joint_id] = (x, y, float(score))

    # 画骨架线
    for start, end in SKELETON:
        if points[start] is not None and points[end] is not None:
            p1 = points[start][:2]
            p2 = points[end][:2]
            cv2.line(frame, p1, p2, (0, 255, 0), 3)

    # 画关键点
    for point in points:
        if point is not None:
            x, y, score = point
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    cv2.putText(
        frame,
        f"RTMPose | frame {frame_id}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    writer.write(frame)

cap.release()
writer.release()

print(f"处理完成：{OUTPUT_VIDEO}")