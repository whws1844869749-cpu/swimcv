from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]

INPUT_VIDEO = (
    ROOT
    / "output"
    / "validation"
    / "test_145436"
    / "rtmpose_results"
    / "visualizations"
    / "rotated_input.mp4"
)

OUTPUT_VIDEO = (
    ROOT
    / "output"
    / "validation"
    / "test_145436"
    / "normal_direction_pose.mp4"
)

cap = cv2.VideoCapture(str(INPUT_VIDEO))

if not cap.isOpened():
    raise FileNotFoundError(f"无法读取：{INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 旋转后宽高对调
input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (input_height, input_width),
)

for _ in range(frame_count):
    ok, frame = cap.read()
    if not ok:
        break

    normal_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    writer.write(normal_frame)

cap.release()
writer.release()

print(f"已输出：{OUTPUT_VIDEO}")