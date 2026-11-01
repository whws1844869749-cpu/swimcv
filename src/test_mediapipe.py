from pathlib import Path

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[1]
INPUT_IMAGE = ROOT / "input" / "test_frame.jpg"
MODEL_PATH = ROOT / "models" / "pose_landmarker_heavy.task"
OUTPUT_IMAGE = ROOT / "output" / "mediapipe_test_frame.jpg"

# MediaPipe 33 点中的躯干与四肢连接关系
CONNECTIONS = [
    (11, 12),  # 左右肩
    (11, 13), (13, 15),  # 左肩-左肘-左腕
    (12, 14), (14, 16),  # 右肩-右肘-右腕
    (11, 23), (12, 24), (23, 24),  # 躯干
    (23, 25), (25, 27),  # 左髋-左膝-左踝
    (24, 26), (26, 28),  # 右髋-右膝-右踝
]

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.2,
    min_pose_presence_confidence=0.2,
)

image = mp.Image.create_from_file(str(INPUT_IMAGE))

with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(image)

frame = cv2.imread(str(INPUT_IMAGE))
if frame is None:
    raise FileNotFoundError(f"无法读取：{INPUT_IMAGE}")

if not result.pose_landmarks:
    print("MediaPipe 未检测到人体姿态。")
else:
    landmarks = result.pose_landmarks[0]
    height, width = frame.shape[:2]

    for start, end in CONNECTIONS:
        a, b = landmarks[start], landmarks[end]
        if a.visibility > 0.3 and b.visibility > 0.3:
            p1 = (int(a.x * width), int(a.y * height))
            p2 = (int(b.x * width), int(b.y * height))
            cv2.line(frame, p1, p2, (0, 255, 0), 3)

    for index, point in enumerate(landmarks):
        if point.visibility > 0.3:
            x, y = int(point.x * width), int(point.y * height)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                frame, str(index), (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )

    OUTPUT_IMAGE.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_IMAGE), frame)
    print(f"检测到 1 个姿态，结果已保存：{OUTPUT_IMAGE}")