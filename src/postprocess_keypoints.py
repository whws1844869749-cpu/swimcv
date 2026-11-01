"""将 RTMPose 的旋转裁剪坐标还原到原始游泳视频，并生成可用的 Demo 输出。"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_JSON = (
    ROOT
    / "output"
    / "validation"
    / "first_half_test"
    / "rtmpose_results"
    / "predictions"
    / "rotated_first_half_input.json"
)
SOURCE_VIDEO = ROOT / "input" / "VID_20260827_005507.mp4"
OUTPUT_DIR = ROOT / "output" / "postprocess"

RAW_CSV = OUTPUT_DIR / "keypoints_raw.csv"
FILTERED_CSV = OUTPUT_DIR / "keypoints_filtered.csv"
RAW_VIDEO = OUTPUT_DIR / "original_pose_raw.mp4"
SMOOTHED_VIDEO = OUTPUT_DIR / "original_pose_smoothed.mp4"

# These values match src/validate_rtmpose_clip.py.
CROP_Y1 = 320
CONFIDENCE_THRESHOLD = 0.30
MAX_INTERPOLATION_GAP = 3
MEDIAN_WINDOW = 5

JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# COCO 17-keypoint skeleton.
SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def rotated_to_original(x_rotated, y_rotated, source_width, crop_y1):
    """Map one point from a counter-clockwise rotated crop to original coordinates."""

    x_original = float(source_width - 1 - y_rotated)
    y_original = float(crop_y1 + x_rotated)
    return x_original, y_original


def interpolate_short_gaps(values, max_gap):
    """Linearly fill only NaN gaps surrounded by data and no longer than max_gap."""

    filled = np.asarray(values, dtype=float).copy()
    interpolated = np.zeros(len(filled), dtype=bool)

    index = 0
    while index < len(filled):
        if np.isfinite(filled[index]):
            index += 1
            continue

        gap_start = index
        while index < len(filled) and not np.isfinite(filled[index]):
            index += 1
        gap_end = index

        gap_length = gap_end - gap_start
        has_left = gap_start > 0 and np.isfinite(filled[gap_start - 1])
        has_right = gap_end < len(filled) and np.isfinite(filled[gap_end])

        if has_left and has_right and gap_length <= max_gap:
            left_value = filled[gap_start - 1]
            right_value = filled[gap_end]
            filled[gap_start:gap_end] = np.linspace(
                left_value,
                right_value,
                gap_length + 2,
            )[1:-1]
            interpolated[gap_start:gap_end] = True

    return filled, interpolated


def median_smooth(values, window):
    """Apply a NaN-aware median filter without inventing values for long missing gaps."""

    values = np.asarray(values, dtype=float)
    result = values.copy()
    radius = window // 2

    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue

        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        neighborhood = values[start:end]
        finite_values = neighborhood[np.isfinite(neighborhood)]

        if finite_values.size:
            result[index] = float(np.median(finite_values))

    return result


def load_keypoints():
    """Read the JSON produced by MMPose and keep one pose per frame."""

    if not PREDICTIONS_JSON.exists():
        raise FileNotFoundError(f"找不到 RTMPose 预测文件：{PREDICTIONS_JSON}")

    with PREDICTIONS_JSON.open("r", encoding="utf-8") as file:
        predictions = json.load(file)

    frame_ids = []
    keypoints = []
    scores = []

    for frame in predictions:
        frame_ids.append(int(frame["frame_id"]))
        instances = frame.get("instances", [])

        if instances:
            instance = instances[0]
            keypoints.append(instance["keypoints"])
            scores.append(instance["keypoint_scores"])
        else:
            keypoints.append(np.full((len(JOINT_NAMES), 2), np.nan).tolist())
            scores.append(np.zeros(len(JOINT_NAMES)).tolist())

    return (
        np.asarray(frame_ids, dtype=int),
        np.asarray(keypoints, dtype=float),
        np.asarray(scores, dtype=float),
    )


def restore_original_coordinates(rotated_keypoints, source_width):
    """Restore all frame/joint coordinates from rotated-crop space to original-video space."""

    restored = np.empty_like(rotated_keypoints, dtype=float)
    restored[..., 0] = source_width - 1 - rotated_keypoints[..., 1]
    restored[..., 1] = CROP_Y1 + rotated_keypoints[..., 0]
    return restored


def prepare_filtered_coordinates(original_keypoints, scores):
    """Mask low-confidence points, interpolate short gaps, then apply a median filter."""

    source_valid = scores >= CONFIDENCE_THRESHOLD
    interpolated_coordinates = np.where(
        source_valid[..., None],
        original_keypoints,
        np.nan,
    )
    interpolated_flags = np.zeros(source_valid.shape, dtype=bool)
    smoothed_coordinates = np.full_like(interpolated_coordinates, np.nan)

    for joint_index in range(len(JOINT_NAMES)):
        x_values, x_interpolated = interpolate_short_gaps(
            interpolated_coordinates[:, joint_index, 0],
            MAX_INTERPOLATION_GAP,
        )
        y_values, y_interpolated = interpolate_short_gaps(
            interpolated_coordinates[:, joint_index, 1],
            MAX_INTERPOLATION_GAP,
        )

        interpolated_coordinates[:, joint_index, 0] = x_values
        interpolated_coordinates[:, joint_index, 1] = y_values
        interpolated_flags[:, joint_index] = x_interpolated | y_interpolated
        smoothed_coordinates[:, joint_index, 0] = median_smooth(
            x_values,
            MEDIAN_WINDOW,
        )
        smoothed_coordinates[:, joint_index, 1] = median_smooth(
            y_values,
            MEDIAN_WINDOW,
        )

    return (
        source_valid,
        interpolated_flags,
        interpolated_coordinates,
        smoothed_coordinates,
    )


def csv_value(value):
    return "" if not np.isfinite(value) else f"{float(value):.3f}"


def write_csv_files(
    frame_ids,
    fps,
    original_keypoints,
    scores,
    source_valid,
    interpolated_flags,
    interpolated_coordinates,
    smoothed_coordinates,
):
    """Write long-format raw and filtered CSV files, one row per frame and joint."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with RAW_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame_id",
                "timestamp_s",
                "joint_id",
                "joint_name",
                "x_raw",
                "y_raw",
                "score",
                "valid_raw",
            ],
        )
        writer.writeheader()

        for frame_index, frame_id in enumerate(frame_ids):
            for joint_id, joint_name in enumerate(JOINT_NAMES):
                writer.writerow(
                    {
                        "frame_id": int(frame_id),
                        "timestamp_s": f"{frame_id / fps:.4f}",
                        "joint_id": joint_id,
                        "joint_name": joint_name,
                        "x_raw": csv_value(original_keypoints[frame_index, joint_id, 0]),
                        "y_raw": csv_value(original_keypoints[frame_index, joint_id, 1]),
                        "score": f"{scores[frame_index, joint_id]:.4f}",
                        "valid_raw": bool(source_valid[frame_index, joint_id]),
                    }
                )

    with FILTERED_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame_id",
                "timestamp_s",
                "joint_id",
                "joint_name",
                "score",
                "valid_raw",
                "interpolated",
                "x_interpolated",
                "y_interpolated",
                "x_filtered",
                "y_filtered",
            ],
        )
        writer.writeheader()

        for frame_index, frame_id in enumerate(frame_ids):
            for joint_id, joint_name in enumerate(JOINT_NAMES):
                writer.writerow(
                    {
                        "frame_id": int(frame_id),
                        "timestamp_s": f"{frame_id / fps:.4f}",
                        "joint_id": joint_id,
                        "joint_name": joint_name,
                        "score": f"{scores[frame_index, joint_id]:.4f}",
                        "valid_raw": bool(source_valid[frame_index, joint_id]),
                        "interpolated": bool(interpolated_flags[frame_index, joint_id]),
                        "x_interpolated": csv_value(
                            interpolated_coordinates[frame_index, joint_id, 0]
                        ),
                        "y_interpolated": csv_value(
                            interpolated_coordinates[frame_index, joint_id, 1]
                        ),
                        "x_filtered": csv_value(
                            smoothed_coordinates[frame_index, joint_id, 0]
                        ),
                        "y_filtered": csv_value(
                            smoothed_coordinates[frame_index, joint_id, 1]
                        ),
                    }
                )


def point_is_drawable(point, visible, frame_width, frame_height):
    if not visible or not np.all(np.isfinite(point)):
        return False

    x, y = point
    return 0 <= x < frame_width and 0 <= y < frame_height


def draw_pose(frame, coordinates, visible, color, interpolated=None):
    """Draw a skeleton on one original, horizontal video frame."""

    height, width = frame.shape[:2]

    for start_joint, end_joint in SKELETON:
        if (
            point_is_drawable(coordinates[start_joint], visible[start_joint], width, height)
            and point_is_drawable(coordinates[end_joint], visible[end_joint], width, height)
        ):
            start = tuple(np.rint(coordinates[start_joint]).astype(int))
            end = tuple(np.rint(coordinates[end_joint]).astype(int))
            cv2.line(frame, start, end, color, 3, cv2.LINE_AA)

    for joint_id, point in enumerate(coordinates):
        if not point_is_drawable(point, visible[joint_id], width, height):
            continue

        point_color = color
        if interpolated is not None and interpolated[joint_id]:
            point_color = (0, 165, 255)

        center = tuple(np.rint(point).astype(int))
        cv2.circle(frame, center, 5, point_color, -1, cv2.LINE_AA)


def write_pose_videos(
    frame_ids,
    original_keypoints,
    source_valid,
    smoothed_coordinates,
):
    """Render raw and smoothed skeletons over matching original-video frames."""

    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(f"找不到原始视频：{SOURCE_VIDEO}")

    cap = cv2.VideoCapture(str(SOURCE_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开原始视频：{SOURCE_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    codec = cv2.VideoWriter_fourcc(*"mp4v")
    raw_writer = cv2.VideoWriter(str(RAW_VIDEO), codec, fps, (width, height))
    smoothed_writer = cv2.VideoWriter(str(SMOOTHED_VIDEO), codec, fps, (width, height))

    if not raw_writer.isOpened() or not smoothed_writer.isOpened():
        cap.release()
        raw_writer.release()
        smoothed_writer.release()
        raise RuntimeError("无法创建输出视频")

    for frame_index, frame_id in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        success, frame = cap.read()

        if not success:
            print(f"无法读取原视频第 {frame_id} 帧，停止输出视频。")
            break

        raw_frame = frame.copy()
        smoothed_frame = frame.copy()

        draw_pose(
            raw_frame,
            original_keypoints[frame_index],
            source_valid[frame_index],
            color=(0, 0, 255),
        )
        draw_pose(
            smoothed_frame,
            smoothed_coordinates[frame_index],
            np.all(np.isfinite(smoothed_coordinates[frame_index]), axis=1),
            color=(0, 255, 0),
        )

        cv2.putText(
            raw_frame,
            "RAW: score >= 0.30",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            smoothed_frame,
            "FILTERED + SMOOTHED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        raw_writer.write(raw_frame)
        smoothed_writer.write(smoothed_frame)

    cap.release()
    raw_writer.release()
    smoothed_writer.release()

    return fps


def main():
    frame_ids, rotated_keypoints, scores = load_keypoints()

    cap = cv2.VideoCapture(str(SOURCE_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开原始视频：{SOURCE_VIDEO}")
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    original_keypoints = restore_original_coordinates(rotated_keypoints, source_width)
    (
        source_valid,
        interpolated_flags,
        interpolated_coordinates,
        smoothed_coordinates,
    ) = prepare_filtered_coordinates(original_keypoints, scores)

    write_csv_files(
        frame_ids,
        fps,
        original_keypoints,
        scores,
        source_valid,
        interpolated_flags,
        interpolated_coordinates,
        smoothed_coordinates,
    )
    write_pose_videos(
        frame_ids,
        original_keypoints,
        source_valid,
        smoothed_coordinates,
    )

    print("\n处理完成：")
    print(f"原始坐标 CSV：{RAW_CSV}")
    print(f"过滤坐标 CSV：{FILTERED_CSV}")
    print(f"原始骨架视频：{RAW_VIDEO}")
    print(f"平滑骨架视频：{SMOOTHED_VIDEO}")


if __name__ == "__main__":
    main()
