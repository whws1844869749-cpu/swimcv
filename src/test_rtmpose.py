import os
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]

# 将模型缓存放到 D 盘项目中
os.environ["TORCH_HOME"] = str(ROOT / "models" / "torch_cache")

import cv2
from mmpose.apis import MMPoseInferencer


INPUT_IMAGE = ROOT / "input" / "test_frame.jpg"
TEST_IMAGE = ROOT / "input" / "rtmpose_rotated_crop.jpg"
OUTPUT_DIR = ROOT / "output" / "rtmpose_test"

# 读取原图
frame = cv2.imread(str(INPUT_IMAGE))

if frame is None:
    raise FileNotFoundError(f"无法读取图片：{INPUT_IMAGE}")

# 截取游泳者所在区域
crop = frame[500:820, 380:1870]

# 将横向游泳者逆时针旋转90度，使其接近模型训练时的直立人体方向
rotated_crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

TEST_IMAGE.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(TEST_IMAGE), rotated_crop)

print(f"测试图片已生成：{TEST_IMAGE}")
print("正在加载 RTMPose，首次运行需要下载模型权重……")

# human 对应 RTMPose-m
# whole_image 表示不使用额外人体检测器
inferencer = MMPoseInferencer(
    pose2d="human",
    det_model="whole_image",
    device="cpu",
)

# 执行推理并保存骨架图及关键点数据
results = inferencer(
    str(TEST_IMAGE),
    out_dir=str(OUTPUT_DIR),
    show=False,
    kpt_thr=0.1,
    radius=5,
    thickness=3,
)

for _ in results:
    pass

print(f"RTMPose 测试完成，结果保存在：{OUTPUT_DIR}")