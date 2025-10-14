# yolo_deepsort_tracking_with_stats.py

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------- 配置参数 -----------------
YOLO_WEIGHT = "yolov8n.pt"        # YOLOv8权重
VIDEO_PATH = "test_video.mp4"     # 输入视频路径
OUTPUT_DIR = "outputs"            # 输出文件夹
CONFIDENCE_THRESHOLD = 0.3        # YOLO置信度阈值
MAX_AGE = 50                       # DeepSORT最大未匹配帧数

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- 初始化模型 -----------------
print("📦 加载 YOLO 模型...")
yolo_model = YOLO(YOLO_WEIGHT)

print("🚀 初始化 DeepSORT 跟踪器...")
tracker = DeepSort(max_age=MAX_AGE)

cap = cv2.VideoCapture(VIDEO_PATH)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video_path = os.path.join(OUTPUT_DIR, "output_video.mp4")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_idx = 0
id_count_dict = {}  # 记录每个ID出现帧数

print("🎬 开始处理视频...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO检测
    results = yolo_model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = box.cpu().numpy()
            detections.append([x1, y1, x2, y2, 1.0])  # 1.0为置信度占位

    # DeepSORT跟踪
    tracks = tracker.update_tracks(detections, frame=frame)

    for t in tracks:
        if not t.is_confirmed():
            continue
        track_id = t.track_id
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 统计每个ID出现帧数
        if track_id not in id_count_dict:
            id_count_dict[track_id] = 1
        else:
            id_count_dict[track_id] += 1

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"✅ 视频处理完成: {output_video_path}")

# ----------------- 输出柱状图 -----------------
df = pd.DataFrame(list(id_count_dict.items()), columns=["TrackID", "AppearFrames"])
df.sort_values("TrackID", inplace=True)

plt.figure(figsize=(8, 5))
plt.bar(df["TrackID"].astype(str), df["AppearFrames"], color="skyblue")
plt.xlabel("Track ID")
plt.ylabel("出现帧数")
plt.title("每个目标ID出现帧数统计")
plt.tight_layout()

output_chart_path = os.path.join(OUTPUT_DIR, "ID_counts.png")
plt.savefig(output_chart_path, dpi=150)
plt.show()
print(f"✅ 柱状图输出完成: {output_chart_path}")

# ----------------- 输出 CSV -----------------
output_csv_path = os.path.join(OUTPUT_DIR, "ID_counts.csv")
df.to_csv(output_csv_path, index=False)
print(f"✅ CSV统计完成: {output_csv_path}")

print("🎉 项目运行完成！")
