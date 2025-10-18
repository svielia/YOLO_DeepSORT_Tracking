#  YOLO + DeepSORT 实时目标跟踪与统计系统

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange.svg)]()
[![DeepSORT](https://img.shields.io/badge/Tracker-DeepSORT-yellow.svg)]()

>  一款基于 **YOLOv8 + DeepSORT** 的多目标检测与跟踪系统，支持统计每个目标出现的帧数，并自动生成视频、统计表格与柱状图

##  项目简介

本项目通过 **YOLOv8** 检测目标，并结合 **DeepSORT** 实现视频中多目标的持续跟踪，最终输出：
- 目标检测与跟踪视频；
- 每个目标 ID 的出现帧数统计表；
- 对应的可视化柱状图。

适用于智能监控、行人计数、车辆跟踪、运动行为分析等场景。

##  功能特性

-  实时检测与跟踪多个目标  
-  自动统计每个目标出现帧数  
-  生成可视化柱状图和 CSV 报表  
-  模块化代码结构，易扩展  
-  支持自定义 YOLO 模型权重与阈值

  环境依赖

请确保你的 Python 环境中安装以下依赖：

```bash
pip install ultralytics opencv-python deep-sort-realtime pandas matplotlib
```

项目结构

yolo_deepsort_tracking_with_stats.py     # 主程序

test_video.mp4                           # 输入视频（自行放置）

outputs/                                 # 输出目录（程序自动生成）

├── output_video.mp4                     # 带跟踪框的视频

├── ID_counts.csv                        # 各目标出现帧数统计表

└── ID_counts.png                        # 柱状图结果

技术说明

YOLOv8 (Ultralytics)：实时目标检测算法。

DeepSORT：基于卡尔曼滤波与匈牙利算法的多目标跟踪器。

Pandas + Matplotlib：用于数据统计与可视化。

OpenCV：负责视频帧读取、绘制与输出。

许可证

本项目基于 MIT License
 授权。
© 2025 Moris Bridges. All rights reserved.
