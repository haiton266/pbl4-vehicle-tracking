from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import numpy as np
from pathlib import Path
from boxmot import BYTETracker
import torch
import time
import csv
from collections import defaultdict

name_video = 'hota1'
type_video = 'MOV'
name_model = 'best_10'
name_tracking = 'BYTETracker'
conf_thres = 0.5


tracker = BYTETracker()

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=f'{name_model}.pt',
    confidence_threshold=conf_thres,
    device="cpu",  # Sử dụng GPU cuda:0
)
track_history = defaultdict(lambda: [])

vid = cv2.VideoCapture(f'video/{name_video}.{type_video}')
total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
widthFrame = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
heightFrame = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (widthFrame, heightFrame)

out = cv2.VideoWriter('output.avi',
                      cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

track_colors = {}

points = np.array([[0, 976], [567, 136], [1630, 140], [1919, 426], [1920, 1080], [
                  0, 1080]], np.int32)

Path(name_tracking).mkdir(parents=True, exist_ok=True)
with open(f'{name_tracking}/{name_video}-{name_model}.txt', 'w', newline='') as file:
    writer = csv.writer(file)

    thickness = 2
    fontscale = 1

    while True:
        ret, im = vid.read()

        if not ret:
            break

        mask = np.zeros(im.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        roi_img = cv2.bitwise_and(im, im, mask=mask)
        cv2.polylines(im, [points], True, (0, 255, 0), 3)

        frame_number = vid.get(cv2.CAP_PROP_POS_FRAMES)

        start_time = time.time()

        result = get_sliced_prediction(
            roi_img,
            detection_model,
            slice_height=4096,
            slice_width=4096,
            overlap_height_ratio=0.001,
            overlap_width_ratio=0.001
        )

        # Tạo mảng dets và điều chỉnh tọa độ
        dets = np.empty((0, 6), dtype=np.float32)
        for object_prediction in result.object_prediction_list:
            bbox = np.array(
                object_prediction.bbox.to_xyxy(), dtype=np.float32)

            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            det = np.array([[bbox[0], bbox[1], bbox[2], bbox[3],
                             object_prediction.score.value, object_prediction.category.id]])
            dets = np.vstack([dets, det])

        # Cập nhật bộ theo dõi
        tracks = tracker.update(dets, roi_img)

        if tracks.shape[0] != 0:
            for track in tracks:
                x1, y1, x2, y2, track_id, confidence, class_id, _ = track

                # Tạo màu ngẫu nhiên cho mỗi ID mới
                if track_id not in track_colors:
                    track_colors[track_id] = (np.random.randint(
                        0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

                # Lấy màu từ từ điển
                color = track_colors[track_id]

                writer.writerow([frame_number, track_id, x1,
                                y1, x2, y2, 1, -1, -1, -1])
                im = cv2.rectangle(im, (int(x1), int(y1)),
                                   (int(x2), int(y2)), color, thickness)
                cv2.putText(im, f'{track_id}', (int(x1), int(
                    y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, fontscale, color, thickness)

                # Cập nhật lịch sử theo dõi
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                track_history[track_id].append((center_x, center_y))
                # Giữ lại lịch sử 50 khung hình gần nhất
                if len(track_history[track_id]) > 50:
                    track_history[track_id].pop(0)

                # Vẽ lịch sử theo dõi
                points2 = np.array(
                    track_history[track_id], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(im, [points2], isClosed=False,
                              color=track_colors[track_id], thickness=2)

        out.write(im)

        end_time = time.time()
        frame_processing_time = end_time - start_time
        progress = (frame_number / total_frames) * 100
        print(
            f"Processing frame {int(frame_number)} / {int(total_frames)} ({progress:.2f}%) - in {frame_processing_time:.2f} s")

    vid.release()
    out.release()
    # torch.cuda.synchronize()

# Tạo file seqinfo.ini
seqinfo_content = f"""
[Sequence]
name=video1
imDir=img1
frameRate=30
seqLength={int(total_frames)}
imWidth={size[0]}
imHeight={size[1]}
imExt=.jpg
"""
with open(f'{name_tracking}/{name_video}-{name_model}.ini', 'w') as f:
    f.write(seqinfo_content)
