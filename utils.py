# utils.py

import cv2
import os

def batch_extract_frames(video_dir, output_dir, label_prefix, frame_rate=5):
    count = 0
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(video_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_dir, filename)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"[SKIP] Cannot open: {filename}")
                continue

            frame_id = 0
            saved = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id % frame_rate == 0:
                    file_name = f"{label_prefix}_{count}_{saved}.jpg"
                    cv2.imwrite(os.path.join(output_dir, file_name), frame)
                    saved += 1
                frame_id += 1
            cap.release()
            print(f"[EXTRACTED] {saved} frames from: {filename}")
            count += 1
