import cv2
import numpy as np
import os
from pathlib import Path

def is_informative_frame(frame, threshold=10, min_nonblack_ratio=0.05):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    non_black_pixels = np.sum(gray > threshold)
    total_pixels = gray.size
    return (non_black_pixels / total_pixels) > min_nonblack_ratio

def extract_frames_from_video(video_path, output_dir, max_frames_per_video, 
                              brightness_threshold=10, min_nonblack_ratio=0.05):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ошибка открытия: {video_path}")
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // (max_frames_per_video * 2))
    saved_count = 0
    frame_idx = 0
    while saved_count < max_frames_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        if is_informative_frame(frame, brightness_threshold, min_nonblack_ratio):
            output_path = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue
    cap.release()
    print(f"{os.path.basename(video_path)}: saved {saved_count} frames from {total_frames}")
    return saved_count

def process_videos(video_list, output_root, target_total_frames=4000):
    os.makedirs(output_root, exist_ok=True)
    frames_per_video = target_total_frames // len(video_list)
    total_saved = 0
    for video_path, solution_name in video_list:
        video_output_dir = os.path.join(output_root, solution_name)
        os.makedirs(video_output_dir, exist_ok=True)
        saved = extract_frames_from_video(
            video_path=video_path,
            output_dir=video_output_dir,
            max_frames_per_video=frames_per_video,
            brightness_threshold=10,
            min_nonblack_ratio=0.05
        )
        total_saved += saved
    print(f"\nвсего успешных файлов : {total_saved} (target: ~{target_total_frames})")
    return total_saved

if __name__ == "__main__":
    videos = [
        ("videos/jp2c00948_si_001.mp4", "solution_0pct"),
        ("videos/jp2c00948_si_002.mp4", "solution_5pct"),
        ("videos/jp2c00948_si_003.mp4", "solution_12.5pct"),
        ("videos/jp2c00948_si_004.mp4", "solution_25pct"),
        ("videos/jp2c00948_si_005.mp4", "solution_50pct"),
        ("videos/jp2c00948_si_006.mp4", "solution_75pct"),
        ("videos/jp2c00948_si_007.mp4", "solution_100pct"),
    ]
    process_videos(
        video_list=videos,
        output_root="output_frames",
        target_total_frames=4000
    )