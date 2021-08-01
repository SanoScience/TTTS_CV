import os
import glob
from tqdm import tqdm
import cv2
import pandas as pd

data_path = "../../own_data/"
output_path = os.path.join(data_path, 'Videos_Frames/')

os.makedirs(output_path, exist_ok=True)
video_files = glob.glob(f"{data_path}*.mp4", recursive=True)


for file in tqdm(video_files, total=len(video_files)):
    filename, _ = os.path.basename(file).split('.')
    file_dir = os.path.join(output_path, filename)
    os.makedirs(file_dir, exist_ok=True)
    video_capture = cv2.VideoCapture(file)
    frame_num = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            cv2.imwrite(os.path.join(file_dir, f'{filename}_{frame_num}.png'), frame)
            frame_num += 1
        else:
            break
