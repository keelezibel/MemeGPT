import os
import cv2
import ffmpeg
import tempfile
import shutil
from cv2 import exp
from cv2 import Mat_DEPTH_MASK
import numpy as np

UPPER_GREEN = np.array([80, 255, 255])
LOWER_GREEN = np.array([40, 230, 200])


def process_vid(video, image, output_vid_path):
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    image = cv2.resize(image, (frame_width, frame_height))

    output_frames = []
    while video.isOpened():
        try:
            ret, frame = video.read()
            if ret:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
                res = cv2.bitwise_and(frame, frame, mask=mask)

                f = frame - res
                output_f = np.where(f == 0, image, f)

                output_frames.append(output_f)
            else:
                break
        except Exception as e:
            print(str(e))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_output_file = os.path.join(tempfile.gettempdir(), "tmp.mp4")
    out = cv2.VideoWriter(
        tmp_output_file, fourcc, video_fps, (frame_width, frame_height)
    )
    for i in range(len(output_frames)):
        try:
            out.write(output_frames[i])
        except Exception as e:
            print(str(e))

    video.release()
    out.release()

    return tmp_output_file


def add_original_audio(original_vid_path, tmp_output_file, output_vid_path):
    original_vid = ffmpeg.input(original_vid_path)
    processed_vid = ffmpeg.input(tmp_output_file)
    out = ffmpeg.output(processed_vid.video, original_vid.audio, output_vid_path)
    out.run()

    try:
        if os.path.isfile(tmp_output_file):
            os.remove(tmp_output_file)
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    original_vid_path = "/assets/videos/Beetlejuice.mp4"
    output_vid_path = "/app/outputs/output.mp4"
    image = cv2.imread("/assets/images/test.jpg")
    video = cv2.VideoCapture(original_vid_path)

    tmp_output_file = process_vid(video, image, output_vid_path)
    add_original_audio(original_vid_path, tmp_output_file, output_vid_path)
