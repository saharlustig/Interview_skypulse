import os

import cv2
import numpy as np
import yaml

from frame_stabilizer import FrameStabilizer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class FrameLoader:
    def __init__(self, video_path):
        self.camera = cv2.VideoCapture(video_path)
        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.camera.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        return self.camera.read()


def display_frames(original_frame, aligned_frame, output_video, save_vid=True):
    cv2.putText(original_frame, f"original video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)
    orig = original_frame[::3, ::3]

    cv2.putText(aligned_frame, f"aligned video", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 6)
    aligned = aligned_frame[::3, ::3]

    display_frame = np.concatenate((orig, aligned), axis=0)
    cv2.imshow('frame', display_frame)
    cv2.waitKey(1)
    if save_vid: output_video.write(np.concatenate((original_frame, aligned_frame), axis=0))


def main():
    config = yaml.safe_load(open(os.path.join(ROOT_DIR, "config.yml")))
    frame_loader = FrameLoader(os.path.join(ROOT_DIR, config['video_dir_relative'], config['video_name']))
    frame_stabilizer = FrameStabilizer(frame_loader.width, frame_loader.height, buffer_size=int(frame_loader.fps * 3))

    out = cv2.VideoWriter(os.path.join(ROOT_DIR, config['video_dir_relative'], config['output_video_name']), cv2.VideoWriter_fourcc(*"mp4v"), frame_loader.fps,
                          (frame_loader.width, frame_loader.height * 2))

    while True:
        ret, frame = frame_loader.get_frame()
        if not ret: break
        stabilized_frame = frame_stabilizer.stabilize(frame)
        display_frames(frame, stabilized_frame, out)


if __name__ == '__main__':
    main()
