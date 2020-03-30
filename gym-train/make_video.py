import cv2
import os

RUN_NUM = 13


def make_video():
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(f'Cart-{RUN_NUM}.mp4', fourcc, 60.0, (1500, 800))

    for i in range(0, 50000, 10):
        img_path = f"qtables_{RUN_NUM}_charts/{i}.png"
        print(img_path)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


make_video()



