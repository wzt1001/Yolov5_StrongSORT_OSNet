import cv2
import subprocess
import logging

# 视频读取对象
cap = cv2.VideoCapture(".../xx.mp4") 


if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    rtmp_list = ["rtmp://172.18.0.152:1935/live/1"]
    
    for rtmp in rtmp_list:
        reader = VideoReader[]
        fps = reader.fps()
        size = reader.width(), reader.height()
        ret, frame = reader.read()
        frame = np.fromstring(frame, dtype='uint8')



# rtmp://54.223.186.221/live/stream224
# rtmp://54.223.186.221/live/stream234


# rtmp://69.230.192.222/live/dev_1597065880603336704
# rtmp://69.230.192.222/live/dev_1597066078826143744

# run command
# python track.py --classes 0 --source ./cam_list.txt --tracking-method bytetrack --reid-weights osnet_x0_25_market1501.pt --vid-stride 20
# python track.py --classes 0 --source 0 --tracking-method bytetrack --reid-weights osnet_x0_25_market1501.pt --vid-stride 20
