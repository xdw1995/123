import os
import cv2
for video in sorted(os.listdir('/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/videos')):

    cap = cv2.VideoCapture('/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/videos/'+video)
    if not os.path.exists('/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/datas/'+video.replace('.avi','')):

        os.makedirs('/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/datas/'+video.replace('.avi',''))

    index = 0
    while (cap.isOpened()):
        index = index+1
        ret, frame = cap.read()
        if ret == False:
            break
        if frame is None:
            break
        cv2.imwrite('/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/datas/'+video.replace('.avi','')+'/'+
                    '0'*(7-len(str(index)))+str(index)+'.jpg',frame)

