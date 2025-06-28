from ultralytics import YOLO, MOTRTrack, RTDETR
import os
import time




if __name__ == '__main__':


    model = MOTRTrack("yolo_track.yaml")  # 7443199 7447807 7443199

    # 使用模型
    model.train(data="dancetracker.yaml", epochs=1, batch=1)  # 训练模型

