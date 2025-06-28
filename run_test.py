from ultralytics import YOLO, MOTRTrack, RTDETR
import os
if __name__ == '__main__':
    # model = YOLO('yolov8n.pt')
    # model = YOLO('yolov8m.yaml')
    # model = YOLO(r"E:\cv\yolo_tracking\weights\yolov8_MOT17.pt")
    # model(source=r"E:\data\MOT17\test\MOT17-03-DPM\img1")
    # model(source=r"E:\data\MOT17\test\MOT17-03-DPM\img1")

    # model = MOTRTrack("yolo_track.yaml")  # 7443199 7447807 7443199
    # model = MOTRTrack("yolov8m_track.yaml")
    # model = MOTRTrack(r"E:\cv\KITTI_detection\weights\best.pt")  # 加载预训练模型（建议用于训练）
    # model = MOTRTrack(r"E:\cv\MOT17_detection\best.pt")  # 加载预训练模型（建议用于训练）
    # folder_path = r'E:\data\dancetrack\test'
    #
    # # 使用os.listdir()获取目录下所有文件和子文件夹
    # subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    #
    #
    # for subpath in subfolders:
    #     model = MOTRTrack(r"F:\cv\tracking_pretrain\runs\track\dancetrack\weights\best.pt")  # 加载预训练模型（建议用于训练）
    #
    #     model(folder_path+'/'+subpath+r'/img1')

    # subfolders 列表中包含了指定目录下的所有子文件夹的名称

    # model = MOTRTrack(r"C:\Users\a\Desktop\train3\weights\best.pt")  # 加载预训练模型（建议用于训练）
    # model = MOTRTrack(r"F:\cv\tracking_pretrain\runs\track\MOT17\weights\best.pt")  # 加载预训练模型（建议用于训练）
    # model = MOTRTrack(r"F:\cv\tracking_pretrain\runs\track\train2\weights\best.pt")  # 加载预训练模型（建议用于训练）
    # model = RTDETR("best.pt")  # 加载预训练模型（建议用于训练）

    # 使用模型
    # model.train(data="KITTI.yaml", epochs=10, batch=1)  # 训练模型
    # model.train(data="MOT.yaml", epochs=20, batch=1)  # 训练模型

    # model(r"E:\data\KITTI_TRACKING\training\images\0000", show=False, save=True)  # 查看结果14
    # model(r"E:\data\KITTI_TRACKING\training\images\0000", show=False, save=False)  # 查看结果
    # model(r"E:\data\MOT17\images\test\MOT17-01-DPM\img1", show=False, save_txt=True)  # 查看结果
    # model(r"E:\data\MOT17\images\train\MOT17-02-DPM\img1", show=False, save_txt=True)  # 查看结果
    # MOT_test = ['03', '06', '07', '08', '12', '14']
    # for i in MOT_test:
    #     model = MOTRTrack(r"C:\Users\a\Desktop\train3\weights\best.pt")  # 加载预训练模型（建议用于训练）
    #     test_url = r"E:\data\MOT17\images\test\MOT17-" + i + r"-DPM\img1"
    #     model(test_url, show=False, save_txt=True)  # 查看结果

    # dancetrack_name = [3, 9, 11, 13, 17, 21, 22, 28, 31, 36, 38, 40, 42, 46, 48, 50, 54, 56, 59, 60, 64, 67, 70, 71, 76, 78, 84, 85, 88, 89, 91, 92, 93, 95, 100]
    # for i in dancetrack_name:
    #     model = MOTRTrack(r'F:\cv\ultralytics_trackingv3\runs\track\train5\weights\best.pt')  # 加载预训练模型（建议用于训练）
    #     test_url = r"E:\data\dancetrack\test\dancetrack" + '{:04d}'.format(i) + r"\img1"
    #     model(test_url, show=False, save_txt=True)  # 查看结果

    # folder_path = r'E:\data\KITTI_TRACKING\testing\images'
    # KITTI_url = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    KITTI_url = [1]
    for i in KITTI_url:
        model = MOTRTrack(r"F:\cv\ultralytics_trackingv3\runs\track\train6\weights\best.pt")  # 加载预训练模型（建议用于训练）

        model(r'E:\data\MOT17\train_yolo_track\images\val', show=True,save=False)  # 查看结果
    # model.train(data="MOT.yaml", epochs=30, batch=1)  # 训练模型
    # metrics = model.val()  # 在验证集上评估模型性能
    # results = model("https://ultralytics.com/images/bus.jpg")  # 对图像进行预测
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式




