import re
import os
import cv2
import dlib
from imutils import face_utils
import shutil
import numpy as np


# 检测输入图像是否需要
def check_img(img_path):
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)

    # file info 文件大小筛选，筛选掉过小的图片
    file_size = os.path.getsize(img_path)
    img_height, img_width = img.shape[:2]
    #10 * 1024代表10kb
    if file_size < 10 * 1024 or img_width < 256 or img_height < 256:
        return False

    # image basic feature 检验图像平滑程度的指标
    img_dy = img[:img_height-1] - img[1:]
    img_dx = img[:, :img_width-1] - img[:, 1:]
    img_gradient = np.mean(np.abs(img_dx)) + np.mean(np.abs(img_dy))
    print(img_path, "img_gradient =", img_gradient)

    ##---------颜色特征筛选---------
    #明亮度筛选
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv_img)
    (v_mean,v_std) = cv2.meanStdDev(v)
    #print("v_mean:",v_mean)

    #人脸图筛选
    detector = dlib.get_frontal_face_detector()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_img, 1)

    #人脸区域筛选
    if len(dets) > 0 or len(dets) <= 3:
        for (i, rect) in enumerate(dets):
            (x, y, w, h) = face_utils.rect_to_bb(dets[0])
            print("w*h:",w*h) #2000
            if w*h < 2000:
                return False

    #纯色筛选
    B,G,R = cv2.split(img)
    (B_mean, B_std) = cv2.meanStdDev(B)
    (G_mean, G_std) = cv2.meanStdDev(G)
    (R_mean, R_std) = cv2.meanStdDev(R)
    mean_var = np.var([B_mean, G_mean, R_mean])
    #print("np.var([B_mean, G_mean, R_mean]):" , mean_var)

    ##---------------------------

    ##边缘提取


    if len(dets)<=0 or len(dets)>3 \
        or img_gradient < 30  \
        or v_mean < 90 :
       # or mean_var > 500 :
        return False
    return True


if __name__ == '__main__':
    root_dir = "E:\download_imgs\download_images\yiren"
    file_suffix = "jpeg|jpg|png"
    remove_dir = root_dir + "/remove"
    if not os.path.exists(remove_dir):
        os.makedirs(remove_dir)
    for img_name in os.listdir(root_dir):
        # print(img_name)
        # 对处理文件的类型进行过滤
        if re.search(file_suffix, img_name) is None:
            continue
        img_path = root_dir + "/" + img_name
        if not check_img(img_path):
            output_path = remove_dir + "/" + img_name
            shutil.move(img_path, output_path)

        # break
