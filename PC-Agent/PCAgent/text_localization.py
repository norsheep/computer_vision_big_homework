import cv2
import numpy as np
from PCAgent.crop import crop_image, calculate_size
from PIL import Image

# 做文本检测和识别
# 具体坐标不重要，反正最后返回的是（min,min）(max,max)


def order_point(coor):
    # 逆时针排序
    arr = np.array(coor).reshape([4, 2])  # 四个点的坐标（x,y）
    sum_ = np.sum(arr, 0)  # 0:[x1+x2+x3+x4, y1+y2+y3+y4]
    centroid = sum_ / arr.shape[0]  # 质心
    theta = np.arctan2(arr[:, 1] - centroid[1],
                       arr[:, 0] - centroid[0])  # (-pi,pi)
    # 求出四个点与质心的连线与x轴的夹角 theta list
    sort_points = arr[np.argsort(theta)]  # 按照theta的大小排序，左下，右下，右上，左上
    sort_points = sort_points.reshape([4, -1])  # 排序后的坐标
    if sort_points[0][0] > centroid[0]:
        # 如果第一个是右下角，调整为左下角，右下角，右上角，左上角
        # 否则无需改变
        sort_points = np.concatenate([sort_points[3:],
                                      sort_points[:3]])  # 左上，右下，左下，右上
    sort_points = sort_points.reshape([4, 2]).astype('float32')
    return sort_points


def longest_common_substring_length(str1, str2):
    # dp法求最长公共子串长度
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]  # m+1行n+1列

    # dp[i][j]表示str1[0...i-1]和str2[0...j-1]的最长公共子串长度
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def ocr(image_path, ocr_detection, ocr_recognition):
    # 传入图片路径，ocr检测和识别函数，返回文本和坐标
    text_data = []
    coordinate = []

    image_full = cv2.imread(image_path)
    try:
        det_result = ocr_detection(image_full)
    except:
        print('not text detected')
        return ['no text'], [[0, 0, 0, 0]]
    # det_result = {'polygons': np.array([[x1,y1,x2,y2,x3,y3,x4,y4],...])}
    det_result = det_result['polygons']
    # 遍历检测到的每一个文本框
    for i in range(det_result.shape[0]):
        pts = order_point(det_result[i])
        image_crop = crop_image(image_full, pts)

        try:
            result = ocr_recognition(image_crop)['text'][0]
        except:
            continue

        box = [int(e) for e in list(pts.reshape(-1))]
        box = [box[0], box[1], box[4],
               box[5]]  # 一个文本框仅需要两个坐标即可表示（min,min,max,max）

        text_data.append(result)  # 文本
        coordinate.append(box)  # 坐标

    return text_data, coordinate
