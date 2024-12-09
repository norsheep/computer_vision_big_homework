import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import clip
import torch


def crop_image(img, position):
    # 透视变换
    # img: 原图，position: 四个点的坐标
    def distance(x1,y1,x2,y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))    
    position = position.tolist()
    # 排序，把四个点按照从左到右，从上到下的顺序排列，左上角为第一个点，右下角为第四个点
    # 第一步，按照点的横坐标升序排列
    for i in range(4):
        for j in range(i+1, 4):
            if(position[i][0] > position[j][0]):
                tmp = position[j]
                position[j] = position[i]
                position[i] = tmp
    # 第二步，按照点的纵坐标升序排列
    # 最小的两个一定在左边 
    if position[0][1] > position[1][1]:
        tmp = position[0]
        position[0] = position[1]
        position[1] = tmp
    # 最大的两个一定在右边
    if position[2][1] > position[3][1]:
        tmp = position[2]
        position[2] = position[3]
        position[3] = tmp

    x1, y1 = position[0][0], position[0][1] # 左上角
    x2, y2 = position[2][0], position[2][1] # 左下角
    x3, y3 = position[3][0], position[3][1] # 右下角
    x4, y4 = position[1][0], position[1][1] # 右上角

    corners = np.zeros((4,2), np.float32) 
    corners[0] = [x1, y1]  # 左上角
    corners[1] = [x2, y2]  # 左下角
    corners[2] = [x4, y4]  # 右上角
    corners[3] = [x3, y3]  # 右下角

    img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
    img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

    corners_trans = np.zeros((4,2), np.float32)
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [0, img_height - 1]
    corners_trans[3] = [img_width - 1, img_height - 1]

    transform = cv2.getPerspectiveTransform(corners, corners_trans) # 透视变换矩阵
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height))) # 原图的透视变换
    return dst


def calculate_size(box):
    return (box[2]-box[0]) * (box[3]-box[1])


def calculate_iou(box1, box2):
    # 计算两个矩形框的IoU
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1]) # 最大，左下
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3]) # 最小，右上
    
    interArea = max(0, xB - xA) * max(0, yB - yA) # 交集面积
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea # 并集面积
    iou = interArea / unionArea # IoU
    
    return iou


def crop(image, box, i, text_data=None):
    # image: 图片路径，box: 矩形框坐标，i: 图片编号，text_data: 文本数据
    # 返回按照box裁剪后的图片
    image = Image.open(image)

    if text_data:
        draw = ImageDraw.Draw(image)
        draw.rectangle(((text_data[0], text_data[1]), (text_data[2], text_data[3])), outline="red", width=5)
        # font_size = int((text_data[3] - text_data[1])*0.75)
        # font = ImageFont.truetype("arial.ttf", font_size)
        # draw.text((text_data[0]+5, text_data[1]+5), str(i), font=font, fill="red")

    cropped_image = image.crop(box)
    cropped_image.save(f"./temp/{i}.jpg")
    

def in_box(box, target):
    # box是否在target中
    # box: [x1, y1, x2, y2], target: [x1, y1, x2, y2]
    if (box[0] > target[0]) and (box[1] > target[1]) and (box[2] < target[2]) and (box[3] < target[3]):
        return True
    else:
        return False

    
def crop_for_clip(image, box, i, position):
    image = Image.open(image)
    w, h = image.size
    if position == "left":
        bound = [0, 0, w/2, h]
    elif position == "right":
        bound = [w/2, 0, w, h]
    elif position == "top":
        bound = [0, 0, w, h/2]
    elif position == "bottom":
        bound = [0, h/2, w, h]
    elif position == "top left":
        bound = [0, 0, w/2, h/2]
    elif position == "top right":
        bound = [w/2, 0, w, h/2]
    elif position == "bottom left":
        bound = [0, h/2, w/2, h]
    elif position == "bottom right":
        bound = [w/2, h/2, w, h]
    else:
        bound = [0, 0, w, h]
    # 按照要求画bound
    
    # 如果box在bound中，则按照box裁剪
    if in_box(box, bound):
        cropped_image = image.crop(box)
        cropped_image.save(f"./temp/{i}.jpg")
        return True
    else:
        return False
    
    
def clip_for_icon(clip_model, clip_preprocess, images, prompt):
    # 裁剪图标，返回最符合prompt的图标的位置（应该是prompt任务要求，需要修改）
    image_features = []
    for image_file in images:
        image = clip_preprocess(Image.open(image_file)).unsqueeze(0).to(next(clip_model.parameters()).device)
        image_feature = clip_model.encode_image(image)
        image_features.append(image_feature)
    image_features = torch.cat(image_features)
    
    text = clip.tokenize([prompt]).to(next(clip_model.parameters()).device)  # tokenize the prompt
    text_features = clip_model.encode_text(text)  # encode the text，得到文本特征

    image_features /= image_features.norm(dim=-1, keepdim=True)  # 对特征归一化
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=0).squeeze(0)
    _, max_pos = torch.max(similarity, dim=0)
    pos = max_pos.item()
    
    return pos
