import PIL
import cv2

import pickle
import random
import numpy as np
import shutil
import os,sys
import json
import openpyxl
import pandas as pd

import numpy as np
import matplotlib
import cv2

# Force matplotlib to not use any Xwindows backend.

import matplotlib.pyplot as plt
import random
import pylab
# from skimage import transform
# display plots in this notebook

import os
import sys
from PIL import Image

import cv2
import argparse

import pickle as pkl

# entry = {}
# entry['image_id'] = data['image_id']
# entry['file_name'] = data['image_name']
#
# entry['ann_bboxes'] = data['ann_bboxes'].tolist()  # (7, 4)
# entry['neighbor_bbox'] = data['neighbor_bbox'][pred_ix].tolist()
# entry['cxt_bboxes'] = data['cxt_bboxes'][pred_ix].tolist()
#
# entry['sent_id'] = sent_id
# entry['sent'] = loader.decode_labels(label.data.cpu().numpy())[0] # gd-truth sent
# entry['erased_sent'] = loader.decode_labels(erased_sent_label.data.cpu().numpy())[0] # gd-truth sent
#
# entry['gd_ann_id'] = data['ann_ids'][gd_ix] # groundtruth annotation bbox
# entry['pred_ann_id'] = data['ann_ids'][pred_ix] # predicted annotation bbox
#
# entry['gd_ann_box'] = data['ann_bboxes'][gd_ix].tolist()  # groundtruth annotation bbox
# entry['pred_ann_box'] = data['ann_bboxes'][pred_ix].tolist()  # predicted annotation bbox
#
# entry['rel_ann_id'] = data['cxt_ann_ids'][pred_ix][rel_ix]        # rel ann_id
# entry['all_scores'] = scores_orig.tolist()
#
# entry['pred_score'] = scores_orig.tolist()[pred_ix]
# entry['sub_grid_attn'] = sub_grid_attn[pred_ix].data.cpu().numpy().tolist() # list of 49 attn
# entry['rel_r_box'] = data['cxt_bboxes'][pred_ix][rel_ix].tolist()
# entry['loc_r_box'] = data['neighbor_bbox'][pred_ix][torch.max(neigh_attn[:, 1:], 1)[1][pred_ix].item()].tolist()
#
# entry['sub_attn'] = sub_attn[pred_ix].data.cpu().numpy().tolist() # list of seq_len attn
# entry['loc_attn'] = loc_attn[pred_ix].data.cpu().numpy().tolist() # list of seq_len attn
# entry['rel_attn'] = rel_attn[pred_ix].data.cpu().numpy().tolist() # list of seq_len attn
#
# entry['weights'] = weights[pred_ix].data.cpu().numpy().tolist()   # list of 2 weights
# entry['sub_matching_score'] = sub_matching_scores[pred_ix].data.cpu().numpy().tolist()   # list of 2 weights
# entry['loc_matching_score'] = loc_matching_scores[pred_ix].data.cpu().numpy().tolist()   # list of 2 weights
# entry['rel_matching_score'] = rel_matching_scores[pred_ix].data.cpu().numpy().tolist()   # list of 2 weights
#
# entry['pred_atts'] = pred_atts # list of (att_wd, score)

from PIL import Image, ImageDraw, ImageFont
class ImgText:
  font = ImageFont.truetype("micross.ttf", 24)
  def __init__(self, text, w, h):
    # 预设宽度 可以修改成你需要的图片宽度
    self.width = int(w)
    self.height = int(h)
    # 文本
    self.text = text
    # 段落 , 行数, 行高
    self.duanluo, self.note_height, self.line_height = self.split_text()
  def get_duanluo(self, text):
    txt = Image.new('RGBA', (100, 100), (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt)
    # 所有文字的段落
    duanluo = ""
    # 宽度总和
    sum_width = 0
    # 几行
    line_count = 1
    # 行高
    line_height = 0
    for char in text:
      width, height = draw.textsize(char, ImgText.font)
      sum_width += width
      if sum_width > self.width: # 超过预设宽度就修改段落 以及当前行数
        line_count += 1
        sum_width = 0
        duanluo += '\n'
      duanluo += char
      line_height = max(height, line_height)
    if not duanluo.endswith('\n'):
      duanluo += '\n'
    return duanluo, line_height, line_count
  def split_text(self):
    # 按规定宽度分组
    max_line_height, total_lines = 0, 0
    allText = []
    for text in self.text.split('\n'):
      duanluo, line_height, line_count = self.get_duanluo(text)
      max_line_height = max(line_height, max_line_height)
      total_lines += line_count
      allText.append((duanluo, line_count))
    line_height = max_line_height
    total_height = total_lines * line_height
    return allText, total_height, line_height
  def draw_text(self, res_name):
    """
    绘图以及文字
    :return:
    """
    # note_img = Image.open("001.png").convert("RGBA")
    note_img = Image.new('RGB', (self.width, self.height))  # 创建一个新图
    draw = ImageDraw.Draw(note_img)
    # 左上角开始
    x, y = 0, 0
    for duanluo, line_count in self.duanluo:
      draw.text((x, y), duanluo, fill=(255, 255, 255), font=ImgText.font)
      y += self.line_height * line_count
    note_img.save(res_name)


def draw_rec(image_name, bboxes, tags, res_name, gt_box=None, pred_box=None, ref_box=None, if_nor=True,
             pred_tag=None, gd_tag=None, ref_tag=None):
    im = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), -1)
    # im = cv2.imread(image_name)
    if if_nor:
        for i, rec in enumerate(bboxes):
            # 画框
            # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)参数img为原图，左上角坐标，右下角坐标，线的颜色，线宽
            cv2.rectangle(im, (int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])),
                          (255, 255, 255), 2)
            # 画圆
            # cv2.circle(im, (100, 100), 10, (0, 0, 255), -1)#图片，圆心坐标，半径，颜色，-1代表实心圆
            # 添加文本cv2.putText(img, str(i), (123, 456)), font, 2, (0, 255, 0), 3)
            # 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            if tags is not None:
                cv2.putText(im, '{:.3f}'.format(tags[i]), (int(rec[0]), int(rec[1]+20)), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                            (255, 255, 255), thickness=2)
            # cv2.imshow('head', im)

        # 保存画框后的图片
        cv2.imencode('.jpg', im)[1].tofile(res_name)

    else:

        for i, rec in enumerate(bboxes):
            # 画框
            # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)参数img为原图，左上角坐标，右下角坐标，线的颜色，线宽
            cv2.rectangle(im, (int(rec[0]), int(rec[1])), (int(rec[0])+int(rec[2]), int(rec[1])+int(rec[3])), (255, 255, 255), 2)
            # 画圆
            # cv2.circle(im, (100, 100), 10, (0, 0, 255), -1)#图片，圆心坐标，半径，颜色，-1代表实心圆
            # 添加文本cv2.putText(img, str(i), (123, 456)), font, 2, (0, 255, 0), 3)
            # 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            if tags is not None:
                cv2.putText(im, '{:.3f}'.format(tags[i]), (int(rec[0]), int(rec[1])+20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), thickness=2)
            # cv2.imshow('head', im)
        # if gt_box is not None:
        #     cv2.rectangle(im, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[0]) + int(gt_box[2]), int(gt_box[1]) + int(gt_box[3])), (0, 255, 0), 2)
        # if pred_box is not None:
        #     cv2.rectangle(im, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[0]) + int(pred_box[2]), int(pred_box[1]) + int(pred_box[3])), (0, 0, 255), 2)

    if pred_box is not None:
        cv2.rectangle(im, (int(pred_box[0]), int(pred_box[1])),
                      (int(pred_box[2]), int(pred_box[3])), (0, 0, 255), 2)
        if pred_tag is not  None:
            cv2.putText(im, '{:.3f}'.format(pred_tag), (int(pred_box[0]), int(pred_box[1]+20)), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (255, 255, 255), thickness=2)
    if ref_box is not None:
        cv2.rectangle(im, (int(ref_box[0]), int(ref_box[1])),
                      (int(ref_box[0]) + int(ref_box[2]), int(ref_box[1]) + int(ref_box[3])), (255, 0, 0), 2)
        if ref_tag is not None:
            cv2.putText(im, '{:.3f}'.format(ref_tag), (int(ref_box[0]), int(ref_box[1]+20)), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (255, 255, 255), thickness=2)
    if gt_box is not None:
        cv2.rectangle(im, (int(gt_box[0]), int(gt_box[1])),
                      (int(gt_box[2]), int(gt_box[3])), (0, 255, 0), 2)
        if gd_tag is not None:
            cv2.putText(im, '{:.3f}'.format(gd_tag), (int(gt_box[0]), int(gt_box[1]+20)), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (255, 255, 255), thickness=2)

    # 保存画框后的图片
    cv2.imencode('.jpg', im)[1].tofile(res_name)


def plt_text(w, h, text, tmp_name):
    n = ImgText(
        text, w, h)
    n.draw_text(tmp_name)

    # bk_img = np.zeros((int(h), int(w), 3), np.uint8)
    # bk_img.fill(0)
    #
    # y0, dy = 10, 10
    #
    # for i, txt in enumerate(text.split('\n')):
    #     y = y0 + i * (dy + 10)
    #     cv2.putText(bk_img, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    #
    # cv2.imwrite(tmp_name, bk_img)
    return tmp_name


def compute_attention_matrix(image_name, boxes, object_weights):
    from PIL import Image
    im = Image.open(image_name, 'r')
    width = im.size[0]
    height = im.size[1]
    if boxes is None:
        boxes = get_grids_bboxes(width, height)
    atten_weight = np.zeros(shape=(height, width), dtype=np.float32)

    for index in range(len(boxes)):
        bbox = np.rint(boxes[index])
        # print(bbox)
        atten_weight[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])] += object_weights[index]
        # atten_weight[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] += object_weights[index]
    return atten_weight


def draw_image(weights, image_name, des_image_image):
    import cv2
    cam = weights - np.min(weights)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)

    img = cv2.imread(image_name)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    result = heatmap * 0.4 + img * 0.5
    cv2.imwrite(des_image_image, result)


def joint_an_instance(image_names, locates, result_name, width):

    im = Image.open(image_names[0], 'r')
    # width = im.size[0]
    height = im.size[1]
    to_image = Image.new('RGB', (int(width), 4 * height))  # 创建一个新图
    for i, img_name in enumerate(image_names):
        locat = [int(tmp) for tmp in locates[i]]
        to_image.paste(Image.open(img_name), locat)

    to_image.save(result_name)


def crop_img(src_img, bboxes, res_name):
    # -*-coding:utf-8-*-
    im = Image.open(src_img)
    width = im.size[0]
    height = im.size[1]
    '''
    裁剪：传入一个元组作为参数
    元组里的元素分别是：（距离图片左边界距离x， 距离图片上边界距离y，距离图片左边界距离+裁剪框宽度x+w，距离图片上边界距离+裁剪框高度y+h）
    '''
    # 截取图片中一块宽和高都是250的
    # region = im.crop((bboxes[0], bboxes[1], width - bboxes[2] - bboxes[0], height - bboxes[3] - bboxes[1]))
    region = im.crop((bboxes[0], bboxes[1], bboxes[2], bboxes[3]))

    region.save(res_name)


def get_grids_bboxes(w, h):
    bboxes = []
    x = 0.0
    y = 0.0
    for i in range(7):
        x = 0.0
        for j in range(7):
            bboxes.append([x, y, w/7.0, h/7.0])
            x = x + w/7.0
        y = y + h/7.0
    return bboxes


def draw_attention(image_name, object_weights, des_image_image):

    # map = np.array(map[0].detach().cpu().numpy() * 255).astype(np.uint8)
    map = np.array(object_weights).reshape(7, 7)

    map = map - np.min(map)
    map = map / np.max(map)
    map = np.uint8(255 * map)

    # oriimg = transforms.ToPILImage()(gloImage.squeeze().detach().cpu())
    oriimg = PIL.Image.open(image_name)
    shapes = (oriimg.size[0], oriimg.size[1])
    oriimg = oriimg.resize(shapes, Image.ANTIALIAS)
    if len(np.array(oriimg).shape) == 2:
        oriimg = Image.fromarray(np.array(oriimg).reshape((oriimg.size[1], oriimg.size[0], 1)).repeat(3, 2))
    cm = plt.get_cmap('jet')
    # min-max normalize the image, you can skip this step
    colored_map = cm(map)
    # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
    # But we want to convert to RGB in uint8 and save it:
    heatmap = Image.fromarray((colored_map[:, :, :3] * 255).astype(np.uint8))
    upheatmap = heatmap.resize(shapes, Image.BILINEAR)
    # oriheatmap = heatmap.resize(shapes, Image.NEAREST)
    # upheatmap.save(des_image_image + "1.jpg")
    # oriheatmap.save(des_image_image + "2.jpg")

    blendimg = Image.blend(oriimg, upheatmap, alpha=0.5)
    blendimg.save(des_image_image)


def plot_joint():
    img_path = "E:\data\VQA V2\\train2014"
    dest_path = "F:\wrong_images"
    data_path = "E:\code\python\\visual_grounding\MINE\\refCOCO"
    tmp_path = "E:\code\python\\visual_grounding\MINE\\tmp"
    info = json.load(
        open(os.path.join(data_path, "wrong_info_val.json"), 'r')
    )

    for entry in info:

        img_name = entry['file_name']
        sent_id = entry['sent_id']
        image_id = entry['image_id']

        image_names = []
        joint_locations = []
        res_name = os.path.join(dest_path, str(image_id) + "_" + str(sent_id) + '.jpg')
        ori_img_name = os.path.join(img_path, img_name)
        print(res_name)

        ori_im = Image.open(ori_img_name, 'r')
        width = ori_im.size[0]
        height = ori_im.size[1]

        ann_boxes = entry['ann_bboxes']
        scores = entry['all_scores']
        gd_ann_box = entry['gd_ann_box']
        pred_ann_box = entry['pred_ann_box']
        draw_rec(ori_img_name, ann_boxes, scores, os.path.join(tmp_path, 'ori_img.jpg'), gt_box=gd_ann_box, pred_box=pred_ann_box)
        image_names.append(os.path.join(tmp_path, 'ori_img.jpg'))
        joint_locations.append((0, 0))

        plt_text(width, height, " original info: \n image_id: %s \n sent_id: %s" % (str(image_id), str(sent_id)), os.path.join(tmp_path, 'img_id_sent_id.jpg'))
        image_names.append(os.path.join(tmp_path, 'img_id_sent_id.jpg'))
        joint_locations.append((0, height))

        crop_img(ori_img_name, pred_ann_box, os.path.join(tmp_path, 'croped_region.jpg'))
        image_names.append(os.path.join(tmp_path, 'croped_region.jpg'))
        joint_locations.append((width + 10, 0))

        sentence = entry['sent']
        erased_sent = entry['erased_sent']
        plt_text(pred_ann_box[2], height, " Select Region \n sentence: \n  %s \n erased_sent: \n  %s" % (sentence, erased_sent),
                 os.path.join(tmp_path, 'sent_e_sent.jpg'))
        image_names.append(os.path.join(tmp_path, 'sent_e_sent.jpg'))
        joint_locations.append((width + 10, height))

        weights = entry['weights']

        atten_weight = compute_attention_matrix(os.path.join(tmp_path, 'croped_region.jpg'),
                                                None, entry['sub_grid_attn'])
        draw_image(atten_weight, os.path.join(tmp_path, 'croped_region.jpg'), os.path.join(tmp_path, 'img_sub.jpg'))
        # draw_rec(os.path.join(tmp_path, 'img_sub.jpg'), get_grids_bboxes(pred_ann_box[2]-pred_ann_box[0], pred_ann_box[3]-pred_ann_box[1]), None,
        #          os.path.join(tmp_path, 'img_sub.jpg'), None, None, if_nor=False)
        image_names.append(os.path.join(tmp_path, 'img_sub.jpg'))
        joint_locations.append((width + 20 + pred_ann_box[2], 0))


        sub_matching_score = entry['sub_matching_score']
        sub_attn = entry['sub_attn']
        plt_text(pred_ann_box[2], height, " Subject Module \n module_weight: %s, \n module_score: %s, \n"
                                               " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[0],2)),
                                                                                 str(round(sub_matching_score,2)),
                                                                                 sentence,
                                                                                 json.dumps(np.around(np.array(sub_attn), 2).tolist())),
                 os.path.join(tmp_path, 'sen_sub.jpg'))
        image_names.append(os.path.join(tmp_path, 'sen_sub.jpg'))
        joint_locations.append((width + 20 + pred_ann_box[2], height))


        neighbor_bbox = entry['neighbor_bbox']

        loc_r_box = entry['loc_r_box']
        draw_rec(ori_img_name, neighbor_bbox, None, os.path.join(tmp_path, 'img_loc.jpg'), ref_box=loc_r_box, pred_box=pred_ann_box, if_nor=False)
        image_names.append(os.path.join(tmp_path, 'img_loc.jpg'))
        joint_locations.append((width + 30 + 2*pred_ann_box[2], 0))


        loc_attn = entry['loc_attn']
        loc_matching_score = entry['loc_matching_score']
        plt_text(width, height, " Location Module \n module_weight: %s, \n module_score: %s, \n"
                                               " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[1], 2)),
                                                                                 str(round(loc_matching_score, 2)),
                                                                                 sentence,
                                                                                 json.dumps(np.around(np.array(loc_attn), 2).tolist())),
                 os.path.join(tmp_path, 'sen_loc.jpg'))
        image_names.append(os.path.join(tmp_path, 'sen_loc.jpg'))
        joint_locations.append((width + 30 + 2*pred_ann_box[2], height))


        cxt_bboxes = entry['cxt_bboxes']
        rel_r_box = entry['rel_r_box']
        draw_rec(ori_img_name, cxt_bboxes, None, os.path.join(tmp_path, 'img_rel.jpg'), ref_box=rel_r_box, pred_box=pred_ann_box, if_nor=False)
        image_names.append(os.path.join(tmp_path, 'img_rel.jpg'))
        joint_locations.append((2*width + 40 + 2*pred_ann_box[2], 0))


        rel_attn = entry['rel_attn']
        rel_matching_score = entry['rel_matching_score']
        plt_text(width, height, " Relation Module \n module_weight: %s, \n module_score: %s, \n"
                                               " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[2], 2)),
                                                                                 str(round(rel_matching_score, 2)),
                                                                                 sentence,
                                                                                 json.dumps(np.around(np.array(rel_attn), 2).tolist())),
                 os.path.join(tmp_path, 'sen_rel.jpg'))
        image_names.append(os.path.join(tmp_path, 'sen_rel.jpg'))
        joint_locations.append((2*width + 40 + 2*pred_ann_box[2], height))

        joint_an_instance(image_names, joint_locations, res_name, 3*width + 50 + 2*pred_ann_box[2])

        # break


def plot_joint2():
    img_path = "E:\data\VQA V2\\train2014"
    dest_path = "F:\wrong_images2"
    data_path = "E:\code\python\\visual_grounding\MINE\\refCOCO"
    tmp_path = "E:\code\python\\visual_grounding\MINE\\tmp"
    info = json.load(
        open(os.path.join(data_path, "wrong_info2_val.json"), 'r')
    )

    for number, entry in enumerate(info):

        img_name = entry['file_name']
        sent_id = entry['sent_id']
        image_id = entry['image_id']
        gd_ix = entry['gd_ix']
        pred_ix = entry['pred_ix']
        rel_ixs = entry['rel_ixs']
        neigh_attn = np.array(entry['neigh_attn'])

        image_names = []
        joint_locations = []
        res_name = os.path.join(dest_path, str(image_id) + "_" + str(sent_id) + '.jpg')
        ori_img_name = os.path.join(img_path, img_name)
        print(number, res_name)
        if number < 831:
            continue

        ori_im = Image.open(ori_img_name, 'r')
        width = ori_im.size[0]
        height = ori_im.size[1]

        ann_boxes = entry['ann_bboxes']
        scores = entry['all_scores']
        gd_ann_box = entry['gd_ann_box']
        pred_ann_box = entry['pred_ann_box']
        if image_id == 80826 and sent_id == 122029:
            print("")
        draw_rec(ori_img_name, ann_boxes, scores, os.path.join(tmp_path, 'ori_img.jpg'), gt_box=gd_ann_box, pred_box=pred_ann_box)
        image_names.append(os.path.join(tmp_path, 'ori_img.jpg'))
        joint_locations.append((0, 0))

        image_names.append(os.path.join(tmp_path, 'ori_img.jpg'))
        joint_locations.append((0, 2*height))

        # the wrong info
        plt_text(width, height, " original info: \n\n image_id: %s \n sent_id: %s \n wrong score: %s" %
                 (str(image_id), str(sent_id), str(round(scores[pred_ix], 2))), os.path.join(tmp_path, 'w_img_id_sent_id.jpg'))
        image_names.append(os.path.join(tmp_path, 'w_img_id_sent_id.jpg'))
        joint_locations.append((0, height))

        crop_img(ori_img_name, pred_ann_box, os.path.join(tmp_path, 'w_croped_region.jpg'))
        image_names.append(os.path.join(tmp_path, 'w_croped_region.jpg'))
        joint_locations.append((width + 10, 0))

        sentence = entry['sent']
        erased_sent = entry['erased_sent']
        plt_text(width, height, " Select Region \n\n sentence: \n  %s \n erased_sent: \n  %s" % (sentence, erased_sent),
                 os.path.join(tmp_path, 'w_sent_e_sent.jpg'))
        image_names.append(os.path.join(tmp_path, 'w_sent_e_sent.jpg'))
        joint_locations.append((width + 10, height))

        weights = entry['weights'][pred_ix]

        atten_weight = compute_attention_matrix(os.path.join(tmp_path, 'w_croped_region.jpg'),
                                                None, entry['sub_grid_attn'][pred_ix])
        draw_image(atten_weight, os.path.join(tmp_path, 'w_croped_region.jpg'), os.path.join(tmp_path, 'w_img_sub.jpg'))
        image_names.append(os.path.join(tmp_path, 'w_img_sub.jpg'))
        joint_locations.append((width + 20 + width, 0))

        sub_matching_score = entry['sub_matching_score'][pred_ix]
        sub_attn = entry['sub_attn'][pred_ix]
        plt_text(width, height, " Subject Module \n\n module_weight: %s, \n module_score: %s, \n"
                                               " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[0],2)),
                                                                                 str(round(sub_matching_score,2)),
                                                                                 sentence,
                                                                                 json.dumps(np.around(np.array(sub_attn), 2).tolist())),
                 os.path.join(tmp_path, 'w_sen_sub.jpg'))
        image_names.append(os.path.join(tmp_path, 'w_sen_sub.jpg'))
        joint_locations.append((width + 20 +width, height))

        neighbor_bbox = entry['neighbor_bbox'][pred_ix]

        loc_r_box = np.array(entry['loc_r_box'])[pred_ix][np.argmax(neigh_attn[:, 1:][pred_ix])].tolist()
        draw_rec(ori_img_name, neighbor_bbox, neigh_attn[pred_ix].tolist(), os.path.join(tmp_path, 'w_img_loc.jpg'),
                 ref_box=loc_r_box, pred_box=pred_ann_box, if_nor=False, pred_tag=neigh_attn[pred_ix][0])
        image_names.append(os.path.join(tmp_path, 'w_img_loc.jpg'))
        joint_locations.append((width + 30 + 2*width, 0))


        loc_attn = entry['loc_attn'][pred_ix]
        loc_matching_score = entry['loc_matching_score'][pred_ix]
        plt_text(width, height, " Location Module \n\n module_weight: %s, \n module_score: %s, \n"
                                               " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[1], 2)),
                                                                                 str(round(loc_matching_score, 2)),
                                                                                 sentence,
                                                                                 json.dumps(np.around(np.array(loc_attn), 2).tolist())),
                 os.path.join(tmp_path, 'w_sen_loc.jpg'))
        image_names.append(os.path.join(tmp_path, 'w_sen_loc.jpg'))
        joint_locations.append((width + 30 + 2*width, height))


        cxt_bboxes = entry['cxt_bboxes'][pred_ix]
        rel_r_box = entry['rel_r_box'][pred_ix][rel_ixs[pred_ix]]
        draw_rec(ori_img_name, cxt_bboxes, None, os.path.join(tmp_path, 'w_img_rel.jpg'), ref_box=rel_r_box,
                 pred_box=pred_ann_box, if_nor=False)
        image_names.append(os.path.join(tmp_path, 'w_img_rel.jpg'))
        joint_locations.append((2*width + 40 + 2*width, 0))


        rel_attn = entry['rel_attn'][pred_ix]
        rel_matching_score = entry['rel_matching_score'][pred_ix]
        plt_text(width, height, " Relation Module \n\n module_weight: %s, \n module_score: %s, \n"
                                               " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[2], 2)),
                                                                                 str(round(rel_matching_score, 2)),
                                                                                 sentence,
                                                                                 json.dumps(np.around(np.array(rel_attn), 2).tolist())),
                 os.path.join(tmp_path, 'w_sen_rel.jpg'))
        image_names.append(os.path.join(tmp_path, 'w_sen_rel.jpg'))
        joint_locations.append((2*width + 40 + 2*width, height))

        # #######################################################################################
        # the ground_truth info
        plt_text(width, height, " original info: \n\n image_id: %s \n sent_id: %s \n gt score: %s" %
                 (str(image_id), str(sent_id), str(round(scores[gd_ix], 2))),
                 os.path.join(tmp_path, 'r_img_id_sent_id.jpg'))
        image_names.append(os.path.join(tmp_path, 'r_img_id_sent_id.jpg'))
        joint_locations.append((0, 3*height))

        crop_img(ori_img_name, gd_ann_box, os.path.join(tmp_path, 'r_croped_region.jpg'))
        image_names.append(os.path.join(tmp_path, 'r_croped_region.jpg'))
        joint_locations.append((width + 10, 2*height))

        sentence = entry['sent']
        erased_sent = entry['erased_sent']
        plt_text(width, height,
                 " Select Region \n\n sentence: \n  %s \n erased_sent: \n  %s" % (sentence, erased_sent),
                 os.path.join(tmp_path, 'r_sent_e_sent.jpg'))
        image_names.append(os.path.join(tmp_path, 'r_sent_e_sent.jpg'))
        joint_locations.append((width + 10, 3*height))

        weights = entry['weights'][gd_ix]

        atten_weight = compute_attention_matrix(os.path.join(tmp_path, 'r_croped_region.jpg'),
                                                None, entry['sub_grid_attn'][gd_ix])
        draw_image(atten_weight, os.path.join(tmp_path, 'r_croped_region.jpg'), os.path.join(tmp_path, 'r_img_sub.jpg'))
        image_names.append(os.path.join(tmp_path, 'r_img_sub.jpg'))
        joint_locations.append((width + 20 + width, 2*height))

        sub_matching_score = entry['sub_matching_score'][gd_ix]
        sub_attn = entry['sub_attn'][gd_ix]
        plt_text(width, height, " Subject Module \n\n module_weight: %s, \n module_score: %s, \n"
                                          " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[0], 2)),
                                                                                   str(round(sub_matching_score, 2)),
                                                                                   sentence,
                                                                                   json.dumps(
                                                                                       np.around(np.array(sub_attn),
                                                                                                 2).tolist())),
                 os.path.join(tmp_path, 'r_sen_sub.jpg'))
        image_names.append(os.path.join(tmp_path, 'r_sen_sub.jpg'))
        joint_locations.append((width + 20 + width, 3*height))

        neighbor_bbox = entry['neighbor_bbox'][gd_ix]

        loc_r_box = np.array(entry['loc_r_box'])[gd_ix][np.argmax(neigh_attn[:, 1:][gd_ix])].tolist()
        draw_rec(ori_img_name, neighbor_bbox, None, os.path.join(tmp_path, 'r_img_loc.jpg'), ref_box=loc_r_box,
                 gt_box=gd_ann_box, if_nor=False)
        image_names.append(os.path.join(tmp_path, 'r_img_loc.jpg'))
        joint_locations.append((width + 30 + 2 * width, 2*height))

        loc_attn = entry['loc_attn'][gd_ix]
        loc_matching_score = entry['loc_matching_score'][gd_ix]
        plt_text(width, height, " Location Module \n\n module_weight: %s, \n module_score: %s, \n"
                                " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[1], 2)),
                                                                         str(round(loc_matching_score, 2)),
                                                                         sentence,
                                                                         json.dumps(np.around(np.array(loc_attn),
                                                                                              2).tolist())),
                 os.path.join(tmp_path, 'r_sen_loc.jpg'))
        image_names.append(os.path.join(tmp_path, 'r_sen_loc.jpg'))
        joint_locations.append((width + 30 + 2 * width, 3*height))

        cxt_bboxes = entry['cxt_bboxes'][gd_ix]
        rel_r_box = entry['rel_r_box'][gd_ix][rel_ixs[gd_ix]]
        draw_rec(ori_img_name, cxt_bboxes, None, os.path.join(tmp_path, 'r_img_rel.jpg'), ref_box=rel_r_box,
                 gt_box=gd_ann_box, if_nor=False)
        image_names.append(os.path.join(tmp_path, 'r_img_rel.jpg'))
        joint_locations.append((2 * width + 40 + 2 * width, 2*height))

        rel_attn = entry['rel_attn'][gd_ix]
        rel_matching_score = entry['rel_matching_score'][gd_ix]
        plt_text(width, height, " Relation Module \n\n module_weight: %s, \n module_score: %s, \n"
                                " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[2], 2)),
                                                                         str(round(rel_matching_score, 2)),
                                                                         sentence,
                                                                         json.dumps(np.around(np.array(rel_attn),
                                                                                              2).tolist())),
                 os.path.join(tmp_path, 'r_sen_rel.jpg'))
        image_names.append(os.path.join(tmp_path, 'r_sen_rel.jpg'))
        joint_locations.append((2 * width + 40 + 2 * width, 3*height))

        joint_an_instance(image_names, joint_locations, res_name, 3*width + 50 + 2*width)

        # break


def softmax(x):
    """ softmax function """

    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行

    x -= np.max(x, axis=0, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素

    # print("减去行最大值 ：\n", x)

    x = np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

    return x


def plot_jointw_r(entry, ox, oy, tmp_prefix, if_wrong=True):

    img_path = "E:\data\VQA V2\\train2014"
    tmp_path = "E:\code\python\\visual_grounding\MINE\\tmp"

    img_name = entry['file_name']
    sent_id = entry['sent_id']
    image_id = entry['image_id']
    gd_ix = entry['gd_ix']
    pred_ix = entry['pred_ix']
    rel_ixs = entry['rel_ixs']
    neigh_attn = np.array(entry['neigh_attn'])
    rel_att = np.array(entry['rel_att'])

    image_names = []
    joint_locations = []
    ori_img_name = os.path.join(img_path, img_name)

    ori_im = Image.open(ori_img_name, 'r')
    width = ori_im.size[0]
    height = ori_im.size[1]

    ann_boxes = entry['ann_bboxes']
    scores = entry['all_scores']
    gd_ann_box = entry['gd_ann_box']
    pred_ann_box = entry['pred_ann_box']

    draw_rec(ori_img_name, ann_boxes, scores, os.path.join(tmp_path, tmp_prefix + '_ori_img.jpg'), gt_box=gd_ann_box, pred_box=pred_ann_box)
    image_names.append(os.path.join(tmp_path, tmp_prefix + '_ori_img.jpg'))
    joint_locations.append((ox, oy))

    scores_nor = softmax(np.array(scores)).tolist()

    plt_text(width, height, " %s original info: \n \n image_id: %s \n sent_id: %s \n w_nor_score: %s \n r_nor_score: %s" %
             (tmp_prefix, str(image_id), str(sent_id), str(scores_nor[pred_ix]), str(scores_nor[gd_ix])), os.path.join(tmp_path, tmp_prefix + '_img_id_sent_id.jpg'))
    image_names.append(os.path.join(tmp_path, tmp_prefix + '_img_id_sent_id.jpg'))
    joint_locations.append((ox, oy + height))

    if if_wrong:
        # the wrong info
        crop_img(ori_img_name, pred_ann_box, os.path.join(tmp_path, tmp_prefix + '_w_croped_region.jpg'))
        image_names.append(os.path.join(tmp_path, tmp_prefix + '_w_croped_region.jpg'))
        joint_locations.append((ox + width + 10, oy))

        sentence = entry['sent']
        erased_sent = entry['erased_sent']
        plt_text(width, height, " Select Region \n \n sentence: \n  %s \n erased_sent: \n  %s \n wrong score: %s" %
                 (sentence, erased_sent, str(round(scores[pred_ix], 2))),
                 os.path.join(tmp_path, tmp_prefix + '_w_sent_e_sent.jpg'))
        image_names.append(os.path.join(tmp_path, tmp_prefix + '_w_sent_e_sent.jpg'))
        joint_locations.append((ox + 2*width + 10, oy))

        weights = entry['weights'][pred_ix]

        draw_attention(os.path.join(tmp_path, tmp_prefix + '_w_croped_region.jpg'), entry['sub_grid_attn'][gd_ix],
                       os.path.join(tmp_path, tmp_prefix + '_w_img_sub.jpg'))
        image_names.append(os.path.join(tmp_path, tmp_prefix + '_w_img_sub.jpg'))
        joint_locations.append((ox + 3*width + 20, oy))

        sub_matching_score = entry['sub_matching_score'][pred_ix]
        sub_attn = entry['sub_attn'][pred_ix]
        plt_text(width, height, " Subject Module \n \n module_weight: %s, \n module_score: %s, \n"
                                               " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[0],2)),
                                                                                 str(round(sub_matching_score,2)),
                                                                                 sentence,
                                                                                 json.dumps(np.around(np.array(sub_attn), 2).tolist())),
                 os.path.join(tmp_path, tmp_prefix + '_w_sen_sub.jpg'))
        image_names.append(os.path.join(tmp_path, tmp_prefix + '_w_sen_sub.jpg'))
        joint_locations.append((ox + 4*width + 20, oy))

        neighbor_bbox = entry['neighbor_bbox'][pred_ix]

        loc_r_box = np.array(entry['loc_r_box'])[pred_ix][np.argmax(neigh_attn[:, 1:][pred_ix])].tolist()
        draw_rec(ori_img_name, neighbor_bbox, neigh_attn[pred_ix][1:].tolist(), os.path.join(tmp_path, tmp_prefix + '_w_img_loc.jpg'),
                 ref_box=loc_r_box, pred_box=pred_ann_box, if_nor=False, pred_tag=neigh_attn[pred_ix][0])
        image_names.append(os.path.join(tmp_path, tmp_prefix + '_w_img_loc.jpg'))
        joint_locations.append((ox + 30 + 5*width, oy))


        loc_attn = entry['loc_attn'][pred_ix]
        loc_matching_score = entry['loc_matching_score'][pred_ix]
        plt_text(width, height, " Location Module \n \n module_weight: %s, \n module_score: %s, \n"
                                               " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[1], 2)),
                                                                                 str(round(loc_matching_score, 2)),
                                                                                 sentence,
                                                                                 json.dumps(np.around(np.array(loc_attn), 2).tolist())),
                 os.path.join(tmp_path, tmp_prefix + '_w_sen_loc.jpg'))
        image_names.append(os.path.join(tmp_path, tmp_prefix + '_w_sen_loc.jpg'))
        joint_locations.append((ox + 30 + 6*width, oy))

        cxt_bboxes = entry['cxt_bboxes'][pred_ix]
        rel_r_box = np.array(entry['rel_r_box'])[pred_ix][np.argmax(rel_att[pred_ix])].tolist()
        draw_rec(ori_img_name, cxt_bboxes, rel_att[pred_ix].tolist(), os.path.join(tmp_path, tmp_prefix + '_w_img_rel.jpg'),
                 ref_box=rel_r_box, pred_box=pred_ann_box, if_nor=False)
        image_names.append(os.path.join(tmp_path, tmp_prefix + '_w_img_rel.jpg'))
        joint_locations.append((ox + 7*width + 40, oy))

        rel_attn = entry['rel_attn'][pred_ix]
        rel_matching_score = entry['rel_matching_score'][pred_ix]
        plt_text(width, height, " Relation Module \n \n module_weight: %s, \n module_score: %s, \n"
                                               " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[2], 2)),
                                                                                 str(round(rel_matching_score, 2)),
                                                                                 sentence,
                                                                                 json.dumps(np.around(np.array(rel_attn), 2).tolist())),
                 os.path.join(tmp_path, tmp_prefix + '_w_sen_rel.jpg'))
        image_names.append(os.path.join(tmp_path, tmp_prefix + '_w_sen_rel.jpg'))
        joint_locations.append((ox + 8*width + 40, oy))

    # #######################################################################################
    # the ground_truth info
    # plt_text(width, height, " original info: \n image_id: %s \n sent_id: %s \n gt score: %s" %
    #          (str(image_id), str(sent_id), str(round(scores[gd_ix], 2))),
    #          os.path.join(tmp_path, tmp_prefix + '_r_img_id_sent_id.jpg'))
    # image_names.append(os.path.join(tmp_path, tmp_prefix + '_r_img_id_sent_id.jpg'))
    # joint_locations.append((0, 3*height))

    crop_img(ori_img_name, gd_ann_box, os.path.join(tmp_path, tmp_prefix + '_r_croped_region.jpg'))
    image_names.append(os.path.join(tmp_path, tmp_prefix + '_r_croped_region.jpg'))
    joint_locations.append((ox + width + 10, oy+height))

    sentence = entry['sent']
    erased_sent = entry['erased_sent']
    plt_text(width, height,
             " Select Region \n \n sentence: \n  %s \n erased_sent: \n  %s \n ground_truth score: % s " %
             (sentence, erased_sent, str(round(scores[gd_ix], 2))),
             os.path.join(tmp_path, tmp_prefix + '_r_sent_e_sent.jpg'))
    image_names.append(os.path.join(tmp_path, tmp_prefix + '_r_sent_e_sent.jpg'))
    joint_locations.append((ox + 2*width + 10, oy+height))

    weights = entry['weights'][gd_ix]

    draw_attention(os.path.join(tmp_path, tmp_prefix + '_r_croped_region.jpg'), entry['sub_grid_attn'][gd_ix],
                   os.path.join(tmp_path, tmp_prefix + '_r_img_sub.jpg'))
    image_names.append(os.path.join(tmp_path, tmp_prefix + '_r_img_sub.jpg'))
    joint_locations.append((ox + 3*width + 20, oy+height))

    sub_matching_score = entry['sub_matching_score'][gd_ix]
    sub_attn = entry['sub_attn'][gd_ix]
    plt_text(width, height, " Subject Module \n \n module_weight: %s, \n module_score: %s, \n"
                                      " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[0], 2)),
                                                                               str(round(sub_matching_score, 2)),
                                                                               sentence,
                                                                               json.dumps(
                                                                                   np.around(np.array(sub_attn),
                                                                                             2).tolist())),
             os.path.join(tmp_path, tmp_prefix + '_r_sen_sub.jpg'))
    image_names.append(os.path.join(tmp_path, tmp_prefix + '_r_sen_sub.jpg'))
    joint_locations.append((ox + 4*width + 20, oy+height))

    neighbor_bbox = entry['neighbor_bbox'][gd_ix]

    loc_r_box = np.array(entry['loc_r_box'])[gd_ix][np.argmax(neigh_attn[:, 1:][gd_ix])].tolist()
    draw_rec(ori_img_name, neighbor_bbox, neigh_attn[gd_ix][1:].tolist(), os.path.join(tmp_path, tmp_prefix + '_r_img_loc.jpg'),
             ref_box=loc_r_box,
             gt_box=gd_ann_box, if_nor=False, gd_tag=neigh_attn[gd_ix][0])
    image_names.append(os.path.join(tmp_path, tmp_prefix + '_r_img_loc.jpg'))
    joint_locations.append((ox + 30 + 5*width, oy+height))

    loc_attn = entry['loc_attn'][gd_ix]
    loc_matching_score = entry['loc_matching_score'][gd_ix]
    plt_text(width, height, " Location Module \n \n module_weight: %s, \n module_score: %s, \n"
                            " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[1], 2)),
                                                                     str(round(loc_matching_score, 2)),
                                                                     sentence,
                                                                     json.dumps(np.around(np.array(loc_attn),
                                                                                          2).tolist())),
             os.path.join(tmp_path, tmp_prefix + '_r_sen_loc.jpg'))
    image_names.append(os.path.join(tmp_path, tmp_prefix + '_r_sen_loc.jpg'))
    joint_locations.append((ox + 30 + 6*width, oy+height))

    cxt_bboxes = entry['cxt_bboxes'][gd_ix]
    rel_r_box = np.array(entry['rel_r_box'])[gd_ix][np.argmax(rel_att[gd_ix])].tolist()
    draw_rec(ori_img_name, cxt_bboxes, rel_att[gd_ix].tolist(), os.path.join(tmp_path, tmp_prefix + '_r_img_rel.jpg'),
             ref_box=rel_r_box,
             gt_box=gd_ann_box, if_nor=False)
    image_names.append(os.path.join(tmp_path, tmp_prefix + '_r_img_rel.jpg'))
    joint_locations.append((ox + 7*width + 40, oy+height))

    rel_attn = entry['rel_attn'][gd_ix]
    rel_matching_score = entry['rel_matching_score'][gd_ix]
    plt_text(width, height, " Relation Module \n \n module_weight: %s, \n module_score: %s, \n"
                            " sentence:\n  %s \n sub_attn:\n  %s" % (str(round(weights[2], 2)),
                                                                     str(round(rel_matching_score, 2)),
                                                                     sentence,
                                                                     json.dumps(np.around(np.array(rel_attn),
                                                                                          2).tolist())),
             os.path.join(tmp_path, tmp_prefix + '_r_sen_rel.jpg'))
    image_names.append(os.path.join(tmp_path, tmp_prefix + '_r_sen_rel.jpg'))
    joint_locations.append((ox + 8*width + 40, oy+height))

    # joint_an_instance(image_names, joint_locations, res_name, 3*width + 50 + 2*width)

    return image_names, joint_locations

        # break


def plot_joint_test():
    img_path = "E:\data\VQA V2\\train2014"
    dest_path = "F:\wrong_images"
    data_path = "E:\code\python\\visual_grounding\MINE\\refCOCO"
    tmp_path = "E:\code\python\\visual_grounding\MINE\\tmp"
    info = json.load(
        open(os.path.join(data_path, "coco_ca_pretrain_all_info3_val.json"), 'r')
    )

    for entry in info:

        img_name = entry['file_name']
        sent_id = entry['sent_id']
        image_id = entry['image_id']

        if image_id == 83353 and sent_id == 121554:

            sent_id = entry['sent_id']
            image_id = entry['image_id']
            key_ = str(image_id) + '_' + str(sent_id)

            gd_ix = entry['gd_ix']
            pred_ix = entry['pred_ix']

            image_names = []
            joint_locations = []
            res_name = os.path.join(dest_path, str(image_id) + "_" + str(sent_id) + '.jpg')
            ori_img_name = os.path.join(img_path, img_name)
            print(res_name)

            ori_im = Image.open(ori_img_name, 'r')
            width = ori_im.size[0]
            height = ori_im.size[1]

            ann_boxes = entry['ann_bboxes']
            scores = entry['all_scores']
            gd_ann_box = entry['gd_ann_box']
            pred_ann_box = entry['pred_ann_box']
            draw_rec(ori_img_name, ann_boxes, scores, os.path.join(tmp_path, 'ori_img.jpg'), gt_box=gd_ann_box,
                     pred_box=pred_ann_box)
            image_names.append(os.path.join(tmp_path, 'ori_img.jpg'))
            joint_locations.append((0, 0))

            plt_text(width, height, " original info: \n image_id: %s \n sent_id: %s" % (str(image_id), str(sent_id)),
                     os.path.join(tmp_path, 'img_id_sent_id.jpg'))
            image_names.append(os.path.join(tmp_path, 'img_id_sent_id.jpg'))
            joint_locations.append((0, height))

            crop_img(ori_img_name, gd_ann_box, os.path.join(tmp_path, '_r_croped_region.jpg'))
            image_names.append(os.path.join(tmp_path, '_r_croped_region.jpg'))
            joint_locations.append((width + 10, height))

            new_matrix = np.array(entry['sub_grid_attn'][gd_ix]).reshape(7, 7).copy()
            new_matrix[3, 1] = np.array(entry['sub_grid_attn'][gd_ix]).reshape(7, 7)[6, 3]
            new_matrix[4, 2] = np.array(entry['sub_grid_attn'][gd_ix]).reshape(7, 7)[6, 2]

            new_matrix[6, 3] = np.array(entry['sub_grid_attn'][gd_ix]).reshape(7, 7)[4, 2]
            new_matrix[6, 2] = np.array(entry['sub_grid_attn'][gd_ix]).reshape(7, 7)[3, 1]

            draw_attention(os.path.join(tmp_path, '_r_croped_region.jpg'), new_matrix.reshape(49).tolist(),
                           os.path.join(tmp_path, '_r_img_sub.jpg'))
            image_names.append(os.path.join(tmp_path, '_r_img_sub.jpg'))
            joint_locations.append((3 * width + 20, height))

            break

        else:
            continue


plot_joint_test()

def plot_compare():
    data_path = "E:\code\python\\visual_grounding\MINE\\refCOCO"
    dest_path = "F:\wrong_images3"
    # info = json.load(
    #     open(os.path.join(data_path, "wrong_info3_val.json"), 'r')
    # )

    SOTA_info = json.load(
        open(os.path.join(data_path, "all_info3_val.json"), 'r')
    )

    OURS_info = json.load(
        open(os.path.join(data_path, "coco_ca_pretrain_all_info3_val.json"), 'r')
    )

    # OURS_info_dict = {}
    # for index, entry in enumerate(OURS_info):
    #     sent_id = entry['sent_id']
    #     image_id = entry['image_id']
    #     key_ = str(image_id) + '_' + str(sent_id)
    #     OURS_info_dict[key_] = entry

    with open(os.path.join(data_path, "dict.pkl"), 'rb') as file:
        # pickle.dump(OURS_info_dict, file=file)
        OURS_info_dict = pickle.load(file)

    img_path = "E:\data\VQA V2\\train2014"

    for number, entry_SOTA in enumerate(SOTA_info):
        if number < 10833:
            continue

        image_names, joint_locations = [], []

        sent_id = entry_SOTA['sent_id']
        image_id = entry_SOTA['image_id']
        key_ = str(image_id) + '_' + str(sent_id)

        gd_ix = entry_SOTA['gd_ix']
        pred_ix = entry_SOTA['pred_ix']
        if gd_ix == pred_ix:
            S1 = True
        else:
            S1 = False

        OURS_entry = OURS_info_dict[key_]
        gd_ix = OURS_entry['gd_ix']
        pred_ix = OURS_entry['pred_ix']

        if gd_ix == pred_ix:
            O1 = True
        else:
            O1 = False

        assert entry_SOTA['file_name'] == OURS_entry['file_name']
        img_name = entry_SOTA['file_name']

        ori_img_name = os.path.join(img_path, img_name)

        ori_im = Image.open(ori_img_name, 'r')
        width = ori_im.size[0]
        height = ori_im.size[1]

        # ############################
        if_wrong_SOTA = False
        if_wrong_OURS = False
        prfix = ""
        if S1:
            prfix += "S1"
        else:
            prfix += "S0"
            if_wrong_SOTA = True

        if O1:
            prfix += "O1"
        else:
            prfix += "O0"
            if_wrong_OURS = True

        names, locations = plot_jointw_r(entry_SOTA, 0, 0, "SOTA", if_wrong_SOTA)
        image_names += names
        joint_locations += locations

        names, locations = plot_jointw_r(OURS_entry, 0, 2 * height, "OURS", if_wrong_OURS)
        image_names += names
        joint_locations += locations

        res_name = os.path.join(dest_path, prfix, key_ + '.jpg')
        joint_an_instance(image_names, joint_locations, res_name, 9 * width + 50)
        print(number, res_name)
        # break


