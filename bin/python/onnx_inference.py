import argparse
import os
import cv2
import numpy as np
import onnxruntime
from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import vis
from yolox.utils import nms

def visual(output, img_info, cls_conf=0.35):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if output is None:
        return img
    bboxes, clses, scores, masks = output
    bboxes /= ratio
    masks = masks.astype(np.uint8)
    mask_dsize = (max(img.shape[0:2]), ) * 2
    if mask_dsize[0] != masks.shape[1]:
        masks = cv2.resize(
            masks.transpose(1, 2, 0), dsize=mask_dsize, 
            interpolation=cv2.INTER_LINEAR
        )
        if masks.ndim == 2:
            masks = masks[None]
        else:
            masks = masks.transpose(2, 0, 1)

    masks = masks[:, : img.shape[0], :img.shape[1]]
    vis_res = vis(img, bboxes, scores, clses, masks, cls_conf, COCO_CLASSES)
    return vis_res


if __name__ == "__main__":
    input_shape = (640, 640)
    # image_path = "/disk2/rcf/coco/test2017/000000000001.jpg"
    image_path = "/home/r/Desktop/Scripts/Detection/bin/images/toliet.jpg"
    save_path = "/home/r/Desktop/Scripts/Detection/bin/images/toliet_results.jpg"
    model_path = "/home/r/Desktop/Scripts/Detection/bin/models/yolox_tensorrt.onnx"
    conf_thre = 0.3
    mask_thre = 0.5
    origin_img = cv2.imread(image_path)
    img, ratio = preprocess(origin_img, input_shape)
    img_info = dict(ratio=ratio, raw_img=origin_img.copy())
    session = onnxruntime.InferenceSession(model_path)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    bboxes, masks = output[0], output[1].squeeze(1)
    conf_filter = (bboxes[:, 4] >= conf_thre)
    bboxes = bboxes[conf_filter]
    masks = masks[conf_filter]
    masks = masks > mask_thre
    scores = bboxes[:, 4]
    clses = bboxes[:, 5]
    bboxes = bboxes[:, :4]
    # keep_list = nms(bboxes, scores, 0.5)
    # bboxes = bboxes[keep_list]
    # scores = scores[keep_list]
    # clses = clses[keep_list]
    # masks = masks[keep_list]
    result_image = visual((bboxes, clses, scores, masks), img_info, conf_thre)
    cv2.imwrite(save_path, result_image)

