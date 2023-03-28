# import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import os
from osgeo import gdal
import os
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
Image.MAX_IMAGE_PIXELS = None
from lib.clcformer_model import CLCFormer
from utils.dataloader import get_loader
from torch.autograd import Variable


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def Pixel_Accuracy_Class(self):
        #         TP                                  TP+FP
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def label_to_rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 0]
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label_to_rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-o", "--output_path", default='./WHU_predict', help="Path where to save resulting masks.")
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images",default=False, action='store_true')
    return parser.parse_args()


def main():
    seed_everything(1234)
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = CLCFormer().cuda()
    model.load_state_dict(torch.load('./snapshots/save_model/CLCFormer-25.pth', map_location=torch.device(device)))  # 报错可以加False
    evaluator = Evaluator(num_class=2)
    evaluator.reset()
    model.eval()
    # if args.tta == "lr":
    #     transforms = tta.Compose(
    #         [
    #             tta.HorizontalFlip(),
    #             tta.VerticalFlip()
    #         ]
    #     )
    #     model = tta.SegmentationTTAWrapper(model, transforms)
    # elif args.tta == "d4":
    #     transforms = tta.Compose(
    #         [
    #             tta.HorizontalFlip(),
    #             tta.VerticalFlip(),
    #             tta.Rotate90(angles=[90]),
    #             tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
    #         ]
    #     )
    #     model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = get_loader('./WHU_bulding/test/image/', batchsize=1)

    with torch.no_grad():

        results = []
        for i, (img_file_name,inputs,pack) in enumerate(tqdm(test_dataset)):
            # raw_prediction NxCxHxW
            images, gts = inputs, pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            _,_,_,raw_predictions = model(images)
            outputs4, outputs5, _,outputs6 = model(torch.flip(images, [-1]))
            predict_2 = torch.flip(outputs6, [-1])
            outputs7, outputs8, _,outputs9 = model(torch.flip(images, [-2]))
            predict_3 = torch.flip(outputs9, [-2])



            image_ids = (os.path.basename(img_file_name[0]))[:-4]
            masks_true = gts.cpu().numpy().squeeze(0)
            masks_true[masks_true>0]= 1
            masks_true[masks_true <= 0] = 0

            raw_predictions = torch.sigmoid(raw_predictions)
            predict_2 = torch.sigmoid(predict_2)
            predict_3 = torch.sigmoid(predict_3)

            predictions = (raw_predictions +predict_2 + predict_3)/3.
            predictions[predictions>0.5] = 1
            predictions[predictions<=0.5] = 0


            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy().squeeze()
                mask = mask.astype(np.int64)
                evaluator.add_batch(pre_image=mask, gt_image=masks_true[i])
                results.append((mask,str(args.output_path + image_ids), args.rgb))
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))
    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    precision = evaluator.Precision()
    recall = evaluator.Recall()
    Classes = ('Background', 'Building')
    for class_name, class_iou, class_f1 in zip(Classes, iou_per_class, f1_per_class):
        print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
    print('F1:{}, mIOU:{}, OA:{}, P:{}, R:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA,
                                                     np.nanmean(precision[:-1]), np.nanmean(recall[:-1])))


if __name__ == "__main__":
    main()