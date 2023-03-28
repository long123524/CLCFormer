import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils.dataloader import *
import glob
from lib.clcformer_model import CLCFormer
from tqdm import tqdm
import numpy as np
import cv2
from scipy import stats
from torch.autograd import Variable
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    if dim_input > expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input < expected_dims:
        np_output = np.expand_dims(np_input, 0)
    assert len(np_output.shape) == expected_dims
    return np_output


def binary_accuracy(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = (pred >= 0.5)
    label = (label >= 0.5)

    TP = float((pred * label).sum())
    FP = float((pred * (1 - label)).sum())
    FN = float(((1 - pred) * (label)).sum())
    TN = float(((1 - pred) * (1 - label)).sum())
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    IoU = TP / (TP + FP + FN + 1e-10)
    acc = (TP + TN) / (TP + FP + FN + TN)
    F1 = 0
    if acc > 0.99 and TP == 0:
        precision = 1
        recall = 1
        IoU = 1
    if precision > 0 and recall > 0:
        F1 = stats.hmean([precision, recall])
    return acc, precision, recall, F1, IoU


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


if __name__ == "__main__":

    seed_everything(1234)
    valpath = './WHU_bulding/test/image'
    val_path = os.path.join(valpath, "*.tif")
    model_file = './snapshots/save_model/13_baseline_9123.pth'
    save_path = './WHU_bulding/output'

    f = open('./WHU_bulding/accuracy.txt', 'w+')

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = CLCFormer().cuda()
    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))    #报错可以加False
    model.eval()


    val_file_names = glob.glob(val_path)
    test_loader = get_loader(valpath, batchsize=1)

    if not os.path.exists(save_path):
        os.mkdir(save_path)


    acc_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()

    total_iter = len(val_file_names)
    with torch.no_grad():
        for i, (img_file_name,inputs,pack) in enumerate(
                tqdm(test_loader)
        ):
            images, gts = inputs, pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            raw_predictions = model(images)
            # outputs4, outputs5, _, outputs6 = model(torch.flip(images, [-1]))
            # predict_2 = torch.flip(outputs6, [-1])
            # outputs7, outputs8, _, outputs9 = model(torch.flip(images, [-2]))
            # predict_3 = torch.flip(outputs9, [-2])

            # raw_predictions = torch.sigmoid(raw_predictions)
            # predict_2 = torch.sigmoid(predict_2)
            # predict_3 = torch.sigmoid(predict_3)

            pred = raw_predictions

            outputs1 = pred.detach().cpu().numpy().squeeze()
            targets1 = gts.detach().cpu().numpy().squeeze()

            res = np.zeros((512, 512))
            res[outputs1>0.5] = 255
            res[outputs1<=0.5] = 0
          #  res = morphology.remove_small_objects(res.astype(int), 800)
            acc, precision, recall, F1, IoU = binary_accuracy(res, targets1)

            acc_meter.update(acc)
            precision_meter.update(precision)
            recall_meter.update(recall)
            F1_meter.update(F1)
            IoU_meter.update(IoU)

            res = np.array(res, dtype='uint8')

            output_path = os.path.join(
                save_path,  os.path.basename(img_file_name[0])
            )
            cv2.imwrite(output_path, res)


            f.write('Eval num %d/%d, Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, IoU %.2f\n' % (
                i, total_iter, acc * 100, precision * 100, recall * 100, F1 * 100, IoU * 100))

        print('avg Acc %.2f, Pre %.2f, Recall %.2f, F1 %.2f, IOU %.2f' % (
            acc_meter.avg * 100, precision_meter.avg * 100, recall_meter.avg * 100, F1_meter.avg * 100,
            IoU_meter.avg * 100))
