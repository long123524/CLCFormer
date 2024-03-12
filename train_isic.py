from torch.autograd import Variable
import argparse
from datetime import datetime
from lib.clcformer_model import CLCFormer
from utils.dataloader import *
from utils.utils import *
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import random
from optimezer_looka import Lookahead


def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

#multi-class loss
# def structure_loss(pred, mask):
#     ce_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=6)
#     wbce = ce_loss(pred,mask.squeeze(1).long())
#     dice_loss = DiceLoss(6)
#     dice = dice_loss(pred, mask, softmax=True)
#
#     return wbce+dice


def train(train_loader, model, optimizer, epoch, best_iou):
    model.train()
    loss_record2, loss_record3, loss_record4,loss_record1 = AvgMeter(), AvgMeter(), AvgMeter(),AvgMeter()
    accum = 0
    for i, (img_file_name,inputs,pack) in enumerate(tqdm(train_loader)):
        # ---- data prepare ----
        images, gts = inputs,pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda().float()


        # ---- forward ----
        lateral_map_4, lateral_map_3, lateral_map_2,lateral_map_1 = model(images)

        # ---- loss function ----
        loss4 = structure_loss(lateral_map_4, gts)
        loss3 = structure_loss(lateral_map_3, gts)
        loss2 = structure_loss(lateral_map_2, gts)

        loss1 = structure_loss(lateral_map_1, gts)

        loss = loss1*0.4 + 0.3 * loss2 + 0.15 * loss3 + 0.15 * loss4


        # ---- backward ----
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # ---- recording loss ----
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record4.update(loss4.data, opt.batchsize)

        loss_record1.update(loss1.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show(),loss_record1.show()))

    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 1 == 0:
        meanIOU = test(model, test_loader)
        if meanIOU > best_iou:
            print('new best iou: ', meanIOU)
            best_iou = meanIOU
            torch.save(model.state_dict(), save_path + 'CLCFormer-%d.pth' % epoch)
            print('[Saving Snapshot:]', save_path + 'CLCFormer-%d.pth'% epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'CLCFormer-%d.pth' % epoch)
            print('[Saving Snapshot:]', save_path + 'CLCFormer-%d.pth'% epoch)
    return best_iou


def test(model, test_data):

    model.eval()
    mean_loss = []

    mean_iou = []

    dice_bank = []
    iou_bank = []
    loss_bank = []
    acc_bank = []

    for i, (img_file_name,inputs,pack) in enumerate(tqdm(test_data)):
        image, gt = inputs,pack
        image = image.cuda()
        gt = gt.cuda().float()

        with torch.no_grad():
            _, _, _, res = model(image)
        loss = structure_loss(res, gt)

        res = res.sigmoid().data.cpu().numpy().squeeze()
        gt = gt.detach().cpu().numpy().squeeze()
        gt = 1*(gt>0.5)
        res = 1*(res > 0.5)

        dice = mean_dice_np(gt, res)
        iou = mean_iou_np(gt, res)
        TP = float((res * gt).sum())
        FP = float((res * (1 - gt)).sum())
        FN = float(((1 - res) * (gt)).sum())
        TN = float(((1 - res) * (1 - gt)).sum())
        acc = (TP + TN) / (TP + FP + FN + TN)
        # acc = np.sum(res == gt) / (res.shape[0]*res.shape[1])

        loss_bank.append(loss.item())
        dice_bank.append(dice)
        iou_bank.append(iou)
        acc_bank.append(acc)

    print('{} Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}'.
        format('test', np.mean(loss_bank), np.mean(dice_bank), np.mean(iou_bank), np.mean(acc_bank)))

    mean_iou.append(np.mean(iou_bank))

    return mean_iou[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  ###
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='./WHU_bulding/train', help='path to train dataset')
    parser.add_argument('--valid_path', type=str,
                        default='./WHU_bulding/val/image', help='path to valid dataset')
    parser.add_argument('--train_save', type=str, default='./save_model')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    opt = parser.parse_args()

    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    # ---- build models ----
    model = CLCFormer(pretrained=True).cuda()

    params = model.parameters()
    base_optimizer = torch.optim.AdamW(params, opt.lr, betas=(opt.beta1, opt.beta2))
    optimizer = Lookahead(base_optimizer)
     
    image_root = '{}/image'.format(opt.train_path)

    train_loader = get_loader(image_root, batchsize=opt.batchsize)

    test_loader = get_loader(opt.valid_path, batchsize=opt.batchsize)

    total_step = len(train_loader)

    print("#"*20, "Start Training", "#"*20)

    best_iou = 1e-5
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.epoch, eta_min=1e-5)

    for epoch in range(1, opt.epoch + 1):
        best_loss = train(train_loader, model, optimizer, epoch, best_iou)
        scheduler.step()
