import os
import cv2
import tqdm
import numpy as np

ImSurf = np.array([255, 255, 255])  # label 0
Building = np.array([255, 0, 0]) # label 1
LowVeg = np.array([255, 255, 0]) # label 2
Tree = np.array([0, 255, 0]) # label 3
Car = np.array([0, 255, 255]) # label 4
Clutter = np.array([0, 0, 255]) # label 5

def rgb_to_2D_label(_label):
    _label = _label.transpose(2, 0, 1)
    label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)
    label_seg[np.all(_label.transpose([1, 2, 0]) == ImSurf, axis=-1)] = 0
    label_seg[np.all(_label.transpose([1, 2, 0]) == Building, axis=-1)] = 1
    label_seg[np.all(_label.transpose([1, 2, 0]) == LowVeg, axis=-1)] = 2
    label_seg[np.all(_label.transpose([1, 2, 0]) == Tree, axis=-1)] = 3
    label_seg[np.all(_label.transpose([1, 2, 0]) == Car, axis=-1)] = 4
    label_seg[np.all(_label.transpose([1, 2, 0]) == Clutter, axis=-1)] = 5
    # label_seg[np.all(_label.transpose([1, 2, 0]) == Boundary, axis=-1)] = 6
    return label_seg

class Vaihingen:
    def __init__(self, dataset_path, target_path):
        self.dataset_path = dataset_path
        self.target_path = target_path
        self.RGB_path = os.path.join(dataset_path, 'image')
        self.Label_path = os.path.join(dataset_path, 'mask')
        self.file_flag = os.listdir(self.Label_path)

    def start_dealWith(self, split_size):
        num = 0
        tqdm_flag = tqdm.tqdm(self.file_flag, total=len(self.file_flag))
        for file in tqdm_flag:
            # 进行数据的读取
            image = cv2.imread(os.path.join(self.RGB_path, file))
            label = cv2.imread(os.path.join(self.Label_path, file))
            # 将像素值进行对应的转换
            label = rgb_to_2D_label(label)
            # label[label == 255] = 1
            # label = label[:, :, 2] * 4 + label[:, :, 1] * 2 + label[:, :, 0]
            # label = label - 1
            # label[label == 5] = 4
            # label[label == 6] = 5
            # 开始进行切割
            min_x = min(image.shape[0],  label.shape[0])
            min_y = min(image.shape[1],  label.shape[1])
            range_x = min_x // split_size
            range_y = min_y // split_size
            for x in range(range_x):
                for y in range(range_y):
                    split_image = image[x * split_size:(x + 1) * split_size, y * split_size:(y + 1) * split_size]
                    split_label = label[x * split_size:(x + 1) * split_size, y * split_size:(y + 1) * split_size]
                    cv2.imwrite(os.path.join(self.target_path, 'image', str(num) + '.png'), split_image)
                    cv2.imwrite(os.path.join(self.target_path, 'mask', str(num) + '.png'), split_label)
                    num += 1
        tqdm_flag.close()

if __name__ == '__main__':
    v = Vaihingen(dataset_path=r'D:\Vaihingen\Vaihingen\ISPRS_semantic_labeling_Vaihingen\test', target_path=r'D:\Vaihingen\Vaihingen\ISPRS_semantic_labeling_Vaihingen\test_post')
    v.start_dealWith(split_size=256)
