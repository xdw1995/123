import os

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import random
from skimage import exposure
import sys
sys.path.append('/data/xudw/SlowFastNetworks-master/SlowFastNetworks-master/libb/')
import zhongjichuli_xdw
class Normalize:
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

normalize = Normalize(mean=mean, std=std)
class VideoDataset(Dataset):

    def __init__(self, mode='train', clip_len=64):
        self.clip_len = clip_len
        self.short_side = [128, 160]
        self.crop_size = 112
        # self.short_side = [180, 200]
        # self.crop_size = 168
        # self.short_side = [256, 320]
        # self.crop_size = 224
        self.mode = mode
        self.fnames, self.labels = [], []  # 一个视频名字对应一个label
        video_path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/datas/'

        if mode == 'train':
            path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/annotations/train_val.txt'
        if mode == 'validation':
            path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/annotations/validation.txt'
        if self.mode != 'test':
            with open(path, 'r') as f:
                for i in f.readlines():
                    # print(video_path+'/'+i.split(' ')[0])
                    # print(i.split(" ")[1].strip())
                    name = video_path + i.split(' ')[0].strip().replace('.avi','')
                    if len(os.listdir(name))<self.clip_len:
                        print(i.split(' ')[0].strip())
                        continue
                    self.fnames.append(name)
                    self.labels.append(int(i.split(" ")[1].strip()) - 1)
        print('xdw')
        if self.mode == 'test':
            path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/annotations/mod-ucf101-test.txt'
            with open(path, 'r') as f:
                for i in f.readlines():
                    # for _ in range(10):#todo
                    self.fnames.append(video_path + '/' + i.split(' ')[0].strip())
                    # self.labels.append(int(i.split(" ")[1].strip())-1)
        # print(self.fnames)
        # print(self.labels)
        # print(self.fnames)

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        # buffer = self.loadvideo(self.fnames[index])

        p = self.fnames[index]
        fnames = list(map(lambda x:p+'/'+x,sorted(os.listdir(p))))
        time_index = np.random.randint(len(fnames) - self.clip_len)
        fnames = np.array(fnames)
        fnames = fnames[time_index:time_index+self.clip_len]
        if self.mode =='train':
            # if np.random.random() < 0.5:
            out = self.crop(fnames, chidu=1.2, mode='randomcrop', flip=True, flipud=False)
            # else:
            #     out = self.crop(fnames, chidu=0.8, mode='randomcrop', flip=True, flipud=False)
        else:

            out = self.crop(fnames, chidu=1.2, mode='centercrop', flip=False, flipud=False,blight = False , doHLS= False)

        # for c , aaa in enumerate(out):
        #     cv2.imwrite(str(index)+'.jpg',aaa)
        #     break

        # if np.random.random() < 0.5:
        #     out = out[::-1].copy()

        out = torch.from_numpy(out.transpose((3, 0, 1, 2))).float() / 255.0
        return normalize(out), torch.from_numpy(np.array(self.labels[index])).long()
    def HLS(self, frame ,random_vars):
        f = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
        hls_limits = [180, 255, 255]
        for i in range(3):
            f[:, :, i] = np.minimum(np.maximum(f[:, :, i] + random_vars[i], 0), hls_limits[i])
        f = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_HLS2RGB)
        return f

    def crop(self,fnames, chidu=1.01, mode='centercrop', flip=True, flipud=False,blight = False, doHLS = True):
        crop_size = self.crop_size
        random_vars = [int(round(random.uniform(-x, x))) for x in [15, 25, 25]]
        flip_time = False
        RD = np.random.random()
        if  RD < 0.4:
            flip_time = True
        height, width = cv2.imread(fnames[0]).shape[:2]
        buffer = []
        if (width > height):
            if chidu > 1:
                scale = float(crop_size) / float(height)
            else:
                scale = float(crop_size) / float(width)
            scale = random.uniform(scale, scale * chidu)
            for frame_name in fnames:
                frame = cv2.imread(frame_name)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if chidu > 1:
                    frame = np.array(
                        cv2.resize(np.array(frame), (int(width * scale + 1), int(height * scale + 1)))).astype(
                        np.float32)
                else:
                    frame = np.array(cv2.resize(np.array(frame), (int(width * scale), int(height * scale)))).astype(
                        np.float32)
                if flip and flip_time:
                    frame = cv2.flip(frame, flipCode=1)
                if flipud and flip_time:
                    frame = np.flipud(frame)
                    frame = np.ascontiguousarray(frame)
                # if blight:
                #     frame = exposure.adjust_gamma(frame, RD)
                if doHLS:
                    frame = self.HLS(frame, random_vars)
                buffer.append(frame)
        else:
            if chidu > 1:
                scale = float(crop_size) / float(width)
            else:
                scale = float(crop_size) / float(height)
            scale = random.uniform(scale, scale * chidu)
            for frame_name in fnames:
                frame = cv2.imread(frame_name)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if chidu > 1:
                    frame = np.array(
                        cv2.resize(np.array(frame), (int(width * scale + 1), int(height * scale + 1)))).astype(
                        np.float32)
                else:
                    frame = np.array(cv2.resize(np.array(frame), (int(width * scale), int(height * scale)))).astype(
                        np.float32)
                if flip and flip_time:
                    frame = cv2.flip(frame, flipCode=1)
                if flipud and flip_time:
                    frame = np.flipud(frame)
                    frame = np.ascontiguousarray(frame)
                # if blight:
                #     frame = exposure.adjust_gamma(frame, RD)
                if doHLS:
                    frame = self.HLS(frame, random_vars)
                buffer.append(frame)
        # print(scale)

        buffer = np.array(buffer)
        if chidu > 1:
            if mode == 'centercrop':
                crop_x = int((frame.shape[0] - crop_size) / 2)
                crop_y = int((frame.shape[1] - crop_size) / 2)
                buffer = buffer[:, crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
            elif mode == 'randomcrop':
                crop_x = np.random.randint(frame.shape[0] - crop_size)
                crop_y = np.random.randint(frame.shape[1] - crop_size)
                buffer = buffer[:, crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        elif chidu < 1:
            buffers = np.zeros((buffer.shape[0], crop_size, crop_size, 3))
            crop_x = np.random.randint(crop_size - buffer.shape[1])
            crop_y = np.random.randint(crop_size - buffer.shape[2])
            buffers[:, crop_x:crop_x + buffer.shape[1], crop_y:crop_y + buffer.shape[2], :] = buffer
            buffer = buffers

        return buffer

    def __len__(self):
        return len(self.fnames)


class VideoDataset2(Dataset):

    def __init__(self, mode='train', clip_len=64, frame_sample_rate=1):
        self.clip_len = clip_len
        self.short_side = [128, 160]
        self.crop_size = 112
        # self.short_side = [180, 200]
        # self.crop_size = 168
        # self.short_side = [256, 320]
        # self.crop_size = 224
        self.frame_sample_rate = frame_sample_rate
        self.mode = mode
        self.fnames, self.labels = [], []  # 一个视频名字对应一个label
        video_path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/datas/'

        if mode == 'train':
            path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/annotations/train_val.txt'
        if mode == 'validation':
            path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/annotations/test.txt'
        if self.mode != 'test':
            with open(path, 'r') as f:
                for i in f.readlines():
                    # print(video_path+'/'+i.split(' ')[0])
                    # print(i.split(" ")[1].strip())
                    name = video_path + i.split(' ')[0].strip().replace('.avi', '')
                    if len(os.listdir(name)) < self.clip_len:
                        print(i.split(' ')[0].strip())
                        continue
                    self.fnames.append(name)
                    self.labels.append(int(i.split(" ")[1].strip()) - 1)
        # print('xdw')

        if self.mode == 'test':
            path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/annotations/mod-ucf101-test.txt'
            with open(path, 'r') as f:
                for i in f.readlines():
                    # for _ in range(10):#todo
                    self.fnames.append(video_path + '/' + i.split(' ')[0].strip())
                    # self.labels.append(int(i.split(" ")[1].strip())-1)


    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        p = self.fnames[index]
        fnames = list(map(lambda x: p + '/' + x, sorted(os.listdir(p))))
        time_index = np.random.randint(len(fnames) - self.clip_len)
        fnames = np.array(fnames)
        fnames = fnames[time_index:time_index + self.clip_len]

        buffer = self.loadvideo(fnames)
        if np.random.random()<0.5:
            buffer = buffer[::-1]
        if self.mode == 'train':
            buffer = self.randomflip(buffer)

        # size =(112,112)
        # if size要改，crop函数里面的随机裁剪也要跟着改
        # buffer = self.crop(buffer, self.clip_len, self.crop_size)

        buffer = np.concatenate(buffer, axis=2)
        # print(buffer.shape)
        # import pdb
        # pdb.set_trace()
        # ________________________________________debug
        # b = video_transform(buffer)
        # b= b.numpy().transpose((1,2,3,0))
        # for i in b:
        #     print(i)
        if self.mode == 'validation':
            return zhongjichuli_xdw.test_transform(buffer), torch.from_numpy(np.array(self.labels[index])).long()
        elif self.mode == 'test':
            return zhongjichuli_xdw.test_transform(buffer)
        else:
            # print(video_transform(buffer).shape,"123")
            return zhongjichuli_xdw.video_transform(buffer), torch.from_numpy(np.array(self.labels[index])).long()

    def loadvideo(self, fname):
        buffer = []
        for i in fname:
            i = cv2.imread(i)
            frame = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            if i is None:
                frame = buffer[-1]
            buffer.append(frame)
        return np.array(buffer)


    # def crop(self, buffer, clip_len, crop_size):
    #     # randomly select time index for temporal jittering
    #     # time_index = np.random.randint(buffer.shape[0] - clip_len)
    #     # Randomly select start indices in order to crop the video
    #     height_index = np.random.randint(buffer.shape[1] - crop_size)
    #     width_index = np.random.randint(buffer.shape[2] - crop_size)
    #
    #     # crop and jitter the video using indexing. The spatial crop is performed on
    #     # the entire array, so each frame is cropped in the same location. The temporal
    #     # jitter takes place via the selection of consecutive frames
    #     buffer = buffer[:,
    #              height_index:height_index + crop_size,
    #              width_index:width_index + crop_size, :]
    #
    #     return buffer

    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]])) / 128.0
            buffer[i] = frame
        return buffer

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)
        return buffer

    def __len__(self):
        return len(self.fnames)

class VideoDataset_RDBdiff(Dataset):

    def __init__(self, mode='train', clip_len=65, frame_sample_rate=1):
        self.clip_len = 65
        self.short_side = [128, 160]
        self.crop_size = 112

        self.frame_sample_rate = frame_sample_rate
        self.mode = mode
        self.fnames, self.labels = [], []  # 一个视频名字对应一个label
        video_path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/datas/'

        if mode == 'train':
            path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/annotations/train_val.txt'
        if mode == 'validation':
            path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/annotations/test.txt'
        if self.mode != 'test':
            with open(path, 'r') as f:
                for i in f.readlines():
                    # print(video_path+'/'+i.split(' ')[0])
                    # print(i.split(" ")[1].strip())
                    name = video_path + i.split(' ')[0].strip().replace('.avi', '')
                    if len(os.listdir(name)) < self.clip_len:
                        print(i.split(' ')[0].strip())
                        continue
                    self.fnames.append(name)
                    self.labels.append(int(i.split(" ")[1].strip()) - 1)
        # print('xdw')

        if self.mode == 'test':
            path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/annotations/mod-ucf101-test.txt'
            with open(path, 'r') as f:
                for i in f.readlines():
                    # for _ in range(10):#todo
                    self.fnames.append(video_path + '/' + i.split(' ')[0].strip())
                    # self.labels.append(int(i.split(" ")[1].strip())-1)
        # print(self.fnames)
        # print(self.labels)
        # print(self.fnames)

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        p = self.fnames[index]
        if len(os.listdir(p))-1<self.clip_len:
            print(len(os.listdir(p)))
            p  = self.fnames[100]
        fnames = list(map(lambda x: p + '/' + x, sorted(os.listdir(p))))
        time_index = np.random.randint(len(fnames) - self.clip_len)
        fnames = np.array(fnames)
        fnames = fnames[time_index:time_index + self.clip_len]

        buffer = self.loadvideo(fnames)
        if np.random.random() < 0.5:
            buffer = buffer[::-1]
        if self.mode == 'train':
            buffer = self.randomflip(buffer)

        buffer = np.concatenate(buffer, axis=2)
        # ________________________________________debug
        # b = video_transform(buffer)
        # b= b.numpy().transpose((1,2,3,0))
        # for i in b:
        #     print(i)
        if self.mode == 'validation':
            buffer = zhongjichuli_xdw.test_transform(buffer)
            label = torch.from_numpy(np.array(self.labels[index])).long()
            return self.diff(buffer), label
        elif self.mode == 'test':
            return self.diff(zhongjichuli_xdw.test_transform(buffer))
        else:
            buffer = zhongjichuli_xdw.video_transform(buffer)
            label = torch.from_numpy(np.array(self.labels[index])).long()

            # ________________________________________debug
            #
            # b = self.diff(buffer)
            # b= b.numpy().transpose((1,2,3,0))
            # for i in b:
            #     cv2.imshow('1',i)
            #     cv2.waitKey(10)
            return self.diff(buffer), label

    def diff(self, c):
        new_data = c[:, 1:, :, :].clone()
        for x in reversed(list(range(1, 64 + 1))):
            new_data[:, x - 1, :, :] = c[:, x, :, :] - c[:, x - 1, :, :]
        return new_data

    def loadvideo(self, fname):
        buffer = []
        for i in fname:
            i = cv2.imread(i)
            frame = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            if i is None:
                frame = buffer[-1]
            buffer.append(frame)
        return np.array(buffer)

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 :,
                 :, :]

        return buffer

    def normalize(self, buffer):
        # Normalize the buffer
        # buffer = (buffer - 128)/128.0
        for i, frame in enumerate(buffer):
            frame = (frame - np.array([[[128.0, 128.0, 128.0]]])) / 128.0
            buffer[i] = frame
        return buffer

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)
        return buffer

    def __len__(self):
        return len(self.fnames)

if __name__ == '__main__':

    datapath = '/disk/data/UCF-101'
    # print(len(VideoDataset( mode='validation',clip_len=5)))
    # print(len(VideoDataset( mode='train',clip_len=5)))
    train_dataloader = \
        DataLoader(VideoDataset2(mode='validation'), batch_size=32, shuffle=False, num_workers=0)
    # test = \
    #     DataLoader(VideoDataset(mode='validation'), batch_size=5, shuffle=False, num_workers=0)
    # print(len(train_dataloader))
    # print(len(test))

    for step, (buffer, label) in enumerate(train_dataloader):
        print(buffer.shape)
        # if step==1:
        #     break
        # print("label: ", label)
    # test_dataloader = \
    #     DataLoader( VideoDataset_RDBdiff( mode='test',clip_len=64), batch_size=1, shuffle=False, num_workers=0)
    # for i in train_dataloader:
    #     print(i[0].shape)
    #     print(i[1])