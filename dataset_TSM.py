import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import sys

sys.path.append("../processing")
from zhongjichuli import video_transform, test_transform

import os
from pathlib import Path



class VideoDataset(Dataset):

    def __init__(self, mode='train', clip_len=64, frame_sample_rate=1):
        self.clip_len = clip_len
        self.short_side = [128, 160]
        self.crop_size = 112
        # self.short_side = [256, 320]
        # self.crop_size = 224
        self.frame_sample_rate = frame_sample_rate
        self.mode = mode
        self.fnames, self.labels = [], []  # 一个视频名字对应一个label
        video_path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/videos'
        # path = '/data/xudw/test_cpu/data_new/' + mode + '.txt'
        path = '/data/xudw/SlowFastNetworks-master/SlowFastNetworks-master/second_operation/data_new/' + mode + '.txt'


        if self.mode != 'test':
            with open(path, 'r') as f:
                for i in f.readlines():
                    # print(video_path+'/'+i.split(' ')[0])
                    # print(i.split(" ")[1].strip())
                    self.fnames.append(video_path + '/' + i.split(' ')[0].strip())
                    if self.mode != 'test':
                        self.labels.append(int(i.split(" ")[1].strip()) - 1)

        if self.mode == 'test':
            path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/annotations/mod-ucf101-test.txt'
            with open(path, 'r') as f:
                for i in f.readlines():
                    # for _ in range(10):
                    self.fnames.append(video_path + '/' + i.split(' ')[0].strip())
                    # self.labels.append(int(i.split(" ")[1].strip())-1)
        # print(self.fnames)
        # print(self.labels)
        # print(self.fnames)

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        buffer = self.loadvideo(self.fnames[index])

        while buffer.shape[0] < self.clip_len + 2:
            index = np.random.randint(self.__len__())
            buffer = self.loadvideo(self.fnames[index])

        if self.mode == 'train':
            buffer = self.randomflip(buffer)

        # size =(112,112)
        # if size要改，crop函数里面的随机裁剪也要跟着改
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        temp = []
        for i in buffer:
            # cv2.imshow("1",i)
            # cv2.waitKey(10)
            temp.append(i)
        buffer = np.concatenate(temp, axis=2)
        # ________________________________________debug
        # b = video_transform(buffer)
        # b= b.numpy().transpose((1,2,3,0))
        # for i in b:
        #     print(i)
        if self.mode == 'validation':
            return test_transform(buffer), torch.from_numpy(np.array(self.labels[index])).long()
        elif self.mode == 'test':
            return test_transform(buffer)
        else:
            # print(video_transform(buffer).shape,"123")
            return video_transform(buffer), torch.from_numpy(np.array(self.labels[index])).long()

    def loadvideo(self, fname):
        remainder = np.random.randint(self.frame_sample_rate)
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_height < frame_width:
            resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_width = int(float(resize_height) / frame_height * frame_width)
        else:
            resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_height = int(float(resize_width) / frame_width * frame_height)

        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = frame_count - 1
        frame_count_sample = frame_count // self.frame_sample_rate - 1
        if frame_count > 300:
            end_idx = np.random.randint(300, frame_count)
            start_idx = end_idx - 300
            frame_count_sample = 301 // self.frame_sample_rate - 1
        # buffer = np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))
        buffer = []
        count = 0
        retaining = True
        sample_count = 0

        # read in each frame, one at a time into the numpy buffer array
        while (count <= end_idx and retaining):
            retaining, frame = capture.read()
            if count < start_idx:
                count += 1
                continue
            if retaining is False or count > end_idx:
                break
            if count % self.frame_sample_rate == remainder and sample_count < frame_count_sample:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # will resize frames if not already final size

                if (frame_height != resize_height) or (frame_width != resize_width):
                    frame = cv2.resize(frame, (resize_width, resize_height))
                buffer.append(frame)
                sample_count = sample_count + 1
            count += 1
        capture.release()

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
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

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


class VideoDataset_RDBdiff(Dataset):

    def __init__(self, mode='train', clip_len=65, frame_sample_rate=1):
        self.clip_len = clip_len
        self.short_side = [128, 160]
        self.crop_size = 112

        self.frame_sample_rate = frame_sample_rate
        self.mode = mode
        self.fnames, self.labels = [], []  # 一个视频名字对应一个label
        video_path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/videos'
        path = '/data/xudw/test_cpu/data_new/' + mode + '.txt'
        if self.mode != 'test':
            with open(path, 'r') as f:
                for i in f.readlines():
                    # print(video_path+'/'+i.split(' ')[0])
                    # print(i.split(" ")[1].strip())
                    self.fnames.append(video_path + '/' + i.split(' ')[0].strip())
                    if self.mode != 'test':
                        self.labels.append(int(i.split(" ")[1].strip()) - 1)

        if self.mode == 'test':
            path = '/data/xudw/test_cpu/xdw_baseline/data/mod-ucf101/annotations/mod-ucf101-test.txt'
            with open(path, 'r') as f:
                for i in f.readlines():
                    for _ in range(10):
                        self.fnames.append(video_path + '/' + i.split(' ')[0].strip())
                    # self.labels.append(int(i.split(" ")[1].strip())-1)
        # print(self.fnames)
        # print(self.labels)
        # print(self.fnames)

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        buffer = self.loadvideo(self.fnames[index])

        while buffer.shape[0] < self.clip_len + 2:
            index = np.random.randint(self.__len__())
            buffer = self.loadvideo(self.fnames[index])

        if self.mode == 'train':
            buffer = self.randomflip(buffer)

        # size =(112,112)
        # if size要改，crop函数里面的随机裁剪也要跟着改
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        temp = []
        for i in buffer:
            # cv2.imshow("1",i)
            # cv2.waitKey(10)
            temp.append(i)
        buffer = np.concatenate(temp, axis=2)
        # ________________________________________debug
        # b = video_transform(buffer)
        # b= b.numpy().transpose((1,2,3,0))
        # for i in b:
        #     print(i)
        if self.mode == 'validation':
            buffer = test_transform(buffer)
            label = torch.from_numpy(np.array(self.labels[index])).long()
            return self.diff(buffer), label
        elif self.mode == 'test':
            return self.diff(test_transform(buffer))
        else:
            buffer = video_transform(buffer)
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
        remainder = np.random.randint(self.frame_sample_rate)
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_height < frame_width:
            resize_height = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_width = int(float(resize_height) / frame_height * frame_width)
        else:
            resize_width = np.random.randint(self.short_side[0], self.short_side[1] + 1)
            resize_height = int(float(resize_width) / frame_width * frame_height)

        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        start_idx = 0
        end_idx = frame_count - 1
        frame_count_sample = frame_count // self.frame_sample_rate - 1
        if frame_count > 300:
            end_idx = np.random.randint(300, frame_count)
            start_idx = end_idx - 300
            frame_count_sample = 301 // self.frame_sample_rate - 1
        # buffer = np.empty((frame_count_sample, resize_height, resize_width, 3), np.dtype('float32'))
        buffer = []
        count = 0
        retaining = True
        sample_count = 0

        # read in each frame, one at a time into the numpy buffer array
        while (count <= end_idx and retaining):
            retaining, frame = capture.read()
            if count < start_idx:
                count += 1
                continue
            if retaining is False or count > end_idx:
                break
            if count % self.frame_sample_rate == remainder and sample_count < frame_count_sample:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # will resize frames if not already final size

                if (frame_height != resize_height) or (frame_width != resize_width):
                    frame = cv2.resize(frame, (resize_width, resize_height))
                buffer.append(frame)
                sample_count = sample_count + 1
            count += 1
        capture.release()

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
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

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
        DataLoader(VideoDataset_RDBdiff(mode='train'), batch_size=32, shuffle=False, num_workers=0)
    test = \
        DataLoader(VideoDataset_RDBdiff(mode='validation'), batch_size=32, shuffle=False, num_workers=0)
    print(len(train_dataloader))
    print(len(test))

    for step, (buffer, label) in enumerate(train_dataloader):
        print(buffer.shape)
        # print("label: ", label)
    # test_dataloader = \
    #     DataLoader( VideoDataset_RDBdiff( mode='test',clip_len=64), batch_size=1, shuffle=False, num_workers=0)
    # for i in train_dataloader:
    #     print(i[0].shape)
    #     print(i[1])