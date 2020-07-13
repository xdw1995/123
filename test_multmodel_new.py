from libb import TSM

import os
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from libb.dataset import VideoDataset,VideoDataset_RDBdiff
from libb import slowfastnet, dataset_tsm
import TSM

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(model, train_dataloader, epoch, criterion, optimizer):
    global niubi
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()
    for step, (inputs, labels) in enumerate(train_dataloader):

        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if (step+1) % params['display'] == 0:
            print('-------------------------------------------------------')
            for param in optimizer.param_groups:
                print('lr: ', param['lr'])
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step+1, len(train_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg,
                top5_acc=top5.avg)
            print(print_string)




def validation(model, val_dataloader, epoch, criterion, optimizer):
    global niubi
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_dataloader):
            data_time.update(time.time() - end)
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            print('----validation----')
            print_string = 'Epoch: [{0}][{1}/{2}]'.format(epoch, step + 1, len(val_dataloader))
            print(print_string)
            print_string = 'data_time: {data_time:.3f}, batch time: {batch_time:.3f}'.format(
                data_time=data_time.val,
                batch_time=batch_time.val)
            print(print_string)
            print_string = 'loss: {loss:.5f}'.format(loss=losses.avg)
            print(print_string)
            print_string = 'Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(
                top1_acc=top1.avg,
                top5_acc=top5.avg)
            print(print_string)
            if int(top1.avg)>=niubi:
                niubi = int(top1.avg)
                torch.save(model.module.state_dict(), 'slowfast_RGBdiff_pool_ELU_xinshuju_60'+".pth.tar")



def main():
    import warnings
    warnings.filterwarnings("ignore")
    from TSM import diff

    cudnn.benchmark = False
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # logdir = os.path.join(params['log'], cur_time)
    # if not os.path.exists(logdir):
    #     os.makedirs(logdir)


    print("Loading dataset")
    train_dataloader = \
        DataLoader(
            VideoDataset_RDBdiff(mode='train', frame_sample_rate=params['frame_sample_rate']),
            batch_size=64, shuffle=True, num_workers=params['num_workers'])

    val_dataloader = \
        DataLoader(
            VideoDataset_RDBdiff(mode='validation', frame_sample_rate=params['frame_sample_rate']),
            batch_size=64, shuffle=True, num_workers=params['num_workers'])
    # test_dataloader = \
    #     DataLoader(
    #         VideoDataset(1,mode='test', clip_len=params['clip_len'], frame_sample_rate=params['frame_sample_rate']),
    #         batch_size=32, shuffle=False, num_workers=params['num_workers'])
    # from libb.R2_1D import model_r3d
    # model = model_r3d(101,depth=50)
    # model = MFnet.MFNET_3D(num_classes=101, pretrained=False)
    model = slowfastnet.resnet50(class_num=101)
    # model = slowfastnet.resnet50(class_num=params['num_classes'],non_local =True)#237.71
    # if params['pretrained'] is not None:
    pretrained_dict = torch.load('/data/xudw/SlowFastNetworks-master/SlowFastNetworks-master/slowfast_RGBdiff_pool_ELU_xinshuju_60.pth.tar', map_location='cpu')
    try:
        model_dict = model.module.state_dict()
    except AttributeError:
        model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print("load pretrain model")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.cuda()
    model = nn.DataParallel(model)  # multi-Gpu

    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

    model_save_dir = os.path.join(params['save_path'], cur_time)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    for epoch in range(params['epoch_num']):
        train(model, train_dataloader, epoch, criterion, optimizer)
        if epoch%2==0:
            validation(model, val_dataloader, epoch, criterion, optimizer)
        # if epoch % 2== 0:
        scheduler.step()
        # if epoch % 1 == 0:
        #     checkpoint = os.path.join(model_save_dir,
        #                               "clip_len_biaozhun" + str(params['clip_len']) + "frame_sample_rate_" +str(params['frame_sample_rate'])+ "_checkpoint_" + str(epoch) + ".pth.tar")
        #     torch.save(model.module.state_dict(), checkpoint)
def load(model,path):
    pretrained_dict = torch.load(path, map_location='cpu')
    try:
        model_dict = model.module.state_dict()
    except AttributeError:
        model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print("load pretrain model")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

def submit():
    cudnn.benchmark = False
    cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    # test_dataloader_RGB_old = \
    #     DataLoader( VideoDataset(mode='test'), batch_size=10, shuffle=False, num_workers=0)
    # test_dataloader_diff_old = \
    #     DataLoader( VideoDataset_RDBdiff(mode='test'), batch_size=10, shuffle=False, num_workers=0)
    # test_dataloader_RGB_balance_class = \
    #     DataLoader(VideoDataset(mode='test'), batch_size=10, shuffle=False, num_workers=0)
    # test_dataloader_diff_balance_class = \
    #     DataLoader(VideoDataset_RDBdiff(mode='test'), batch_size=10, shuffle=False, num_workers=0)

    test_dataloader_tsm = \
        DataLoader(
            VideoDataset(mode='test', frame_sample_rate=1, clip_len=16),
            batch_size=10, shuffle=False, num_workers=0)

    model_tsm = TSM.TSN(101, 16, modality='RGB', is_shift=True, pretrain=None, base_model='resnet50')
    model_tsm = load(model_tsm,'tsmjiaocha1.pth')
    # pretrained_dict = torch.load('tsmjiaocha1.pth', map_location='cpu')
    #
    # model_tsm.load_state_dict(pretrained_dict)
    # print("load model")

    model_tsm = model_tsm.cuda()
    # model = nn.DataParallel(model)  # multi-Gpu
    # model_rgb_old.eval()
    # model_diff_old.eval()
    # model_rgb_.eval()
    # model_diff_.eval()
    model_tsm.eval()
    # f_RGB_ = open('RGB_85_10_2_rand_ge2.txt', 'w')
    # f_diff_ = open('diff_85_10_2_rand_ge2.txt', 'w')
    f_tsm = open('tsm_rgb_::-1.txt', 'w')

    test = open('mod-ucf101-test.txt','r')
    t = test.readlines()
    with open('submission_randge2.txt','w') as f:
        for step, inputs in enumerate(test_dataloader_tsm):
            with torch.no_grad():
                # outputs_RGB_ = model_rgb_(inputs[0].cuda())
                # outputs_diff_l = model_diff_(inputs[1].cuda())
                # print(inputs.shape)
                outputs_tsm = model_tsm(inputs.cuda())

                # measure accuracy and record loss

                # outputs_RGB_ = torch.sum(outputs_RGB_*0.1,dim=0)
                # outputs_diff_ = torch.sum(outputs_diff_ * 0.1, dim=0)
                # print(outputs_tsm.shape)
                outputs_tsm = torch.sum(outputs_tsm * 0.1, dim=0)
                outputs_tsm = outputs_tsm.reshape(1,101)
                # f_RGB_.write(' '.join(map(str, outputs_RGB_.cpu().numpy())))
                # f_RGB_.write('\n')
                # f_diff_.write(' '.join(map(str, outputs_diff_.cpu().numpy())))
                # f_diff_.write('\n')
                f_tsm.write(' '.join(map(str, outputs_tsm.cpu().numpy())))
                f_tsm.write('\n')
                outputs = outputs_tsm
                # print(outputs.shape)
                # outputs = outputs_RGB_*0.33 +outputs_diff_*0.33
                # outputs = outputs.reshape(-1,101)
                maxk = 5
                _, pred = outputs.data.topk(maxk, 1, True, True)
                pred = pred[0].cpu().numpy()+np.ones((1,))
                pred = pred.astype(np.int)
                f.write(t[step].strip()+" "+' '.join(map(str,pred)))
                print(step)
                print(t[step].strip()+" "+' '.join(map(str,pred)))
                f.write('\n')
                f.flush()
    test.close()

if __name__ == '__main__':
    global niubi
    niubi =72
    print("xdw")
    submit()