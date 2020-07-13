# from libb import TSM

import os
import time
import numpy as np
import torch
# from config import params
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from lastdataset import VideoDataset,VideoDataset_RDBdiff
import sys
sys.path.append('/data/xudw/SlowFastNetworks-master/SlowFastNetworks-master/')
from libb import slowfastnet
os.environ['CUDA_VISIBLE_DEVICES']='2'
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
    test_dataloader_RGB_label_balance_class = \
        DataLoader(VideoDataset(mode='test'), batch_size=1, shuffle=False, num_workers=0)
    test_dataloader_diff_label_balance_class = \
        DataLoader(VideoDataset_RDBdiff(mode='test'), batch_size=1, shuffle=False, num_workers=0)

    print("load model")
    # model_rgb_old = slowfastnet.resnet50(class_num=101)
    # model_diff_old = slowfastnet.resnet50(class_num=101)
    model_rgb_label = slowfastnet.resnet50(class_num=101)
    model_diff_label = slowfastnet.resnet50(class_num=101)
    # model_tsm = TSM.TSN(101, 16, modality='RGB', is_shift=True, pretrain=None, base_model='resnet50')
    #todo boss!!!!!!!!!!!!  @241.84
    # if params['pretrained'] is not None:
    # model_rgb_old = load(model_rgb_old,'/data/xudw/SlowFastNetworks-master/SlowFastNetworks-master/slOwfast_112_pool_new_data_xinshujuquanbu_75.pth.tar')
    # model_diff_old = load(model_diff_old,'/data/xudw/SlowFastNetworks-master/SlowFastNetworks-master/slowfast_RGBdiff_pool_ELU_xinshuju_60.pth.tar')
    model_rgb_label = load(model_rgb_label,'79.01136664426613RGB.pth')
    model_diff_label = load(model_diff_label,'78.0597409453305diff.pth')
    # model_tsm = load(model_tsm,'second_operation/81.25weilaeltsm_16frames.pth')

    # model_rgb_old = model_rgb_old.cuda()
    # model_diff_old = model_diff_old.cuda()
    model_rgb_label = model_rgb_label.cuda()
    model_diff_label = model_diff_label.cuda()
    # model_tsm = model_tsm.cuda()
    # model = nn.DataParallel(model)  # multi-Gpu
    # model_rgb_old.eval()
    # model_diff_old.eval()
    model_rgb_label.eval()
    model_diff_label.eval()
    # model_tsm.eval()
    test = open('/data/xudw/test_cpu/xdw_baseline/data/annotations/mod-ucf101-test.txt','r')
    t = test.readlines()
    f_rgb = open('95rgb.txt','w')
    f_diff = open('95diff.txt','w')
    with open('95labelrgbdiff.txt','w') as f:
        for step, inputs in enumerate(zip(test_dataloader_RGB_label_balance_class,test_dataloader_diff_label_balance_class)):
            with torch.no_grad():
                outputs_RGB_weilabel = model_rgb_label(inputs[0].cuda())
                outputs_diff_weilabel = model_diff_label(inputs[1].cuda())
                # outputs_tsm = model_tsm(inputs[2].cuda())

                # measure accuracy and record loss

                outputs_RGB_weilabel = torch.sum(outputs_RGB_weilabel,dim=0)
                outputs_diff_weilabel = torch.sum(outputs_diff_weilabel , dim=0)
                # outputs_tsm = torch.sum(outputs_tsm * 0.1, dim=0)

                f_rgb.write(' '.join(map(str, outputs_RGB_weilabel.cpu().numpy())))
                f_rgb.write('\n')
                f_diff.write(' '.join(map(str, outputs_diff_weilabel.cpu().numpy())))
                f_diff.write('\n')
                outputs = outputs_RGB_weilabel*0.5 +outputs_diff_weilabel*0.5
                outputs = outputs.reshape(1,101)
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