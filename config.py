params = dict()

params['num_classes'] = 101

params['epoch_num'] = 200
params['batch_size'] = 16
params['step'] = 10
params['num_workers'] = 4
params['learning_rate'] = 1e-2
params['momentum'] = 0.9
params['weight_decay'] = 1e-5
params['display'] = 10
params['pretrained'] = '/data/xudw/SlowFastNetworks-master/SlowFastNetworks-master/xudaw__65'
params['log'] = 'log'
params['save_path'] = 'UCF101'
params['clip_len'] = 64
params['frame_sample_rate'] = 1
# from libb.TSM import TSN
# import torch
# model = TSN(101,1,modality = 'RGB',partial_bn=True,is_shift=False)
# print(model)
# # (layer3): Sequential(
# #     (0): Bottleneck(
# #     (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
# # (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
# politics =model.get_optim_policies()
# # #--gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
# #
# pretrained_dict = torch.load('/data/xudw/SlowFastNetworks-master/SlowFastNetworks-master/TSM_pretrained.pth.tar', map_location='cpu')
# try:
#     model_dict = model.module.state_dict()
# except AttributeError:
#     model_dict = model.state_dict()
#
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# print("load pretrain model")
# for k, v in pretrained_dict.items():
#     if k in model_dict:
#         print(k)
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
# a = torch.ones(1,3,1,112,112)
# print(model(a).shape)
#
