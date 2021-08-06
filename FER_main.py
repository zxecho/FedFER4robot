"""Train CK+ with PyTorch."""
import numpy as np
import argparse
import os
from typing import Dict

# 10 crop for data enhancement
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import FedFER.vision_utils as utils
import transforms as transforms
# import torchvision.transforms as transforms
from FedFER.FD_data_loading import CK
from FedFER.CK_utils import CK as CK_S
from FedFER.vgg import VGG
from FedFER.resnet import ResNet18
from FedFER.Fe2D import FE2D
from FedFER.myInceptionV3_FeatureExtractor import Inception3
from FedFER.myInceptionRestNet_FeatExt import Inception_ResNetv2, My_IncepRestNet, MySimpleNet
from FedFER.function_utils import plot_confusion_matrix, count_vars, save2json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class FER:
    def __init__(self, args):

        self.use_cuda = torch.cuda.is_available()

        self.best_Test_acc = 0  # best PrivateTest accuracy
        self.best_Test_acc_epoch = 0
        self.test_acc = 0

        self.learning_rate_decay_start = 10  # 50
        self.learning_rate_decay_every = 5  # 5
        self.learning_rate_decay_rate = 0.95  # 0.9

        cut_size = 44
        resize_ = 48
        # self.label_map = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        self.label_map = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

        self.path = os.path.join('StandardModel_results', opt.dataset_name + '_' + opt.model +
                                 '_{}'.format(opt.exp_num))

        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(resize_),
            # transforms.Grayscale(3),
            transforms.RandomCrop(cut_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # BU3DFE mean=[0.27676, 0.27676, 0.27676], std=[0.26701, 0.26701, 0.26701]
            # jaffe mean=[0.43192, 0.43192, 0.43192], std=[0.27979, 0.27979, 0.27979]
            # oulu mean=[0.36418, 0.36418, 0.36418], std=[0.20384, 0.20384, 0.20384]
            # ck-48  mean=[0.51194, 0.51194, 0.51194], std=[0.28913, 0.28913, 0.28913]
            transforms.Normalize(mean=args.dataset_mean_std[args.dataset_name]['mean'],
                                 std=args.dataset_mean_std[args.dataset_name]['std']),
        ])

        transform_test = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(resize_),
            # transforms.ToTensor(),
            # transforms.Grayscale(3),
            transforms.TenCrop(cut_size),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(
                lambda crops: torch.stack([transforms.Normalize(mean=args.dataset_mean_std[args.dataset_name]['mean'],
                                                                std=args.dataset_mean_std[args.dataset_name]['std'])(
                    transforms.ToTensor()(crop)) for crop in crops])),
            # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
        # trainset = CK_S(split='Training', fold=opt.fold, transform=transform_train)
        trainset = CK(opt.dataset_name + '.h5', split='training', transform=transform_train)
        self.trainloader = DataLoader(trainset, batch_size=opt.bs, shuffle=True)
        testset = CK(opt.dataset_name + '.h5', split='testing', transform=transform_test)
        self.testloader = DataLoader(testset, batch_size=4, shuffle=False)

        # Model
        if 'VGG' in opt.model:
            self.net = VGG(in_chanels=3, vgg_name=opt.model, class_num=len(self.label_map))
        elif 'ResNet' in opt.model:
            self.net = ResNet18(len(self.label_map))
        elif 'Fe2D' in opt.model:
            self.net = FE2D(3, len(self.label_map))
        elif 'myInception' in opt.model:
            self.net = Inception3(num_classes=6)
        elif 'myInceptRes' in opt.model:
            self.net = My_IncepRestNet(classes=6)
        elif 'mySimpleIncepRes' in args.model:
            self.net = MySimpleNet(classes=6)

        print('[INFO]Parameter numbers: ', count_vars(self.net))

        if opt.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir(self.path), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(os.path.join(self.path, 'Ipdt_{}_Test_model.t7'.format(opt.dataset_name)))

            self.net.load_state_dict(checkpoint['net'])
            self.best_Test_acc = checkpoint['Test_acc']
            self.best_Test_acc_epoch = checkpoint['best_Test_acc_epoch']
            self.start_epoch = self.best_Test_acc_epoch + 1
        else:
            print('==> Building model..')

        if self.use_cuda:
            self.net.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)

    # Training
    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        global Train_acc
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0

        if epoch > self.learning_rate_decay_start >= 0:
            frac = (epoch - self.learning_rate_decay_start) // self.learning_rate_decay_every
            decay_factor = self.learning_rate_decay_rate ** frac
            current_lr = opt.lr * decay_factor
            utils.set_lr(self.optimizer, current_lr)  # set the decayed rate
        else:
            current_lr = opt.lr
        print('learning_rate: %s' % str(current_lr))

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            utils.clip_gradient(self.optimizer, 0.1)
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            print("[>>>Training {}/{}]  ".format(batch_idx + 1, len(self.trainloader)),
                  'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            # utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #                    % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        Train_acc = 100. * correct / total

    def evaluate_v2(self, epoch=0, model_testing=False):

        self.net.eval()
        PrivateTest_loss = 0
        correct = 0
        total = 0

        conf_matrix = torch.zeros(len(self.label_map), len(self.label_map), dtype=torch.int32)

        # 绘制混淆矩阵
        def confusion_matrix(preds, labels, conf_matrix):
            preds = torch.argmax(preds, 1)
            for p, t in zip(preds, labels):
                conf_matrix[p, t] += 1
            return conf_matrix

        for inputs, targets in self.testloader:
            bs, ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            with torch.no_grad():
                outputs = self.net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops

            # 用于记录混淆矩阵
            conf_matrix = confusion_matrix(outputs_avg, targets, conf_matrix)

            loss = self.criterion(outputs_avg, targets)
            PrivateTest_loss += loss.item()
            _, predicted = torch.max(outputs_avg.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print("[>>>Evaluating]  ", 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (PrivateTest_loss, 100. * correct / total, correct, total))

        # Save checkpoint.
        Test_acc = 100. * correct / total
        self.test_acc = Test_acc.item()
        print('Test acc: ', Test_acc)

        if Test_acc > self.best_Test_acc:
            print('Saving..')
            print("best_Test_acc: %0.3f" % Test_acc)
            state = {'net': self.net.state_dict() if self.use_cuda else self.net,
                     'Test_acc': Test_acc,
                     'best_Test_acc_epoch': epoch,
                     }
            # if not os.path.isdir(opt.dataset_name + '_' + opt.model):
            #     os.mkdir(opt.dataset_name + '_' + opt.model)
            if not os.path.isdir(self.path):
                os.mkdir(self.path)
            torch.save(state, os.path.join(self.path, 'Ipdt_{}_model.t7'.format(opt.dataset_name)))

            self.best_Test_acc = Test_acc
            self.best_Test_acc_epoch = epoch
        if model_testing:
            plot_confusion_matrix(conf_matrix, self.label_map,
                                  normalize=False,
                                  title='Confusion matrix, without normalization',
                                  save_name='{}_test'.format(opt.dataset_name), save_path=self.path)


if __name__ == '__main__':
    dataset_mean_std = {'CK_48_SetAll': {'mean': [0.51194, 0.51194, 0.51194], 'std': [0.28913, 0.28913, 0.28913]},
                        'BU3DFE_SetAll': {'mean': [0.27676, 0.27676, 0.27676], 'std': [0.26701, 0.26701, 0.26701]},
                        'oulu_SetAll': {'mean': [0.36418, 0.36418, 0.36418], 'std': [0.20384, 0.20384, 0.20384]},
                        'oulu_All_Mixed_IID': {'mean': [0.36418, 0.36418, 0.36418],
                                               'std': [0.20384, 0.20384, 0.20384]},
                        'oulu_Fed_NonIID_L4_mixed_train': {'mean': [0.36418, 0.36418, 0.36418],
                                                           'std': [0.20384, 0.20384, 0.20384]},
                        'BU3DFE_All_Mixed_IID': {'mean': [0.27676, 0.27676, 0.27676],
                                                 'std': [0.26701, 0.26701, 0.26701]},
                        'BU3DFE_Fed_NonIID_L4_mixed_train': {'mean': [0.27676, 0.27676, 0.27676],
                                                             'std': [0.26701, 0.26701, 0.26701]},
                        'CK_All_Mixed_IID': {'mean': [0.51194, 0.51194, 0.51194], 'std': [0.28913, 0.28913, 0.28913]},
                        'CK_Fed_NonIID_L4_mixed_train': {'mean': [0.51194, 0.51194, 0.51194],
                                                         'std': [0.28913, 0.28913, 0.28913]},
                        'jaffe_All_Mixed_IID': {'mean': [0.43192, 0.43192, 0.43192],
                                                'std': [0.27979, 0.27979, 0.27979]},
                        'jaffe_Fed_NonIID_L4_mixed_train': {'mean': [0.43192, 0.43192, 0.43192],
                                                            'std': [0.27979, 0.27979, 0.27979]},
                        # test
                        'oulu_Fed_NonIID_test_mixed_train': {'mean': [0.36418, 0.36418, 0.36418],
                                                             'std': [0.20384, 0.20384, 0.20384]},
                        }
    parser = argparse.ArgumentParser(description='PyTorch FER Training')
    parser.add_argument('--model', type=str, default='ResNet',
                        help='CNN architecture (VGGs1, VGG11, ResNet, Fe2D, myInception, myInceptRes, mySimpleIncepRes)')
    parser.add_argument('--dataset_name', type=str, default='oulu_Fed_NonIID_L4_mixed_train',
                        help='dataset (BU3DFE_SetAll, oulu_SetAll, CK_48_SetAll, '
                             'oulu_All_Mixed_IID, oulu_Fed_NonIID_L4_mixed_train, oulu_Fed_NonIID_test_mixed_train'
                             'BU3DFE_All_Mixed_IID, BU3DFE_Fed_NonIID_L4_mixed_train,'
                             'CK_All_Mixed_IID, CK_Fed_NonIID_L4_mixed_train, '
                             'jaffe_All_Mixed_IID, jaffe_Fed_NonIID_L4_mixed_train)')
    parser.add_argument('--dataset_mean_std', default=dataset_mean_std, type=Dict,
                        help='mean and std of data set for normalization')
    parser.add_argument('--exp_num', type=str, default='IID_vs_NonIID_T1', help='exp number')
    parser.add_argument('--fold', default=2, type=int, help='k fold number')
    parser.add_argument('--bs', default=64, type=int, help='batch_size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', type=bool, default=False, help='resume from checkpoint')

    opt = parser.parse_args()
    total_epoch = 100
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    param_list = ['CK_All_Mixed_IID', 'oulu_All_Mixed_IID', 'jaffe_All_Mixed_IID', 'BU3DFE_All_Mixed_IID',
                  'CK_Fed_NonIID_L4_mixed_train', 'oulu_Fed_NonIID_L4_mixed_train', 'jaffe_Fed_NonIID_L4_mixed_train',
                  'BU3DFE_Fed_NonIID_L4_mixed_train']
    for param in param_list:
        opt.dataset_name = param
        FERmodel = FER(opt)
        for epoch in range(start_epoch, total_epoch):
            FERmodel.train(epoch)
            # evaluate(epoch)
            FERmodel.evaluate_v2(epoch=epoch, model_testing=False)
        # 在测试集上进行验证，绘制混淆矩阵
        FERmodel.evaluate_v2(epoch=0, model_testing=True)
        print("best_Test_acc: %0.3f" % FERmodel.best_Test_acc)
        print("best_Test_acc_epoch: %d" % FERmodel.best_Test_acc_epoch)
        save2json(os.path.join(opt.dataset_name + '_' + opt.model + '_{}'.format(opt.exp_num)),
                  {'best_acc': FERmodel.best_Test_acc.item(),
                   'best_acc_epoch': FERmodel.best_Test_acc_epoch,
                   'last_test_acc': FERmodel.test_acc},
                  'best_acc&epoch')
        # 清空GPU缓存
        torch.cuda.empty_cache()
