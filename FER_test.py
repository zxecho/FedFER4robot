import torch
import torch.nn as nn
import torch.optim as optim
import transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from FedFER.vgg import VGG
from FedFER.resnet import ResNet18
from FedFER.Fe2D import FE2D
from FedFER.myInceptionV3_FeatureExtractor import Inception3
from FedFER.myInceptionRestNet_FeatExt import Inception_ResNetv2, My_IncepRestNet
from FedFER.param_options import args_parser
from FedFER.FD_data_loading import CK, get_Family_dataset
from FedFER.Fed_FamData_load import get_fed_fam_dataset
from FedFER.function_utils import load_model, plot_confusion_matrix, mkdir

label_map = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


def FERevaluate(args, net, testset_loader, save_path, name, epoch=None):
    Test_acc = 0
    Avg_test_acc_list = []
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    # 绘制混淆矩阵
    def confusion_matrix(preds, labels, conf_matrix):
        preds = torch.argmax(preds, 1)
        for p, t in zip(preds, labels):
            conf_matrix[p, t] += 1
        return conf_matrix

    conf_matrix = torch.zeros(len(args.label_map), len(args.label_map), dtype=torch.int32)
    with torch.no_grad():
        with tqdm(testset_loader) as tq:
            for inputs, targets in tq:
                # for batch_idx, (inputs, targets) in enumerate(dataloader):
                # for inputs, targets in testloader:
                bs, ncrops, c, h, w = np.shape(inputs)
                inputs = inputs.view(-1, c, h, w)

                if args.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
                outputs = net(inputs)
                outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
                # 用于记录混淆矩阵
                conf_matrix = confusion_matrix(outputs_avg, targets, conf_matrix)

                loss = criterion(outputs_avg, targets)
                PrivateTest_loss += loss.item()
                _, predicted = torch.max(outputs_avg.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                tq.set_postfix(loss=PrivateTest_loss, Acc=100. * correct / total)

    # Save checkpoint.
    Test_acc = 100. * correct / total

    eval_r_save_path = save_path + 'idpt_eval_info/'
    mkdir(eval_r_save_path)

    if epoch is not None and (epoch + 1) % 20 == 0:
        plot_confusion_matrix(conf_matrix, args.label_map,
                              normalize=False,
                              title='Confusion matrix, without normalization',
                              save_name='{}_test{}'.format(name, epoch if epoch is not None else ''),
                              save_path=eval_r_save_path)
    if epoch == 'one-time':
        plot_confusion_matrix(conf_matrix, args.label_map,
                              normalize=True,
                              title='Confusion matrix, without normalization',
                              save_name='{}_test{}'.format(name, epoch if epoch is not None else ''),
                              save_path=eval_r_save_path)

    return Test_acc


def FERevaluateV2(args, net, testloader_list, save_path, name, epoch=None):
    net.eval()
    PrivateTest_loss = 0
    each_fam_test_acc = []
    g_correct = 0
    g_total = 0

    criterion = nn.CrossEntropyLoss()

    # 绘制混淆矩阵
    def confusion_matrix(preds, labels, conf_matrix):
        preds = torch.argmax(preds, 1)
        for p, t in zip(preds, labels):
            conf_matrix[p, t] += 1
        return conf_matrix

    conf_matrix = torch.zeros(len(args.label_map), len(args.label_map), dtype=torch.int32)

    with torch.no_grad():
        for testloader in testloader_list:
            correct = 0
            total = 0
            with tqdm(testloader) as tq:
                for inputs, targets in tq:
                    # for batch_idx, (inputs, targets) in enumerate(dataloader):
                    # for inputs, targets in testloader:
                    bs, ncrops, c, h, w = np.shape(inputs)
                    inputs = inputs.view(-1, c, h, w)

                    if args.use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()
                    inputs, targets = Variable(inputs, volatile=True), Variable(targets)
                    outputs = net(inputs)
                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
                    # 用于记录混淆矩阵
                    conf_matrix = confusion_matrix(outputs_avg, targets, conf_matrix)

                    loss = criterion(outputs_avg, targets)
                    PrivateTest_loss += loss.item()
                    _, predicted = torch.max(outputs_avg.data, 1)
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()
                    g_total += total
                    g_correct += correct
                    fam_acc = torch.true_divide(correct, total)
                    tq.set_postfix(FamAcc=100. * fam_acc)
            each_fam_test_acc.append(fam_acc.item())
    g_Test_acc = 100. * g_correct / g_total
    print("Global Test_acc: %0.3f" % g_Test_acc)

    # Save checkpoint.
    # print('Saving..')
    # state = {'net': net.state_dict() if args.use_cuda else net,
    #          'Test_acc': g_Test_acc,
    #          }
    #
    # torch.save(state, os.path.join(save_path, '{}.t7'.format(name)))

    if epoch is not None and (epoch + 1) % 20 == 0:
        plot_confusion_matrix(conf_matrix, args.label_map,
                              normalize=False,
                              title='Confusion matrix, without normalization',
                              save_name='{}_test{}'.format(name, epoch if epoch is not None else ''),
                              save_path=save_path)

    return g_Test_acc,  each_fam_test_acc


def evaluate_v2(args, exp_time, save_path=''):
    """
    用于单独载入已经训练好的模型，获取模型在测试集上的混淆矩阵
    :param args:
    :param exp_time:
    :param save_path:
    :return:
    """
    use_cuda = torch.cuda.is_available()

    best_Test_acc = 0  # best PrivateTest accuracy
    best_Test_acc_epoch = 0

    cut_size = 80
    resize_ = 84
    label_map = ['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

    # Data
    print('==> Preparing data..')

    transform_test = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(resize_),
        # transforms.ToTensor(),
        # transforms.Grayscale(3),
        transforms.TenCrop(cut_size),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.36418, 0.36418, 0.36418],
                                                                          std=[0.20384, 0.20384, 0.20384])(
            transforms.ToTensor()(crop)) for crop in crops])),
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    ])

    # Model
    if 'VGG' in model:
        net = VGG(in_chanels=3, vgg_name=model, class_num=len(label_map))
    elif 'ResNet' in model:
        net = ResNet18(len(label_map))
    elif 'Fe2D' in model:
        net = FE2D(3, len(label_map))
    elif 'myInception' in model:
        net = Inception3(num_classes=6)
    elif 'myInceptRes' in model:
        net = My_IncepRestNet(classes=6)

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkp_path = args.saved_path + args.exp_name + '/{}'.format(exp_time)
    assert os.path.isdir(checkp_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(checkp_path, 'FedNet.pkl'))

    net.load_state_dict(checkpoint)
    net.eval()

    PrivateTest_loss = 0
    correct = 0
    total = 0

    testset = CK(args.testing_dataset_name + '.h5', split='testing', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    conf_matrix = torch.zeros(7, 7, dtype=torch.int32)

    # 绘制混淆矩阵
    def confusion_matrix(preds, labels, conf_matrix):
        preds = torch.argmax(preds, 1)
        for p, t in zip(preds, labels):
            conf_matrix[p, t] += 1
        return conf_matrix

    for inputs, targets in testloader:
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops

        # 用于记录混淆矩阵
        conf_matrix = confusion_matrix(outputs_avg, targets, conf_matrix)

        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print("[>>>Evaluating]  ", 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (PrivateTest_loss, 100. * correct / total, correct, total))

    # Save checkpoint.
    Test_acc = 100. * correct / total

    print('Test acc: ', Test_acc)

    plot_confusion_matrix(conf_matrix, label_map,
                          normalize=False,
                          title='Confusion matrix, without normalization',
                          save_name='{}_best_model_eval'.format(args.exp_name), save_path=save_path)


if __name__ == "__main__":
    args = args_parser()

    # 做实验
    exp_t = 1
    cross_validation_sets = 1

    model = 'myInceptRes'
    results_saved_file = 'results'
    results_plot_file = 'plot_results'
    dataset_name = 'oulu_SetAll'  # oulu_SetAll, CK_48_SetAll
    exp_no = 'T2'

    args.exp_name = '{}_{}_{}'.format(dataset_name, args.model, exp_no)
    result_save_file = './{}/'.format(results_saved_file) + args.exp_name + '/' + str(exp_t) + '/'
    evaluate_v2(args, exp_t, result_save_file)
