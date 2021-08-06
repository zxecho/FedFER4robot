"""Train CK+ with PyTorch."""
import os
import copy
import json
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# 10 crop for data enhancement
import torch

import transforms as transforms
from FedFER.param_options import args_parser
from FedFER.FD_data_loading import CK, get_Family_dataset
from FedFER.Fed_FamData_load import get_fed_fam_dataset
from FedFER.FER_train import IdpTrain, LocalTrain
from FedFER.FER_test import FERevaluate, FERevaluateV2
from FedFER.function_utils import norm, save_model, loss_plot, mkdir, weights_init, count_vars
from FedFER.function_utils import save2json
from FedFER.FedAvg import FedWeightedAvg, FedAvg
from FedFER.CK_utils import CK as CK_S
from FedFER.vgg import VGG
from FedFER.resnet import ResNet18
from FedFER.Fe2D import FE2D
from FedFER.myInceptionV3_FeatureExtractor import Inception3
from FedFER.myInceptionRestNet_FeatExt import Inception_ResNetv2, My_IncepRestNet, MySimpleNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def fed_main(args, save_path=''):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 载入数据并进行构造
    # 载入所有选定站点的数据集
    trainset_loader_list, training_fm_n, f_n_list = get_fed_fam_dataset(args,
                                                                        data_name='{}.h5'.format(
                                                                            args.training_dataset_name),
                                                                        split='training')

    testset_loader_list, testing_fm_n, f_n_list = get_fed_fam_dataset(args,
                                                                      data_name='{}.h5'.format(
                                                                          args.training_dataset_name),
                                                                      split='testing')

    # 建立CNN model
    # FedNet = VGG(args.model, len(args.label_map)).to(device=device)
    FedNet = None
    # Model
    if 'VGG' in args.model:
        FedNet = VGG(in_chanels=3, vgg_name=args.model, class_num=len(args.label_map))
    elif 'ResNet' in args.model:
        FedNet = ResNet18(len(args.label_map))
    elif 'Fe2D' in args.model:
        FedNet = FE2D(3, len(args.label_map))
    elif 'myInception' in args.model:
        FedNet = Inception3(num_classes=6)
    elif 'myInceptRes' in args.model:
        FedNet = My_IncepRestNet(classes=6)
    elif 'mySimpleIncepRes' in args.model:
        FedNet = MySimpleNet(classes=6)

    print('[INFO]Number of model parameters: ', count_vars(FedNet))

    # # 权重初始化
    # FedNet.apply(weights_init)
    FedNet.cuda()

    # 建立每个客户的独立个体
    family_Nets_list = []

    for _ in range(training_fm_n):
        # 为每个站点新建一个独立的个体网络
        f_G = deepcopy(FedNet)
        family_Nets_list.append(f_G)

    print('FER network :\n', FedNet)

    FedNet.train()

    # 拷贝网络参数
    w_glob = FedNet.state_dict()

    # training
    # 用于记录训练过程数据
    loss_train = []
    best_acc = 0
    best_epoch = 0

    # 所有参与方进行独立的本地训练
    if args.idpt_usrs_training:
        print('Start local family model training!\n')
        for idp_N, dataset, number in zip(family_Nets_list, trainset_loader_list, range(training_fm_n)):
            local_name = 'Family{}'.format(number)
            print('==========={} Training process==========='.format(local_name))
            IdpTrain(args, idp_N, name=local_name, dataset_loader=dataset, save_path=save_path)
            FERevaluate(args, idp_N, save_path, local_name)
        # 清除之前独立学习的主循环所占用的显存空间
        torch.cuda.empty_cache()

    # 写入训练过程数据
    fw_name = save_path + 'Fed_main_training_' + 'log.txt'
    fw_fed_main = open(fw_name, 'w+')
    fw_fed_main.write('iter\t loss\t  Eval acc\t \n')

    # 联邦学习主循环
    fed_eval_acc_list = []
    for fed_iter in range(args.epochs):

        if fed_iter > args.decay_start >= 0:
            frac = (fed_iter - args.decay_start) // args.fed_lr_decay_step
            decay_factor = args.fed_lr_decay ** frac
            current_lr = args.fed_lr * decay_factor
        else:
            current_lr = args.fed_lr
        print('>>>>>>>>>>>>>>>>>>> Current Fed training iters: {}<<<<<<<<<<<<<<<<<<'.format(fed_iter))
        print('============= Fed learning_rate: %s' % str(current_lr))

        w_locals, loss_locals = [], []
        usrs_weights = []
        # 用于随机抽取指定数量的参与者加入联邦学习
        m = max(args.num_users, 1)
        idxs_users = np.random.choice(training_fm_n, m, replace=False)
        for idx in idxs_users:
            w_l, local_loss, train_no = LocalTrain(args,
                                                   LocalModel=copy.deepcopy(FedNet).to(device),
                                                   dataset_loader=trainset_loader_list[idx],
                                                   client_numb=idx,
                                                   lr=current_lr)

            # 记录weights
            w_locals.append(copy.deepcopy(w_l))
            # 记录参与方的模型参数权重
            usrs_weights.append(train_no)
            # 记录loss
            loss_locals.append(local_loss)

        # 使用联邦学习算法更新全局权重
        if args.weights_avg:
            w = norm(usrs_weights)
            w_glob = FedWeightedAvg(w_locals, w, use_soft=False)
        else:
            w_glob = FedAvg(w_locals)

        # 全局模型载入联邦平均化之后的模型参数
        FedNet.load_state_dict(w_glob)

        # 打印训练过程的loss
        loss_avg = sum(loss_locals) / len(loss_locals)

        print('Fed Main Loop Round {:3d}, Average loss {:.3f}'.format(fed_iter, loss_avg))
        loss_train.append(loss_avg)

        FER_eval_acc = FERevaluate(args, FedNet, save_path, 'FedNet', epoch=fed_iter)
        fed_eval_acc_list.append(FER_eval_acc.numpy().item())

        # 保存模型
        if best_acc < FER_eval_acc:
            best_acc = FER_eval_acc
            best_epoch = fed_iter
            # save_model_file = save_path.split('/')[2]
            file_name = save_path.split('/')
            save_model_file = file_name[2] + '/' + file_name[3]
            save_model(FedNet, save_model_file, 'FedNet')
        fw_fed_main.write('{}\t {:.5f}\t  {}\t \n'.format(fed_iter, loss_avg, FER_eval_acc))

    print('Best test acc: ', best_acc, 'Best acc eval epoch: ', best_epoch)
    # Evaluating FedNet
    # FER_eval_acc = FERevaluate(args, FedNet, save_path, 'FedNet')

    # 绘制曲线
    fig, axs = plt.subplots()
    loss_plot(axs, loss_train, 'FedNet train loss')
    plt.savefig(save_path + 'FedNet_training_loss.png')
    plt.cla()

    fig, axs = plt.subplots()
    loss_plot(axs, fed_eval_acc_list, 'FedNet evaluationg accuracy')
    # plt.savefig(save_path + 'fed_{}.eps'.format(args.epochs))
    plt.savefig(save_path + 'FedNet_evaluationg_accuracy.png')
    # 保存数据到本地
    save2json(save_path, {'fed_eval_acc': fed_eval_acc_list}, '{}_FedNet_eval_acc'.format(args.exp_name))

    # 关闭写入
    fw_fed_main.close()
    plt.cla()  # 清除之前绘图
    plt.close()
    # 清空GPU缓存
    torch.cuda.empty_cache()

    return best_acc


if __name__ == "__main__":
    args = args_parser()

    # 做实验
    exp_total_time = 1
    cross_validation_sets = 1

    results_saved_file = 'results'
    results_plot_file = 'plot_results'

    params_test_list = [5]
    test_param_name = 'numbers'

    dataset_name = '{}'.format(args.dataset)
    for param in params_test_list:
        print('**  {} params test: {}  **'.format(test_param_name, param))

        # dataset_number = 'one_mi((A{})_1)'.format(param)
        args.num_users = param
        args.exp_name = '{}_{}_n{}_T1'.format(dataset_name, args.model, param)
        ex_params_settings = {
            'algo_name': 'FedFER',
            'WeightAvg_method': 'non-soft',
            'dataset': dataset_name,
            'datasets_number': cross_validation_sets,
            'epochs': args.epochs,
            'local_ep': args.local_ep,
            'local_bacthsize': args.local_bs,
            'activate_function': 'ReLU',
            'optimizer': 'SGD',
            'participant num': args.num_users,
            'fed_lr': args.fed_lr,
            'fed_lr_decay_fractor': args.fed_lr_decay,
            'decay_every_step': args.decay_every,
            'idpt_lr': args.lr,
        }

        # 存储主文件路径
        result_save_file = './{}/'.format(results_saved_file) + args.exp_name + '/'
        mkdir(result_save_file)

        # 保存参数配置
        params_save_name = result_save_file + 'params_settings.json'
        with open(params_save_name, 'w+') as jsonf:
            json.dump(ex_params_settings, jsonf)

        exp_eval_r_list = []

        for exp_t in range(exp_total_time):
            print('******* Training epoch {} *******'.format(exp_t))
            save_path_pre = result_save_file + str(exp_t) + '/'
            mkdir(save_path_pre)
            eval_result = fed_main(args, save_path_pre)
            exp_eval_r_list.append(eval_result.item())

        print('Exps results:  ', exp_eval_r_list)
