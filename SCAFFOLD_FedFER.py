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
from FedFER.FER_train import IdpTrain
from FedFER.FER_train import Local_SCAFFOLD_Train as LocalTrain
from FedFER.FER_test import FERevaluate, FERevaluateV2
from FedFER.function_utils import norm, save_model, loss_plot, mkdir, weights_init
from FedFER.function_utils import save2json, count_vars, save2pkl
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
    TNet = None
    FedNet = None
    server_c = None
    clients_c = None
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
        server_c = MySimpleNet(classes=6)
        # 建立每个客户的独立个体
        clients_c_list = []

        for _ in range(training_fm_n):
            # 为每个站点新建一个独立的个体网络
            client_c = deepcopy(FedNet)
            clients_c_list.append(client_c)

    print('Number of model parameters: ', count_vars(FedNet))
    # # 权重初始化
    # FedNet.apply(weights_init)
    FedNet.cuda()
    server_c.cuda()

    # control vars init with 0.0
    server_c_w = server_c.state_dict()
    for key in server_c_w:
        if 'bn' not in key:
            server_c_w[key] *= 0.0
    server_c.load_state_dict(server_c_w)
    for param in server_c.parameters():
        param.requires_grad = False

    for client_c in clients_c_list:
        for param in client_c.parameters():
            param.requires_grad = False
        client_c_w = client_c.state_dict()
        for key in client_c_w:
            if 'bn' not in key:
                client_c_w[key] *= 0.0

    file_name = save_path.split('/')
    save_model_file = file_name[2] + '/' + file_name[3]

    if args.use_teacher:
        # Load checkpoint.
        print('==> Resuming from teacher model ..')
        assert os.path.isdir(args.Tnet_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(args.Tnet_path, 'Ipdt_{}_model.t7'.format(args.dataset_name)))

        TNet.load_state_dict(checkpoint['net'])
    elif args.use_pretrained_model:
        pretrained_model_path = './{}/Ipdt_{}_model.t7'.format(args.pretrained_model_path, args.pretrained_model_name)
        checkpoint = torch.load(pretrained_model_path)
        FedNet.load_state_dict(checkpoint['net'])
    elif args.resume:
        checkpoint = torch.load(os.path.join(model_saved_file, save_model_file, 'FedNet.pkl'))

        FedNet.load_state_dict(checkpoint)
    else:
        print('==> Building model..')

    # 建立每个客户的独立个体
    family_Nets_list = []

    for _ in range(training_fm_n):
        # 为每个站点新建一个独立的个体网络
        f_G = deepcopy(FedNet)
        family_Nets_list.append(f_G)

    print('FER network :\n', FedNet)

    FedNet.train()

    # training
    # 用于记录训练过程数据
    loss_train = []
    best_acc = 0
    best_epoch = 0

    # 所有参与方进行独立的本地训练
    if args.idpt_usrs_training:
        idpt_eval_acc_list = []
        print('Start local family model training!\n')
        for idp_N, train_dataset, test_dataset, f_name in zip(family_Nets_list, trainset_loader_list,
                                                              testset_loader_list, f_n_list):
            local_name = '{}'.format(f_name)
            print('[Idpt]==========={} Training process==========='.format(local_name))
            IdpTrain(args, idp_N, name=local_name, dataset_loader=train_dataset, save_path=save_path)
            idpt_test_acc = FERevaluate(args, idp_N, test_dataset, save_path, local_name)
            idpt_eval_acc_list.append(idpt_test_acc)
        # 保存每个家庭单独在本地训练集训练训练过程
        idpt_fig, axs = plt.subplots()
        axs.plot(range(len(idpt_eval_acc_list)), idpt_eval_acc_list, 'o')
        plt.savefig(save_path + 'all_fams_idpt_eval_acc.png')
        plt.cla()
        # save2json(save_path, idpt_eval_acc_list, 'idpt_eval_acc')
        save2pkl(save_path, idpt_eval_acc_list, 'idpt_eval_acc')
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
        total_delta = deepcopy(server_c.state_dict())
        for key in total_delta:
            total_delta[key] = 0.0
        client_c_locals = []
        usrs_weights = []
        # 用于随机抽取指定数量的参与者加入联邦学习
        m = min(args.num_users, training_fm_n)  # 对随机选择的数量进行判断限制
        m = max(m, 1)
        idxs_users = np.random.choice(training_fm_n, m, replace=False)
        for idx in idxs_users:
            w_l, local_loss, train_no, c_delta = LocalTrain(args,
                                                            LocalModel=copy.deepcopy(FedNet).to(device),
                                                            server_c=server_c,
                                                            client_c=clients_c_list[idx].cuda(),
                                                            dataset_loader=trainset_loader_list[idx],
                                                            client_numb=idx,
                                                            lr=current_lr)

            # 记录weights
            w_locals.append(copy.deepcopy(w_l))
            # 保存 client c
            client_c_locals.append(c_delta)

            for key in total_delta:
                total_delta[key] += c_delta[key]

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

        # aggregate and update server_c params
        for key in total_delta:
            total_delta[key] /= m
        c_global_para = server_c.state_dict()
        for key in c_global_para:
            if c_global_para[key].type() == 'torch.LongTensor':
                c_global_para[key] += total_delta[key].type(torch.LongTensor)
            elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
            else:
                # print(c_global_para[key].type())
                c_global_para[key] += total_delta[key]
        server_c.load_state_dict(c_global_para)


        # 全局模型载入联邦平均化之后的模型参数
        FedNet.load_state_dict(w_glob)
        # server_c.load_state_dict(server_c_w)

        # 打印训练过程的loss
        loss_avg = sum(loss_locals) / len(loss_locals)

        print('Fed Main Loop Round {:3d}, Average loss {:.3f}\n'.format(fed_iter, loss_avg))
        loss_train.append(loss_avg)

        FER_eval_acc, family_Nets_list = FERevaluateV2(args, FedNet, testset_loader_list, save_path, 'FedNet',
                                                       epoch=fed_iter)
        FER_eval_acc = FER_eval_acc.numpy().item()
        fed_eval_acc_list.append(FER_eval_acc)

        # 保存模型
        if best_acc < FER_eval_acc:
            best_acc = FER_eval_acc
            best_epoch = fed_iter
            # save_model_file = save_path.split('/')[2]
            save_model(FedNet, save_model_file, 'FedNet')

            # 保存联邦模型在各个本地训练集训练训练过程
            idpt_fig, axs = plt.subplots()
            axs.plot(range(len(family_Nets_list)), family_Nets_list, 'x')
            plt.savefig(save_path + 'all_fams_fed_eval_acc.png')
            plt.cla()
            # save2json(save_path, idpt_eval_acc_list, 'idpt_eval_acc')
            save2pkl(save_path, family_Nets_list, 'FedNet_eval_acc')
        fw_fed_main.write('{}\t {:.5f}\t  {}\t \n'.format(fed_iter, loss_avg, FER_eval_acc))

    print('Best test acc: ', best_acc, 'Best acc eval epoch: ', best_epoch)
    save2json(save_path, {'best_acc': best_acc, 'best acc epoch': best_epoch}, 'best_result&epoch')
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
    exp_total_time = 3
    cross_validation_sets = 1

    results_saved_file = 'results'
    results_plot_file = 'plot_results'
    model_saved_file = 'saved_model'
    # 'oulu_Fed_NonIID_L4_training', 'BU3DFE_Fed_NonIID_L4_training', 'jaffe_Fed_NonIID_L4_training'
    params_test_list = ['oulu_Fed_NonIID_L4_training', 'BU3DFE_Fed_NonIID_L4_training', 'jaffe_Fed_NonIID_L4_training']  # 用于设置不同参数
    test_param_name = 'dataset_name'

    for param in params_test_list:
        print('**  {} params test: {}  **'.format(test_param_name, param))

        # dataset_number = 'one_mi((A{})_1)'.format(param)
        args.dataset = param
        args.training_dataset_name = param
        args.testing_dataset_name = param
        args.exp_name = 'SCAFFOLD_vs_Idpt_{}_{}_baseline_T2'.format(args.dataset, args.model, param)
        ex_params_settings = {
            'algo_name': 'FedFER',
            'WeightAvg_method': args.weights_avg,
            'if_use_pretrained_model': args.use_pretrained_model,
            'if_grad_clip': args.grad_clip,
            'dataset': args.dataset,
            'exp_total_time': exp_total_time,
            'epochs': args.epochs,
            'batch_size': args.bs,
            'local_ep': args.local_ep,
            'local_bacthsize': args.local_bs,
            'activate_function': 'ReLU',
            'optimizer': 'SGD',
            'participant num': args.num_users,
            'fed_lr': args.fed_lr,
            'fed_lr_decay_fractor': args.fed_lr_decay,
            'L2_weight_decay': args.L2_weight_decay,
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
            exp_eval_r_list.append(eval_result)

        print('Exps results:  ', exp_eval_r_list)
        save2json(result_save_file, {'Exps_results': exp_eval_r_list, 'avg_best_acc': np.mean(exp_eval_r_list)},
                  args.dataset+'_all_avg_acc')
