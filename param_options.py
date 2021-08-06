import argparse
from typing import Dict

dataset_mean_std = {'CK_48_SetAll': {'mean': [0.51194, 0.51194, 0.51194], 'std': [0.28913, 0.28913, 0.28913]},
                    'BU3DFE_SetAll': {'mean': [0.27676, 0.27676, 0.27676], 'std': [0.26701, 0.26701, 0.26701]},
                    'oulu_SetAll': {'mean': [0.36418, 0.36418, 0.36418], 'std': [0.20384, 0.20384, 0.20384]},
                    'oulu_FedSetAll_training': {'mean': [0.36418, 0.36418, 0.36418],
                                                'std': [0.20384, 0.20384, 0.20384]},
                    'BU3DFE_FedSetAll_training': {'mean': [0.27676, 0.27676, 0.27676],
                                                  'std': [0.26701, 0.26701, 0.26701]},
                    'CK_FedSetAll_training': {'mean': [0.51212, 0.51212, 0.51212], 'std': [0.29292, 0.29292, 0.29292]},
                    'oulu_Fed_NonIID_L4_training': {'mean': [0.36418, 0.36418, 0.36418],
                                                    'std': [0.20384, 0.20384, 0.20384]},
                    'BU3DFE_Fed_NonIID_L4_training': {'mean': [0.27676, 0.27676, 0.27676],
                                                      'std': [0.26701, 0.26701, 0.26701]},
                    'CK_Fed_NonIID_L4_training': {'mean': [0.51194, 0.51194, 0.51194],
                                                  'std': [0.28913, 0.28913, 0.28913]},
                    'jaffe_Fed_NonIID_L4_training': {'mean': [0.43192, 0.43192, 0.43192],
                                                     'std': [0.27979, 0.27979, 0.27979]},
                    'robot_FE_family_dataset': {'mean': [0.35028, 0.35028, 0.35028],
                                                'std': [0.18529, 0.18529, 0.18529]}
                    }


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Fed FER Training')

    # FER params
    parser.add_argument('--model', type=str, default='mySimpleIncepRes',
                        help='CNN architecture (VGGs1, VGG11, Fe2D, myInception, myInceptRes, mySimpleIncepRes)')
    parser.add_argument('--dataset', type=str, default='robot_FE_family_dataset',
                        help='dataset (CK_48_SetAll, BU3DFE_SetAll, '
                             'oulu_SetAll, jaffe_SetAll, '
                             'oulu_FedSetAll_training, BU3DFE_FedSetAll_training, CK_FedSetAll_training,'
                             'oulu_Fed_NonIID_L4_training, CK_Fed_NonIID_training, BU3DFE_Fed_NonIID_L4_training,'
                             'jaffe_Fed_NonIID_L4_training, robot_FE_family_dataset)')
    parser.add_argument('--training_dataset_name', type=str, default='robot_FE_family_dataset',
                        help='test dataset name')
    parser.add_argument('--testing_dataset_name', type=str, default='robot_FE_family_dataset',
                        help='test dataset name')
    parser.add_argument('--dataset_mean_std', default=dataset_mean_std, type=Dict,
                        help='mean and std of data set for normalization')
    parser.add_argument('--resize', default=48, type=int, help='resize')
    parser.add_argument('--cut_size', default=44, type=int, help='cut size')
    parser.add_argument('--fold', default=1, type=int, help='k fold number')
    parser.add_argument('--local_bs', type=int, default=64, help="local batch size: B")
    # 用于数据蒸馏
    parser.add_argument('--use_teacher', type=bool, default=False, help='resume from teacher model')
    parser.add_argument('--Tnet_path', type=str, default='oulu_SetAll_myInceptRes_T1', help='Teacher net path')

    # ------------------ 消融实验控制量 --------------------------
    parser.add_argument('--bs', default=8, type=int, help='batch_size')
    # 是否进行local训练
    parser.add_argument('--idpt_usrs_training', type=bool, default=True, help='If train family independently')
    parser.add_argument('--fed_usrs_training', type=bool, default=True, help='If train family federally')

    # 是否使用预训练模型
    parser.add_argument('--use_pretrained_model', type=bool, default=True, help='resume from pretrained model')
    parser.add_argument('--pretrained_model_path', type=str, default='CK_All_Mixed_IID_mySimpleIncepRes_gn_T1',
                        help='Pre-trained net path (CK_All_Mixed_IID_mySimpleIncepRes_gn_T1, '
                             'CK_All_Mixed_IID_mySimpleIncepRes_T1)')
    parser.add_argument('--pretrained_model_name', type=str, default='CK_All_Mixed_IID',
                        help='Pre-trained model name')
    # 是否使用之前训练过的模型
    parser.add_argument('--resume', '-r', type=bool, default=False, help='resume from checkpoint')
    parser.add_argument('--reuse_model_path', type=str, default='CK_All_Mixed_IID_mySimpleIncepRes_gn_T1',
                        help='Pre-trained net path')
    parser.add_argument('--label_map', type=list, default=['anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise'],
                        help='label map list')

    # federated arguments
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--idpt_epochs', type=int, default=150, help="rounds of idpt model training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    # WeightedAvg
    parser.add_argument('--weights_avg', type=bool, default=False, help="if use weights avg")
    #  ========= optimizer 相关 ==========
    # 是否使用Gradient clip
    parser.add_argument('--grad_clip', type=bool, default=True, help="if use gradients clip")
    parser.add_argument('--lr', type=float, default=2e-3, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.95, help="D learning rate decay rate")
    parser.add_argument('--decay_every', type=int, default=5, help="D learning rate decay ")
    # 专门用于联邦的网络
    parser.add_argument('--fed_lr', type=float, default=2e-3, help="Fed G learning rate")
    parser.add_argument('--fed_lr_decay', type=float, default=0.95, help="Fed G learning rate decay rate")
    parser.add_argument('--fed_lr_decay_step', type=int, default=5, help="Fed learning rate")

    # 一般固定，除非调优
    parser.add_argument('--decay_start', type=int, default=10, help="D learning rate decay start")
    parser.add_argument('--L2_weight_decay', type=float, default=1e-4, help="weight_decay rate")

    # ------------------------ 分割线 ----------------------------

    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")

    # other arguments
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name')
    parser.add_argument('--saved_path', type=str, default='./saved_model/', help='Experiment name')
    parser.add_argument('--use_amp', action='store_true', help='whether AUTOMATIC MIXED PRECISION(混合精度) or not')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether using cuda')
    parser.add_argument('--use_saved_model', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--training_batchsize', type=int, default=8, help="batch size")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 1)')
    args = parser.parse_args()

    return args
