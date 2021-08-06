import os
import json
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch

from FedFER.function_utils import save2json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# label_map = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

def get_normalization_value(data_name):
    lbs = os.listdir(data_name)
    all_dataset = []
    for lb in lbs:
        lb_dir = os.path.join(data_path, lb)
        fs = os.listdir(lb_dir)
        for img_path in fs:
            pixel_data = get_img((lb, img_path), data_name)
            all_dataset.append(pixel_data)
    all_dataset = np.array(all_dataset) / 255
    mean = all_dataset.mean()
    std = all_dataset.std()

    return mean, std


def all_fe_label_classification(data_path, lbs, source_dataset_name, mark_len=4):
    # 将原本的表情数据集按照被试划分
    fs_dict = dict()
    for k in range(len(lbs)):
        lb = lbs[k]
        lb_dir = data_path + lb
        fs = os.listdir(lb_dir)
        print('Current processing label dir: ', fs)

        p_mark = fs[0][:mark_len]
        if p_mark not in fs_dict.keys():
            fs_dict[p_mark] = dict()  # 用于存储不同人的表情字典数据集
        p_f = []  # 用于临时存储当前表情标签的不同人的列表
        for i in range(len(fs)):
            n_mark = fs[i][:mark_len]
            if p_mark == n_mark:
                p_f.append(fs[i])
            else:
                fs_dict[p_mark][lb] = p_f
                p_mark = n_mark
                if p_mark not in fs_dict.keys():
                    fs_dict[p_mark] = dict()
                p_f = []
                p_f.append(fs[i])
            # 判断是否是最后的对象，如果是则加入，避免遗漏最后的对象数据
            if i == len(fs) - 1:
                fs_dict[p_mark][lb] = p_f

    print('Person numbers: ', len(fs_dict.keys()))
    # print(fs_dict)
    # for k in fs_dict.keys():
    #     print('{}: {}'.format(k, len(fs_dict[k].keys())))

    save_to_json('./dataset/{}_Sat.json'.format(source_dataset_name), fs_dict)

    return fs_dict


def get_fed_dataset_Sat(data_path, lbs, source_dataset_name, mark_len=4):
    # 将原本的表情数据集按照被试划分，所有数据{‘People_ID’: [{'data_name':label}, {}, ..., {}]}, 并构建联邦数据集
    fs_dict = dict()
    for k in range(len(lbs)):
        lb = lbs[k]
        lb_dir = data_path + lb  # label
        fs = os.listdir(lb_dir)  # data file name
        print('Current processing label dir: ', fs)

        p_mark = fs[0][:mark_len]
        if p_mark not in fs_dict.keys():
            fs_dict[p_mark] = []  # 用于存储不同人的表情字典数据集

        for i in range(len(fs)):
            n_mark = fs[i][:mark_len]
            if p_mark == n_mark:
                fs_dict[p_mark].append({fs[i]: lb})
            else:
                p_mark = n_mark
                if p_mark not in fs_dict.keys():
                    fs_dict[p_mark] = []
                fs_dict[p_mark].append({fs[i]: lb})

    print('Person numbers: ', len(fs_dict.keys()))
    # print(fs_dict)
    # for k in fs_dict.keys():
    #     print('{}: {}'.format(k, len(fs_dict[k].keys())))

    save_to_json('./dataset/{}_FedSet.json'.format(source_dataset_name), fs_dict)

    return fs_dict


def construct_all_fe_mixed_dataset(data_path, source_dataset_name, save_name):
    # 构建混合数据集，将所有表情数据混合，划分测试和训练集
    lbs = os.listdir(data_path)
    label_map = lbs
    d_l = []  # 用于存储所有数据
    for k in range(len(lbs)):
        lb = lbs[k]
        lb_dir = data_path + lb
        fs = os.listdir(lb_dir)
        print('Current processing label dir: ', fs)

        for i in range(len(fs)):
            data_t = dict()  # 用于存储每个表情字典数据集
            data_t[k] = fs[i]
            d_l.append(data_t)

    # print(fs_dict)
    # for k in fs_dict.keys():
    #     print('{}: {}'.format(k, len(fs_dict[k].keys())))

    save_to_json('./dataset/{}_AllMixedIID.json'.format(source_dataset_name), d_l)

    n = len(d_l)
    print('data set numbers: \n', n)
    print('data list: ', d_l)

    dataset_array = np.array([i for i in range(len(d_l))])
    training_rate = 0.8
    training_num = int(n * training_rate)
    testing_num = n - training_num
    random = np.random.permutation(np.array([i for i in range(n)])).tolist()
    training_set = dataset_array[random[:training_num]]
    testing_set = dataset_array[random[training_num:]]
    print(random)
    print("training data: ", training_set)
    print("testing data: ", testing_set)

    with h5py.File('./dataset/{}.h5'.format(save_name), 'w') as h5f:
        group = h5f.create_group('training')
        training_data_list = []
        training_label_list = []
        for d in training_set:
            for k, v in d_l[d].items():
                pixel_data = get_img((label_map[k], v), source_dataset_name)
                training_data_list.append(pixel_data)
                training_label_list.append(k)

            training_data = np.array(training_data_list)
        training_label = np.array(training_label_list, dtype=np.int64)

        group.create_dataset('FEdata_pixel', data=training_data)
        group.create_dataset('FEdata_label', data=training_label)

        group = h5f.create_group('testing')
        testing_data_list = []
        testing_label_list = []
        for d in testing_set:
            for k, v in d_l[d].items():
                pixel_data = get_img((label_map[k], v), source_dataset_name)
                testing_data_list.append(pixel_data)
                testing_label_list.append(k)
        testing_data = np.array(testing_data_list)
        testing_label = np.array(testing_label_list, dtype=np.int64)

        group.create_dataset('FEdata_pixel', data=testing_data)
        group.create_dataset('FEdata_label', data=testing_label)

    h5f.close()


def select_fe_people(n, data_sat, source_dataset_name):
    # 包含n种不同表情的人的数据集筛选
    select_dict = dict()
    for k in data_sat.keys():
        if len(data_sat[k]) >= n:
            select_dict[k] = data_sat[k]

    save_to_json('./dataset/{}_Set{}.json'.format(source_dataset_name, n), select_dict)

    return select_dict


def construct_training_testing_set(source_data_sat, dataset_name='SetAll', src_dataset_name='', norm=True):
    """"
    :parameter
        source_data_sat: 原始表情数据的按个人划分的统计 dictionary
    """
    keys_array = np.array([k for k in source_data_sat.keys()])
    n = len(source_data_sat.keys())
    training_rate = 0.8
    training_num = int(n * training_rate)
    testing_num = n - training_num
    random = np.random.permutation(np.array([i for i in range(n)])).tolist()
    training_set = keys_array[random[:training_num]]
    testing_set = keys_array[random[training_num:]]
    print(random)
    print("training data: ", training_set)
    print("testing data: ", testing_set)

    # save all data set to .h5
    save_all_data_as_h5('{}'.format(dataset_name), source_data_sat, src_dataset_name, training_set, testing_set, norm)
    check_h5('./dataset/{}.h5'.format(dataset_name))

    # 生成训练集家庭
    training_set_list = []
    p = 0
    while p <= training_num:
        m = np.random.randint(1, 5)
        t = []
        if p + m < training_num:
            for i in range(m):
                t.append(training_set[p + i])
        else:
            for i in range(training_num - p):
                t.append(training_set[p + i])
            training_set_list.append(t)
            break
        p = p + m
        training_set_list.append(t)
    print('Training families: ', training_set_list)

    save_family_data_as_h5('{}_training'.format(dataset_name), source_data_sat, src_dataset_name,
                           training_set_list, norm)

    # 查看h5文件
    check_h5('./dataset/{}_training.h5'.format(dataset_name))

    # 生成测试集家庭
    testing_set_list = []
    p = 0
    while p <= testing_num:
        m = np.random.randint(1, 4)
        t = []
        if p + m < testing_num:
            for i in range(m):
                t.append(testing_set[p + i])
        else:
            for i in range(testing_num - p):
                t.append(testing_set[p + i])
            testing_set_list.append(t)
            break
        p = p + m
        testing_set_list.append(t)
    print('Testing families: ', testing_set_list)

    save_family_data_as_h5('{}_testing'.format(dataset_name), source_data_sat, src_dataset_name,
                           testing_set_list, norm)

    # 查看h5文件
    check_h5('./dataset/{}_testing.h5'.format(dataset_name))


def construct_fed_dataset(source_data_sat, source_dataset_name, save_name):
    """"
    （ID 同，label异）
    按照被试划分，构建联邦问题数据集，以家庭为单位，每个家庭每个家庭成员即每个人的0.8为训练集，0.2为测试集
    :parameter
        source_data_sat: 原始表情数据的按被试划分的统计 dictionary
    """
    keys_array = np.array([k for k in source_data_sat.keys()])
    # 打乱排序
    keys_array = np.random.permutation(keys_array)
    n = len(source_data_sat.keys())
    # 生成训练集家庭
    training_set_list = []
    p = 0
    while p <= n:
        m = np.random.randint(1, 5)
        t = []
        if p + m < n:
            for i in range(m):
                t.append(keys_array[p + i])
        else:
            for i in range(n - p):
                t.append(keys_array[p + i])
            training_set_list.append(t)
            break
        p = p + m
        training_set_list.append(t)
    print('Training families: ', training_set_list)

    training_rate = 0.8
    # 将所有家庭数据混合,用于对比non-IID 与 IID
    mixed_fam_data_train_list = []
    mixed_fam_label_train_list = []
    mixed_fam_data_test_list = []
    mixed_fam_label_test_list = []

    # save all training data set to .h5
    with h5py.File('./dataset/{}_training.h5'.format(save_name), 'w') as h5f:

        # 按每个家庭进行数据构造
        group_train = h5f.create_group("training")
        group_test = h5f.create_group("testing")
        for i, fam in enumerate(training_set_list):
            train_f_group = group_train.create_group("Family{}".format(i))
            test_f_group = group_test.create_group("Family{}".format(i))
            training_data_list = []
            training_label_list = []
            test_data_list = []
            test_label_list = []
            for p in fam:
                p_data = source_data_sat[p]  # 获取当前背视的数据
                p_dn = len(p_data)  # 获得每个被试的数据数量
                # 随机化处理
                p_data = np.random.permutation(p_data)
                training_num = int(p_dn * training_rate)
                cur_p_training_set = p_data[:training_num]
                cur_p_test_set = p_data[training_num:]
                for d in cur_p_training_set:
                    for k, v in d.items():
                        pixel_data = get_img((v, k), source_dataset_name)  # 获得原本img data
                        training_data_list.append(pixel_data)
                        training_label_list.append(label_map.index(v))

                        # 添加到混合dataset train dataset
                        mixed_fam_data_train_list.append(pixel_data)
                        mixed_fam_label_train_list.append(label_map.index(v))

                for d in cur_p_test_set:
                    for k, v in d.items():
                        pixel_data = get_img((v, k), source_dataset_name)  # 获得原本img data
                        test_data_list.append(pixel_data)
                        test_label_list.append(label_map.index(v))

                        # 添加到混合dataset train dataset
                        mixed_fam_data_test_list.append(pixel_data)
                        mixed_fam_label_test_list.append(label_map.index(v))

            training_data = np.array(training_data_list)
            training_label = np.array(training_label_list, dtype=np.int64)

            test_data = np.array(test_data_list)
            test_label = np.array(test_label_list, dtype=np.int64)

            train_f_group.create_dataset('FEdata_pixel', data=training_data)
            train_f_group.create_dataset('FEdata_label', data=training_label)
            test_f_group.create_dataset('FEdata_pixel', data=test_data)
            test_f_group.create_dataset('FEdata_label', data=test_label)

    h5f.close()

    check_h5('./dataset/{}_training.h5'.format(save_name))
    # 存储mixed data set 为 .h5文件
    with h5py.File('./dataset/{}_mixed_train.h5'.format(save_name), 'w') as mh5f:
        mixed_group_train = mh5f.create_group("training")
        mixed_group_test = mh5f.create_group("testing")

        training_data = np.array(mixed_fam_data_train_list)
        training_label = np.array(mixed_fam_label_train_list, dtype=np.int64)

        test_data = np.array(mixed_fam_data_test_list)
        test_label = np.array(mixed_fam_label_test_list, dtype=np.int64)

        mixed_group_train.create_dataset('FEdata_pixel', data=training_data)
        mixed_group_train.create_dataset('FEdata_label', data=training_label)
        mixed_group_test.create_dataset('FEdata_pixel', data=test_data)
        mixed_group_test.create_dataset('FEdata_label', data=test_label)
    mh5f.close()
    check_h5('./dataset/{}_mixed_train.h5'.format(save_name))


def construct_fed_mixed_dataset(source_data_sat, source_dataset_name, save_name, mark_len=None):
    """"
    （ID，label）均随机抽取
    按照被试划分，构建联邦问题数据集，以家庭为单位，每个家庭所有成员样本混合，每个家庭的0.8为训练集，0.2为测试集
    :parameter
        source_data_sat: 原始表情数据的按被试划分的统计 dictionary
    """
    keys_array = np.array([k for k in source_data_sat.keys()])
    # 打乱排序
    keys_array = np.random.permutation(keys_array)
    n = len(source_data_sat.keys())
    # 生成训练集家庭
    training_set_list = []
    p = 0
    while p <= n:
        m = np.random.randint(1, 5)
        t = []
        if p + m < n:
            for i in range(m):
                t.append(keys_array[p + i])
        else:
            for i in range(n - p):
                t.append(keys_array[p + i])
            training_set_list.append(t)
            break
        p = p + m
        training_set_list.append(t)
    print('Training families: ', training_set_list)

    training_rate = 0.8
    # 用于统计数据集（ID, label）概况
    train_ID_dict = dict()
    train_label_dict = {k: 0 for k in label_map}
    test_ID_dict = dict()
    test_label_dict = {k: 0 for k in label_map}
    # 将所有家庭数据混合,用于对比non-IID 与 IID
    mixed_fam_data_train_list = []
    mixed_fam_label_train_list = []
    mixed_fam_data_test_list = []
    mixed_fam_label_test_list = []

    # 用于总体统计当前数据集每个家庭的训练集与测试集的状况
    fam_train_stat_dict = dict()
    fam_test_stat_dict = dict()

    # save all training data set to .h5
    with h5py.File('./dataset/{}_training.h5'.format(save_name), 'w') as h5f:

        # 按每个家庭进行数据构造
        group_train = h5f.create_group("training")
        group_test = h5f.create_group("testing")
        for i, fam in enumerate(training_set_list):
            train_f_group = group_train.create_group("Family{}".format(i))
            test_f_group = group_test.create_group("Family{}".format(i))
            # 用于记录当前家庭的训练集和测试集数据
            training_data_list = []
            training_label_list = []
            test_data_list = []
            test_label_list = []
            # 用于记录家庭所有成员数据
            fam_pds = []
            fam_l = 0

            # 将家庭成员数据合并
            for p in fam:
                fam_pds = fam_pds + source_data_sat[p]  # 获取当前背视的数据
                fam_l = fam_l + len(source_data_sat[p])  # 获得每个被试的数据数量

            # 构造local data set
            ''' 1. 所有家庭数据随机打乱'''
            # fam_pds = np.random.permutation(fam_pds)
            ''' 2. 随机选择每个家庭的label'''
            # select local data set by labels for constructing Non-IID data set
            # select_ls = np.random.permutation(label_map)
            # train_labels = select_ls[:4]
            # test_labels = select_ls[4:]
            # cur_f_training_set = get_selected_lbs_data(fam_pds, train_labels)
            # cur_f_test_set = get_selected_lbs_data(fam_pds, test_labels)
            ''' 3. 每个家庭数据进行截断'''
            # training_num = int(fam_l * training_rate)
            # cur_p_training_set = fam_pds[:training_num]
            # cur_p_test_set = fam_pds[training_num:]

            ''' 4. Dirichlet 采样 '''
            cur_f_training_set, cur_f_test_set, f_train_stat, f_test_stat = get_data_by_dirichlet_dist(fam_pds,
                                                                                                       label_map,
                                                                                                       training_rate,
                                                                                                       i,
                                                                                                       save_name,
                                                                                                       10.0)

            fam_train_stat_dict['Family{}'.format(i)] = [f_train_stat[k] for k in label_map]
            fam_test_stat_dict['Family{}'.format(i)] = [f_test_stat[k] for k in label_map]

            for d in cur_f_training_set:
                for k, v in d.items():
                    # 统计训练集信息
                    if k[:mark_len] not in train_ID_dict.keys():
                        train_ID_dict[k[:mark_len]] = 1
                    else:
                        train_ID_dict[k[:mark_len]] += 1

                    train_label_dict[v] += 1

                    pixel_data = get_img((v, k), source_dataset_name)  # 获得原本img data
                    training_data_list.append(pixel_data)
                    training_label_list.append(label_map.index(v))

                    # 添加到混合dataset train dataset
                    mixed_fam_data_train_list.append(pixel_data)
                    mixed_fam_label_train_list.append(label_map.index(v))

            for d in cur_f_test_set:
                for k, v in d.items():
                    # 统计测试集信息
                    if k[:mark_len] not in test_ID_dict.keys():
                        test_ID_dict[k[:mark_len]] = 1
                    else:
                        test_ID_dict[k[:mark_len]] += 1

                    test_label_dict[v] += 1

                    pixel_data = get_img((v, k), source_dataset_name)  # 获得原本img data
                    test_data_list.append(pixel_data)
                    test_label_list.append(label_map.index(v))

                    # 添加到混合dataset train dataset
                    mixed_fam_data_test_list.append(pixel_data)
                    mixed_fam_label_test_list.append(label_map.index(v))

            training_data = np.array(training_data_list)
            training_label = np.array(training_label_list, dtype=np.int64)

            test_data = np.array(test_data_list)
            test_label = np.array(test_label_list, dtype=np.int64)

            train_f_group.create_dataset('FEdata_pixel', data=training_data)
            train_f_group.create_dataset('FEdata_label', data=training_label)
            test_f_group.create_dataset('FEdata_pixel', data=test_data)
            test_f_group.create_dataset('FEdata_label', data=test_label)

    h5f.close()

    # 存储统计数据
    save2json('./dataset/{}/'.format(save_name), train_ID_dict, 'train_ID_stat_info')
    save2json('./dataset/{}/'.format(save_name), train_label_dict, 'train_label_stat_info')
    save2json('./dataset/{}/'.format(save_name), test_ID_dict, 'test_ID_stat_info')
    save2json('./dataset/{}/'.format(save_name), test_label_dict, 'test_label_stat_info')

    plot_fam_stat(fam_train_stat_dict, label_map, save_name+'_train')
    plot_fam_stat(fam_test_stat_dict, label_map, save_name+'_test')

    check_h5('./dataset/{}_training.h5'.format(save_name))
    # 存储mixed data set 为 .h5文件
    with h5py.File('./dataset/{}_mixed_train.h5'.format(save_name), 'w') as mh5f:
        mixed_group_train = mh5f.create_group("training")
        mixed_group_test = mh5f.create_group("testing")

        training_data = np.array(mixed_fam_data_train_list)
        training_label = np.array(mixed_fam_label_train_list, dtype=np.int64)

        test_data = np.array(mixed_fam_data_test_list)
        test_label = np.array(mixed_fam_label_test_list, dtype=np.int64)

        mixed_group_train.create_dataset('FEdata_pixel', data=training_data)
        mixed_group_train.create_dataset('FEdata_label', data=training_label)
        mixed_group_test.create_dataset('FEdata_pixel', data=test_data)
        mixed_group_test.create_dataset('FEdata_label', data=test_label)
    mh5f.close()
    check_h5('./dataset/{}_mixed_train.h5'.format(save_name))


def get_img(img_info, src_dataset):
    img_path = './{}/{}/{}'.format(src_dataset, img_info[0], img_info[1])
    img = Image.open(img_path).convert('L')
    array_img = np.array(img)
    if len(array_img.shape) < 3:
        array_img = np.expand_dims(array_img, axis=2)
    return array_img


def get_selected_lbs_data(fam_data_list, label_list):
    # 获取选择的label的数据
    t_list = []
    for fm_d in fam_data_list:
        for v in fm_d.values():
            if v in label_list:
                t_list.append(fm_d)

    return t_list


def get_data_by_dirichlet_dist(fam_data_list, label_list, train_rate, fam_No, save_name, Concentration=10.):
    # 用于当前家庭数据的统计, 并画图
    cur_fam_train_dict = dict()
    cur_fam_test_dict = dict()

    ln = len(label_list)  # 标签长度
    train_list = []
    test_list = []
    fam_data_dict_set_by_lb = get_label_dict_set(fam_data_list, label_list)
    p = torch.tensor([1 / ln for _ in range(ln)])  # 初始化Dirichlet采样概率
    # 初始化Dirichlet 采样函数
    dist = torch.distributions.dirichlet.Dirichlet(Concentration * p)
    p = dist.sample()  # 采样概率
    fam_lp = np.array([round(p.item(), 1) for p in p])  # 根据采样得到的概率
    for lb, lp in zip(label_list, fam_lp):
        cur_lb_data = fam_data_dict_set_by_lb[lb]  # 当前标签的家庭成员所有数据
        lsn = 0
        if lp > 0:
            train_rate = max((1 - lp), train_rate)
            lsn = int(len(cur_lb_data) * train_rate)  # 当前样本需要采样个数

        cur_fam_train_dict[lb] = lsn
        cur_fam_test_dict[lb] = len(cur_lb_data) - lsn

        # 将原本标签对应的所有数据随机化
        ramdom_lb_fam_data = np.random.permutation(cur_lb_data)
        train_list.extend(ramdom_lb_fam_data[:lsn])
        test_list.extend(ramdom_lb_fam_data[lsn:])

    save2json('./dataset/{}/fam_stat/train/'.format(save_name), cur_fam_train_dict,
              'fam{}_train_stat_info'.format(fam_No))
    save2json('./dataset/{}/fam_stat/test/'.format(save_name), cur_fam_test_dict,
              'fam{}_test_stat_info'.format(fam_No))

    return train_list, test_list, cur_fam_train_dict, cur_fam_test_dict


def get_label_dict_set(data_list, label_list):
    # 按照标签分类，将一个家庭的所有数据保存到一个字典中 {‘label1’: [{}, {}...{}], 'label2':[{}, {}...{}1]}
    t = {lb: [] for lb in label_list}
    for data in data_list:
        for v in data.values():
            t[v].append(data)
    return t


def plot_fam_stat(data, labels, save_name):
    fam_names = list(data.keys())
    data = np.array(list(data.values()), dtype=np.int32)
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.invert_yaxis()
    ax.xaxis.set_visible(True)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(labels, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(fam_names, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(labels), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    plt.savefig('E:/zx/FedLearning/FedFER/datasets_overview/{}_fam_stat.svg'.format(save_name))
    plt.close(fig)


def normalize(x):
    return (x - mean) / std


def get_label(p_mark, data_sat):
    return


def save_to_json(fname, data):
    with open(fname, 'w') as file_obj:
        json.dump(data, file_obj)


def load_json(fname):
    with open(fname, 'r') as file_obj:
        data = json.load(file_obj)

    return data


def save_all_data_as_h5(fname, source_dataset_sat, src_dataset_name, training_set, testing_set, norm=True):
    """:parameter
        将training data set 与 testing data set 保存为h5文件
    """
    with h5py.File('./dataset/{}.h5'.format(fname), 'w') as h5f:
        group = h5f.create_group('training')
        training_data_list = []
        training_label_list = []
        for p in training_set:
            for fe in source_dataset_sat[p].keys():
                for data_n in source_dataset_sat[p][fe]:
                    pixel_data = get_img((fe, data_n), src_dataset_name)
                    training_data_list.append(pixel_data)
                    training_label_list.append(label_map.index(fe))
        if norm:
            training_data = normalize(np.array(training_data_list))
        else:
            training_data = np.array(training_data_list)
        training_label = np.array(training_label_list, dtype=np.int64)
        group.create_dataset('FEdata_pixel', data=training_data)
        group.create_dataset('FEdata_label', data=training_label)

        group = h5f.create_group('testing')
        testing_data_list = []
        testing_label_list = []
        for p in testing_set:
            for fe in source_dataset_sat[p].keys():
                for data_n in source_dataset_sat[p][fe]:
                    pixel_data = get_img((fe, data_n), src_dataset_name)
                    testing_data_list.append(pixel_data)
                    testing_label_list.append(label_map.index(fe))
        testing_data = np.array(testing_data_list)
        testing_label = np.array(testing_label_list, dtype=np.int64)

        group.create_dataset('FEdata_pixel', data=testing_data)
        group.create_dataset('FEdata_label', data=testing_label)

    h5f.close()


def save_family_data_as_h5(fname, source_dataset_sat, src_dataset_name, split_dataset, norm=True):
    """
    fname: string
    source_dataset_sat: dict
    split_dataste: List
    """
    Nf = len(split_dataset)  # 数据里家庭个数
    print('split_dataset: ', split_dataset)
    with h5py.File('./dataset/{}.h5'.format(fname), 'w') as h5f:
        for i, famliy in enumerate(split_dataset):
            group = h5f.create_group("Family{}".format(i))
            famliy_data_list = []
            famliy_label_list = []
            for fm in famliy:
                for fe in source_dataset_sat[fm].keys():
                    for data_n in source_dataset_sat[fm][fe]:
                        pixel_data = get_img((fe, data_n), src_dataset_name)
                        famliy_data_list.append(pixel_data)
                        famliy_label_list.append(label_map.index(fe))
            if norm:
                family_data = normalize(np.array(famliy_data_list))
            else:
                family_data = np.array(famliy_data_list)
            family_label = np.array(famliy_label_list, dtype=np.int64)
            print('Family {}  members: {}: '.format(i, len(famliy)), family_data.shape, family_label.shape)

            group.create_dataset('FEdata_pixel', data=family_data)
            group.create_dataset('FEdata_label', data=family_label)

    print(h5f)

    h5f.close()


def check_h5(fname):
    with h5py.File(fname, 'r') as f:
        for fkey in f.keys():
            print(f[fkey], fkey)

        print("======= 优雅的分割线 =========")

        for fm in f.keys():
            fm_group = f[fm]
            print('>>> Group: ', fm)
            for fm_p in fm_group.keys():
                print(fm_p, fm_group[fm_p])
        # dogs_group = f["dogs"]  # 从上面的结果可以发现根目录/下有个dogs的group,所以我们来研究一下它
        # for dkey in dogs_group.keys():
        #     print(dkey, dogs_group[dkey], dogs_group[dkey].name, dogs_group[dkey].value)

    f.close()


def construct_fedset(data_fp, src_dataset_name, mark_len, norm=False):
    lbs = os.listdir(data_fp)
    print(lbs)
    global label_map
    label_map = lbs
    # 整体数据处理和筛选
    # sat_data = all_fe_label_classification(data_fp, lbs, src_dataset_name, mark_len=mark_len)   # 用于构建所有数据混合在一起的数据集
    sat_data = get_fed_dataset_Sat(data_fp, lbs, src_dataset_name, mark_len=mark_len)   # 用于构建fed problem data sets.
    sat_data = load_json('./dataset/{}_FedSet.json'.format(src_dataset_name))
    print(sat_data)
    # 构造联邦学习具体数据集

    # construct_training_testing_set(sat_data, dataset_name=src_dataset_name + '_SetAll',
    #                                src_dataset_name=src_dataset_name, norm=norm)

    # construct_fed_dataset(sat_data, src_dataset_name, save_name=src_dataset_name + '_FedSetAll')

    construct_fed_mixed_dataset(sat_data, src_dataset_name,
                                save_name=src_dataset_name + '_Fed_NonIID_L4',
                                mark_len=mark_len)


if __name__ == '__main__':
    data_path = 'CK/'
    dataset_name = 'CK'
    mean, std = get_normalization_value(dataset_name)
    print('mean: ', mean, 'std: ', std)
    mark_len_dict = {'BU3DFE': 5, 'oulu': 4, 'CK': 4, 'jaffe': 2}
    construct_fedset(data_path, dataset_name, mark_len=mark_len_dict[dataset_name], norm=False)
    # construct_all_fe_mixed_dataset(data_path, dataset_name, dataset_name+'_All_Mixed_IID')
