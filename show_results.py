import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from FedFER.function_utils import load_pkl_file


def plot_fed_acc():
    # plt.style.use('fivethirtyeight')
    files_path = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/exp_acc_data'
    dir_l = os.listdir(files_path)
    datafile_list = []
    fig, ax = plt.subplots()
    for i, log in enumerate(dir_l):
        file_path = os.path.join(files_path, log)
        datafile = pd.read_table(file_path, header=None)
        datafile_list.append(datafile)
        x_axis_index = np.array(datafile.values[1:, 0], dtype=np.int)
        y_axis_value = np.array(datafile.values[1:, 2], dtype=np.float)
        plt.plot(x_axis_index, y_axis_value, label=str(3 + i))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.legend()

    plt.show()


def plot_fed_avg_acc():
    """
    用于绘制联邦模型的测试集准确率
    :return:
    """
    files_path = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/BU3DFE_exp_acc_data_K'
    pnum_dir_l = os.listdir(files_path)
    print(pnum_dir_l)
    # 创建绘图
    fig, ax = plt.subplots()

    for pdir in pnum_dir_l:
        filep_l = os.listdir(os.path.join(files_path, pdir))
        datafile_list = []
        for i, f in enumerate(filep_l):
            file_path = os.path.join(files_path, pdir, f)
            datafile = pd.read_table(file_path, header=None)
            datafile_list.append(np.array(datafile.values[1:, 2], dtype=np.float).reshape(-1, 1))
            x_axis_index = np.array(datafile.values[1:, 0], dtype=np.int)

        # 用于替代seaborn中的tsplot绘图函数
        def tsplot(ax, x, data, label_name='', **kw):
            est = np.mean(data, axis=1)
            sd = np.std(data, axis=1)
            cis = (est - sd, est + sd)

            ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
            ax.plot(x, est, label=label_name, **kw)
            ax.tick_params(labelsize=15)
            ax.set_ylabel('Test Accuracy(%)', size=15)
            ax.set_xlabel('Communication rounds', size=15)
            ax.margins(x=0)

            return est, sd

        xaxis_from_sets = []
        all_data = np.concatenate(datafile_list, axis=1)
        indicator_avg, indicator_std = tsplot(ax, x_axis_index, all_data,
                                              label_name='K=5 E=5 B={}'.format(pdir[1:]))

    ax.legend(loc=4, fontsize=15)
    plt.savefig('G:/实验室/实验室项目资料/联邦表情识别/experiment_results/BU3DFE_exp_acc_data_K.svg')
    plt.show()


def plot_fed_training_loss():
    """
    用于绘制联邦模型的训练过程损失函数
    :return:
    """
    files_path = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/BU3DFE_exp_acc_data_K'
    pnum_dir_l = os.listdir(files_path)
    print(pnum_dir_l)
    # 创建绘图
    fig, ax = plt.subplots()

    for pdir in pnum_dir_l:
        filep_l = os.listdir(os.path.join(files_path, pdir))
        datafile_list = []
        for i, f in enumerate(filep_l):
            file_path = os.path.join(files_path, pdir, f)
            datafile = pd.read_table(file_path, header=None)
            datafile_list.append(np.array(datafile.values[1:, 1], dtype=np.float).reshape(-1, 1))
            x_axis_index = np.array(datafile.values[1:, 0], dtype=np.int)

        # 用于替代seaborn中的tsplot绘图函数
        def tsplot(ax, x, data, label_name='', **kw):
            est = np.mean(data, axis=1)
            sd = np.std(data, axis=1)
            cis = (est - sd, est + sd)

            ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
            ax.plot(x, est, label=label_name, **kw)
            ax.tick_params(labelsize=15)
            ax.set_ylabel('Loss', size=15)
            ax.set_xlabel('Communication rounds', size=15)
            ax.margins(x=0)

            return est, sd

        xaxis_from_sets = []
        all_data = np.concatenate(datafile_list, axis=1)
        indicator_avg, indicator_std = tsplot(ax, x_axis_index, all_data,
                                              label_name='K=5 E=5 B={}'.format(pdir[1:]))

    ax.legend(loc=1, fontsize=15)
    plt.savefig('G:/实验室/实验室项目资料/联邦表情识别/experiment_results/BU3DFE_exp_loss_data_K.svg')
    plt.show()


def show_fed_vs_idpt():
    fed_result_file = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/baseline_exp_data/Fed_vs_Idpt/BU3DFE/FedNet_eval_acc.pkl'
    idpt_result_file = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/baseline_exp_data/Fed_vs_Idpt/BU3DFE/idpt_eval_acc.pkl'
    fed_data = load_pkl_file(fed_result_file)
    fed_data = [d * 100. for d in fed_data]
    idpt_data = load_pkl_file(idpt_result_file)
    fig, ax = plt.subplots()
    x = range(len(fed_data))
    labels = ['Fed model', 'Idpt model']
    # axes.stackplot(x, fed_data, idpt_data, labels=labels)
    fed_data_array = np.array(fed_data)
    idpt_data_array = np.array(idpt_data)
    # ax.scatter(x, fed_data_array, label='Fed model', s=36, alpha=0.5)
    # ax.scatter(x, idpt_data_array, label='Idpt model', s=36, alpha=0.5)

    # ax.plot(x, [fed_data_array.mean()] * len(x), label='Fed avg acc', linestyle='dashed')
    ax.plot(x, idpt_data_array, marker='.', label='Independent model')
    ax.plot(x, [86.111] * len(x), linestyle='dashed', label='Non-Fed & IID baseline')
    ax.set_xlabel('Family Number')
    ax.set_ylabel('Test accuracy(%)')
    plt.legend()
    plt.show()


def show_each_family_acc_on_FedNonIID_dataset():
    """
    用于展示每个联邦Non-IID数据集上每个家庭的测试结果
    :return:
    """
    dataset_list = ['BU3DFE', 'Jaffe', 'Oulu']
    var_dataset_NonIID_acc_list = [81.25, 60.784313, 79.53216]
    var_dataset_IID_acc_list = [87.5, 94.5946, 88.02083]

    fig, ax = plt.subplots(1, 3)
    for i, dataset in enumerate(dataset_list):
        idpt_result_file = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/' \
                           'VGG11_FedNonIID_dataset_resutls/{}/idpt_eval_acc.pkl'.format(dataset)
        idpt_data = load_pkl_file(idpt_result_file)
        x = range(len(idpt_data))
        # axes.stackplot(x, fed_data, idpt_data, labels=labels)
        idpt_data_array = np.array(idpt_data)
        # ax.scatter(x, fed_data_array, label='Fed model', s=36, alpha=0.5)
        # ax.scatter(x, idpt_data_array, label='Idpt model', s=36, alpha=0.5)

        # ax.plot(x, [fed_data_array.mean()] * len(x), label='Fed avg acc', linestyle='dashed')
        ax[i].plot(x, idpt_data_array, marker='.', label='FedNonIID dataset')
        ax[i].plot(x, [var_dataset_NonIID_acc_list[i]] * len(x), linestyle='dashed', label='Non-Fed & NonIID dataset')
        ax[i].plot(x, [var_dataset_IID_acc_list[i]] * len(x), linestyle='dashed', label='Non-Fed & IID dataset')
        ax[i].tick_params(labelsize=16)
        ax[i].set_xlabel('Client Number', size=16)
        if i == 0:
            ax[i].set_ylabel('Test accuracy(%)', size=16)
        ax[i].set_title(dataset, size=16)
    plt.legend(loc=0, prop={'size': 12})
    plt.show()


def show_various_model_test_acc():
    """
    用于绘制各个模型的测试集准确率，在独立同分布条件或者非独立同分布条件柱形图
    :return:
    """
    data_pd = pd.read_excel('G:/实验室/实验室项目资料/联邦表情识别/experiment_results/exp_results.xls',
                            sheet_name='VGG11_mixed_acc')
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=1.8)

    # Draw a nested barplot by species and sex
    g = sns.catplot(
        data=data_pd, kind="bar",
        x="dataset_condition", y="Eval acc", hue="dataset_type",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    g.despine(left=True)
    g.set_axis_labels("Data sets", "Test accuracy (%)")
    g.legend.set_title("")
    plt.show()


def look_eval_acc_saved_data():
    data_file = './results/Fed_vs_Idpt_oulu_Fed_NonIID_L4_training_mySimpleIncepRes_V2_bn_T1/0/FedNet_eval_acc.pkl'
    data = load_pkl_file(data_file)
    print(data)


def seaborn_fed_vs_idpt_on_VarDatasets_plot():
    """
    用于绘制fed与idpt模型在各个数据集上的效果对比
    :return:
    """
    dataset_list = ['BU3DFE', 'Jaffe', 'Oulu']
    df_list = []
    for dataset in dataset_list:
        fed_result_file = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/baseline_exp_data/Fed_vs_Idpt/{}' \
                          '/FedNet_eval_acc.pkl '.format(dataset)
        idpt_result_file = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/baseline_exp_data/Fed_vs_Idpt/{}/idpt_eval_acc' \
                           '.pkl '.format(dataset)

        fed_data = load_pkl_file(fed_result_file)
        # fed_data = [d for d in fed_data]
        idpt_data = load_pkl_file(idpt_result_file)
        idpt_data = [data.item() / 100 for data in idpt_data]

        fed_dict = {'FamID': [i for i in range(len(fed_data))], 'Test accuracy': fed_data,
                    'Dataset': [dataset for _ in range(len(fed_data))],
                    'Model': ['FedNet' for _ in range(len(fed_data))]}

        idpt_dict = {'FamID': [i for i in range(len(idpt_data))], 'Test accuracy': idpt_data,
                     'Dataset': [dataset for _ in range(len(idpt_data))],
                     'Model': ['IndependentNet' for _ in range(len(idpt_data))]}

        df_fed_data = pd.DataFrame(fed_dict)
        df_idpt_data = pd.DataFrame(idpt_dict)

        df_data = pd.concat([df_fed_data, df_idpt_data], axis=0, ignore_index=True)

        df_list.append(df_data)

    df_all = pd.concat(df_list, ignore_index=True)

    sns.set_theme(style="whitegrid", palette="pastel")

    # Draw a nested violinplot and split the violins for easier comparison

    sns.violinplot(data=df_all, x="Dataset", y="Test accuracy", hue="Model",
                   split=True, inner="quart", linewidth=1,
                   palette={"FedNet": "b", "IndependentNet": ".85"})
    sns.despine(left=True)

    # sns.boxplot(x="Dataset", y="Test accuracy",
    #             hue="Model", palette=["r", "b"],
    #             data=df_all)
    # sns.despine(offset=10, trim=True)

    plt.show()


def plot_ablation_exp_results():
    """
    用于绘制消融实验 (TL: transfer learning; GN: Group normalization; GC: Gradient clip)
    test accuracy
    :return:
    """
    dataset_labels = ['Oulu', 'BU3DFE', 'Jaffe']
    baseline = [55.631, 66.93767, 70.6976]
    # baseline = [57.30944, 67.66033, 65.11628]
    B_TL = [69.19795, 77.37128, 76.7442]
    B_GC = [55.88737, 61.7886, 75.81395]
    B_GN = [60.58020, 76.55826, 70.2325]
    B_TL_GC = [70.90443, 80.0813, 77.51938]
    B_GN_TL = [75.08532, 80.0813, 75.34883]
    B_GN_TL_GC = [78.58361, 84.2818, 77.67442]

    plt.style.use('fivethirtyeight')

    fig, ax = plt.subplots()

    ax.plot(dataset_labels, baseline, marker='s', ms=13, linestyle='dashed', label='Baseline')
    ax.plot(dataset_labels, B_TL, marker='o', ms=13, label='Baseline+TL')
    ax.plot(dataset_labels, B_GC, marker='v', ms=13, label='Baseline+GC')
    ax.plot(dataset_labels, B_GN, marker='^', ms=13, label='Baseline+GN')
    ax.plot(dataset_labels, B_TL_GC, marker='h', ms=13, label='Baseline+GC_TL')
    ax.plot(dataset_labels, B_GN_TL, marker='d', ms=13, label='Baseline+GN_TL')
    ax.plot(dataset_labels, B_GN_TL_GC, marker='*', ms=13, label='Baseline+GN_TL_GC')
    ax.set_xlabel('Data sets')
    ax.set_ylabel('Test accuracy (%)')
    plt.legend()
    plt.show()


def plot_ablation_exp_fam_acc():
    dataset_list = ['BU3DFE', 'Jaffe', 'Oulu']
    exp_condition_list = ['Baseline', 'Baseline+GN+TL']
    control_group = 'B+GN+TL'

    for dataset in dataset_list:
        df_list = []
        for exp_condition in exp_condition_list:
            fed_result_file = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/Ablation_exp_acc_1/' \
                              'B_vs_{}/{}/{}/FedNet_eval_acc.pkl '.format(control_group, dataset, exp_condition)
            idpt_result_file = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/Ablation_exp_acc_1/' \
                               'B_vs_{}/{}/{}/idpt_eval_acc.pkl '.format(control_group, dataset, exp_condition)

            fed_data = load_pkl_file(fed_result_file)
            # fed_data = [d for d in fed_data]
            idpt_data = load_pkl_file(idpt_result_file)
            idpt_data = [data.item() / 100 for data in idpt_data]

            fed_dict = {'FamID': [i for i in range(len(fed_data))], 'Test accuracy': fed_data,
                        'Condition': [exp_condition for _ in range(len(fed_data))],
                        'Model': ['FedNet' for _ in range(len(fed_data))]}

            idpt_dict = {'FamID': [i for i in range(len(idpt_data))], 'Test accuracy': idpt_data,
                         'Condition': [exp_condition for _ in range(len(idpt_data))],
                         'Model': ['IndependentNet' for _ in range(len(idpt_data))]}

            df_fed_data = pd.DataFrame(fed_dict)
            df_idpt_data = pd.DataFrame(idpt_dict)

            df_data = pd.concat([df_fed_data, df_idpt_data], axis=0, ignore_index=True)

            df_list.append(df_data)

        df_all = pd.concat(df_list, ignore_index=True)

        sns.set_theme(style="whitegrid", palette="pastel")
        sns.set(font_scale=1.8)

        # Draw a nested violinplot and split the violins for easier comparison

        # sns.violinplot(data=df_all, x="Dataset", y="Test accuracy", hue="Model",
        #                split=True, inner="quart", linewidth=1,
        #                palette={"FedNet": "b", "IndependentNet": ".85"})
        # sns.despine(left=True)

        sns.boxplot(x="Condition", y="Test accuracy",
                    hue="Model", palette=["r", "b"],
                    data=df_all)
        sns.despine(offset=10, trim=True)
        plt.title(dataset)
        plt.show()


def plot_ablation_exp_fed_acc():
    """
    用于绘制消融实验下，各个模型的准确率随着通信轮次的变化
    :return:
    """
    dataset_list = ['BU3DFE', 'Jaffe', 'Oulu']

    for dataset in dataset_list:
        df_list = []
        dataset_dict = dict()
        fed_log_dir = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/Ablation_exp_loss_1/{}/'.format(dataset)
        fed_log_files_list = os.listdir(fed_log_dir)
        for log_f in fed_log_files_list:
            ablation_model = log_f.split('_')[0]
            file_path = os.path.join(fed_log_dir, log_f)
            df_data = pd.read_table(file_path)

            columns = df_data.columns
            for column in columns:
                df_data.rename(columns={column: column.strip()}, inplace=True)

            dataset_dict[ablation_model] = df_data['Eval acc']

        df_all = pd.DataFrame(dataset_dict)
        sns.set_theme(style="whitegrid")
        sns.set(font_scale=1.8)
        sns.lineplot(data=df_all, palette="tab10", markers=False, dashes=True, linewidth=2.2)
        plt.xlabel('# Communication rounds')
        plt.ylabel('Test accuracy (%)')
        plt.title(dataset)

        plt.show()


def plot_ablation_exp_fed_training_loss():
    """
    用于绘制消融实验下，各个模型的training loss随着通信轮次的变化
    :return:
    """
    dataset_list = ['BU3DFE', 'Jaffe', 'Oulu']

    for dataset in dataset_list:
        df_list = []
        dataset_dict = dict()
        fed_log_dir = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/Ablation_exp_loss_1/{}/'.format(dataset)
        fed_log_files_list = os.listdir(fed_log_dir)
        for log_f in fed_log_files_list:
            ablation_model = log_f.split('_')[0]
            file_path = os.path.join(fed_log_dir, log_f)
            df_data = pd.read_table(file_path)

            columns = df_data.columns
            for column in columns:
                df_data.rename(columns={column: column.strip()}, inplace=True)

            dataset_dict[ablation_model] = df_data['loss']

        df_all = pd.DataFrame(dataset_dict)
        sns.set_theme(style="whitegrid")
        sns.set(font_scale=1.8)
        sns.lineplot(data=df_all, palette="tab10", markers=False, dashes=True, linewidth=2.2)
        plt.xlabel('# Communication rounds')
        plt.ylabel('Training loss')
        plt.title(dataset)

        plt.show()


def plot_var_fedmodel_acc():
    """
    用于绘制多个联邦Non-IID改进算法的实验，各个模型的准确率随着通信轮次的变化
    :return:
    """
    dataset_list = ['BU3DFE', 'jaffe', 'oulu']

    for dataset in dataset_list:
        files_path = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/var_methods_compare/{}/'.format(dataset)
        pnum_dir_l = os.listdir(files_path)
        print(pnum_dir_l)
        # 创建绘图
        fig, ax = plt.subplots()

        for pdir in pnum_dir_l:
            filep_l = os.listdir(os.path.join(files_path, pdir))
            datafile_list = []
            for i, f in enumerate(filep_l):
                file_path = os.path.join(files_path, pdir, f)
                datafile = pd.read_table(file_path, header=None)
                datafile_list.append(np.array(datafile.values[1:, 2], dtype=np.float).reshape(-1, 1))
                x_axis_index = np.array(datafile.values[1:, 0], dtype=np.int)

            # 用于替代seaborn中的tsplot绘图函数
            def tsplot(ax, x, data, label_name='', **kw):
                est = np.mean(data, axis=1)
                sd = np.std(data, axis=1)
                cis = (est - sd, est + sd)

                ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
                ax.plot(x, est, label=label_name, **kw)
                ax.tick_params(labelsize=13)
                ax.set_ylabel('Test Accuracy(%)', size=13)
                ax.set_xlabel('#Communication rounds', size=13)
                ax.margins(x=0)

                return est, sd

            xaxis_from_sets = []
            all_data = np.concatenate(datafile_list, axis=1)
            indicator_avg, indicator_std = tsplot(ax, x_axis_index, all_data,
                                                  label_name='{}'.format(pdir))
        ax.set_title(dataset, size=15)
        ax.legend(loc=4, fontsize=18)
        # plt.savefig('G:/实验室/实验室项目资料/联邦表情识别/experiment_results/BU3DFE_exp_acc_data_B.svg')
        plt.show()


def plot_var_fedmodel_training_loss():
    """
    用于绘制多个联邦Non-IID改进算法的实验，各个模型的 training loss 随着通信轮次的变化
    :return:
    """
    dataset_list = ['BU3DFE', 'jaffe', 'oulu']

    for dataset in dataset_list:
        files_path = 'G:/实验室/实验室项目资料/联邦表情识别/experiment_results/var_methods_compare/{}/'.format(dataset)
        pnum_dir_l = os.listdir(files_path)
        print(pnum_dir_l)
        # 创建绘图
        fig, ax = plt.subplots()

        for pdir in pnum_dir_l:
            filep_l = os.listdir(os.path.join(files_path, pdir))
            datafile_list = []
            for i, f in enumerate(filep_l):
                file_path = os.path.join(files_path, pdir, f)
                datafile = pd.read_table(file_path, header=None)
                datafile_list.append(np.array(datafile.values[1:, 1], dtype=np.float).reshape(-1, 1))
                x_axis_index = np.array(datafile.values[1:, 0], dtype=np.int)

            # 用于替代seaborn中的tsplot绘图函数
            def tsplot(ax, x, data, label_name='', **kw):
                est = np.mean(data, axis=1)
                sd = np.std(data, axis=1)
                cis = (est - sd, est + sd)

                ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
                ax.plot(x, est, label=label_name, **kw)
                ax.tick_params(labelsize=13)
                ax.set_ylabel('Training loss', size=13)
                ax.set_xlabel('#Communication rounds', size=13)
                ax.margins(x=0)

                return est, sd

            xaxis_from_sets = []
            all_data = np.concatenate(datafile_list, axis=1)
            indicator_avg, indicator_std = tsplot(ax, x_axis_index, all_data,
                                                  label_name='{}'.format(pdir))
        ax.set_title(dataset, size=15)
        ax.legend(loc=1, fontsize=20)
        # plt.savefig('G:/实验室/实验室项目资料/联邦表情识别/experiment_results/BU3DFE_exp_acc_data_B.svg')
        plt.show()


def plot_var_fedmodels_acc_results():
    """
    用于绘制消融实验 (FedAvg, FedProx, FedNova, SCAFFOLD)
    test accuracy
    :return:
    """
    dataset_labels = ['Oulu', 'BU3DFE', 'Jaffe']
    FedProx = [54.69283, 65.04065, 62.0155]
    FedAvg = [57.30944, 67.66033, 65.11628]
    FedNova = [58.95904, 63.95663, 70.54264]
    SCAFFOLD = [59.95449, 71.54471, 70.07752]
    Ours = [77.44596, 83.15266, 76.4341]
    Non_Fed_Non_IID = [80.117, 83.482, 82.353]

    plt.style.use('fivethirtyeight')

    fig, ax = plt.subplots()

    ax.plot(dataset_labels, FedAvg, marker='s', ms=13, label='FedAvg')
    ax.plot(dataset_labels, FedProx, marker='o', ms=13, label='FedProx')
    ax.plot(dataset_labels, FedNova, marker='v', ms=13, label='FedNova')
    ax.plot(dataset_labels, SCAFFOLD, marker='^', ms=13, label='SCAFFOLD')
    ax.plot(dataset_labels, Ours, marker='*', ms=13, label='Ours')
    ax.plot(dataset_labels, Non_Fed_Non_IID, marker='P', ms=13, linestyle='dashed', label='NonFed_NonIID')
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Test accuracy (%)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # plot_fed_acc()
    '''
    Tools
    '''
    # look_eval_acc_saved_data()
    '''
    For federated model
    '''
    # plot_fed_avg_acc()
    # plot_fed_training_loss()
    '''
    For compare  
    '''
    # show_fed_vs_idpt()
    # show_each_family_acc_on_FedNonIID_dataset()
    # show_various_model_test_acc()
    # seaborn_fed_vs_idpt_on_VarDatasets_plot()

    '''
    用于消融实验
    '''
    # plot_ablation_exp_results()
    plot_ablation_exp_fam_acc()

    # plot_ablation_exp_fed_acc()
    # plot_ablation_exp_fed_training_loss()
    '''
    用于不同Non-IID 联邦算法的比较结果
    '''
    # plot_var_fedmodel_acc()
    # plot_var_fedmodel_training_loss()
    # plot_var_fedmodels_acc_results()
