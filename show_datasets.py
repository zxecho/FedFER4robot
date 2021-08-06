import numpy as np
import matplotlib.pyplot as plt
import json
import os

from FedFER.function_utils import mkdir

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def show_NonIID_datasets(dataset_name):
    dataset_file_path = 'F:/zx/PycharmPro/Federated_learning/FedFER/dataset/{}'.format(dataset_name)
    files = os.listdir(dataset_file_path)
    for i, f in enumerate(files):
        fig, ax = plt.subplots()
        with open(dataset_file_path + '/' + f, 'r') as jf:
            locals()[f[:-5]] = json.load(jf)
            print(locals()[f[:-5]])
            x = []
            y = []
            for k, v in zip(locals()[f[:-5]].keys(), locals()[f[:-5]].values()):
                x.append(k)
                y.append(v)
            if 'ID' in f:
                # x = range(len(x))
                ax.set_xlabel('People ID')
            else:
                ax.set_xlabel('Facial expression label')
            ax.set_ylabel('Sample number')
            ax.stem(x, y, use_line_collection=True)
            ax.set_title(f[:-5])
            savepath = 'E:/zx/FedLearning/FedFER/datasets_overview/{}'.format(dataset_name)
            mkdir(savepath)
            plt.savefig(savepath + '/{}.svg'.format(f[:-5]))
            plt.close(fig)


show_NonIID_datasets('jaffe_Fed_NonIID')  # 'oulu_Fed_NonIID', 'BU3DFE_Fed_NonIID', 'CK_Fed_NonIID', 'jaffe_Fed_NonIID'
