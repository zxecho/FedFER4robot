from PIL import Image
import numpy as np
import h5py
import torch
import transforms as transforms
import torch.utils.data as data


class CK(data.Dataset):
    """
    CK_48 famliy split Dataset.

    Set4_training 的数据结构
    group: Family0 ---> dataset: {"FEdata_pixel", "FEdata_label"}
    """

    def __init__(self, data_name, split='training', transform=None):
        self.transform = transform

        self.h5file = h5py.File('./dataset/{}'.format(data_name), 'r', driver='core')

        self.h5data = self.h5file[split]

        # self.data_list = []
        # self.labels_list = []
        # for ind in range(self.number):
        #     self.data_list.append(self.data['FEdata_pixel'][ind])
        #     self.labels_list.append(self.data['FEdata_label'][ind])
        #
        self.data_array = self.h5data['FEdata_pixel']
        self.label_array = self.h5data['FEdata_label']

        if 'Family' in split:
            self.number = self.data_array.shape[0]
        else:
            self.number = len(self.h5file[split])  # 该家庭有多少数据量

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data_array[index], self.label_array[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if len(img.shape) < 3:
            img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.data_array.shape[0]


def get_Family_dataset(args, data_name, split='training'):

    if 'training' in data_name:
        # transform = transforms.Compose([
        #     transforms.RandomCrop(args.cut_size),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        # ])
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(args.resize),
            # transforms.Grayscale(3),
            transforms.RandomCrop(args.cut_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # BU3DFE mean=[0.27676, 0.27676, 0.27676], std=[0.26701, 0.26701, 0.26701]
            # jaffe mean=[0.43192, 0.43192, 0.43192], std=[0.27979, 0.27979, 0.27979]
            # oulu mean=[0.36418, 0.36418, 0.36418], std=[0.20384, 0.20384, 0.20384]
            # ck-48  mean=[0.51194, 0.51194, 0.51194], std=[0.28913, 0.28913, 0.28913]
            transforms.Normalize(mean=args.dataset_mean_std[args.dataset]['mean'],
                                 std=args.dataset_mean_std[args.dataset]['std']),
        ])
    else:
        # transform = transforms.Compose([
        #     transforms.TenCrop(args.cut_size),
        #     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        # ])
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(args.resize),
            # transforms.ToTensor(),
            # transforms.Grayscale(3),
            transforms.TenCrop(args.cut_size),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=args.dataset_mean_std[args.dataset]['mean'],
                                                                              std=args.dataset_mean_std[args.dataset]['std'])(transforms.ToTensor()(crop))for crop in crops])),
            # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])

    h5file = h5py.File('./dataset/{}'.format(data_name), 'r', driver='core')

    fm_keys = h5file.keys()
    fm_n = len(fm_keys)

    loader_list = []
    f_n_list = []
    loader_dict = {}
    for fm_name in fm_keys:
        dataset = CK(data_name, split=fm_name, transform=transform)
        if 'training' == split:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=False)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)
        loader_list.append(dataloader)
        loader_dict[fm_name] = dataloader
        f_n_list.append(dataset.number)

    h5file.close()

    return loader_list, fm_n, loader_dict


