import torch
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing
import h5py

DATASETS_CONFIG = {
    'IndianPines': {
        'img': 'indian_pines.mat',
        'gt': 'indian_pines_gt.mat'
    },
    'Botswana': {
        'img': 'Botswana.mat',
        'gt': 'Botswana_gt.mat'
    },
    'HyRANK': {
        'img': 'HyRANK.mat',
        'gt': 'HyRANK_gt.mat'
    },
    'Houston18': {
        'img': 'Houston18.mat',
        'gt': 'Houston18_gt.mat'
    }
}


def get_dataset(dataset_name, target_folder="./dataset/", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    img_file = target_folder + dataset_name + '/' + datasets[dataset_name].get('img')
    gt_file = target_folder + dataset_name + '/' + datasets[dataset_name].get('gt')
    if dataset_name == 'IndianPines':
        img = loadmat(img_file)
        img = img['HSI_original']
        gt = loadmat(gt_file)['Data_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]
        ignored_labels = [0]

    elif dataset_name == 'Botswana':
        img = loadmat(img_file)['Botswana']
        gt = loadmat(gt_file)['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]
        ignored_labels = [0]

    elif dataset_name == 'HyRANK':
        img = loadmat(img_file)['Dioni']
        gt = loadmat(gt_file)['Dioni_GT']
        label_values = ["Undefined", "Dense urban fabric", "Mineral extraction site",
                        "Non-irrigated arable land", "Fruit trees", "Olive groves", "Broad-leaved forest",
                        "Coniferous forest", "Mixed forest", "Dense sclerophyllous vegetation",
                        "Sparce sclerophyllous vegetation", "Sparsely vegetated areas",
                        "Rocks and sand", "Water", "Coastal water"]
        ignored_labels = [0]

    elif dataset_name == 'Houston18':
        img_feature = h5py.File(img_file) 
        img = img_feature['houstonU'][:, :, :-2]
        gt = h5py.File(gt_file)['houstonU_gt'][:, :-2]
        img = img.transpose( 1,2,0)
        label_values = ["Undefined", "Healthy grass", "Stressed grass",
                        "Artificial turf", "Evergreen trees", "Deciduous trees", "Bare earth",
                        "Water", "Residential buildings", "Non-residential buildings",
                        "Roads", "Sidewalks",
                        "Crosswalks", "Major thoroughfares", "Highways", 
                        "Railways", "Paved parking lots", "Unpaved parking lots", 
                        "Cars", "Trains", "Stadium seats"]
        ignored_labels = [0]

    nan_mask = np.isnan(img.sum(axis=-1))
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)
    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:]))
    data = preprocessing.minmax_scale(data, axis=1)
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    data = scaler.fit_transform(data)
    img = data.reshape(img.shape) 
    return img, gt, label_values, ignored_labels


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.dataset_name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def get_data(data, x, y, patch_size, data_3D=True):
        x1, y1 = x - patch_size // 2, y - patch_size // 2
        x2, y2 = x1 + patch_size, y1 + patch_size
        data = data[x1:x2, y1:y2]
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        data = torch.from_numpy(data)
        if data_3D:
            data = data.unsqueeze(0)
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        data = self.get_data(self.data, x, y, self.patch_size, data_3D=False)
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        label = self.label[x1:x2, y1:y2]
        label = np.asarray(np.copy(label), dtype='int64')
        label = torch.from_numpy(label)
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        
        return data, label

class Hyper2X(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(Hyper2X, self).__init__()
        self.flip_augmentation = hyperparams['flip_augmentation']
        if hyperparams['flip_augmentation']:
            data_copy = data.copy()
            gt_copy = gt.copy()
            for i in range(1):  # range(1)生成一个positive sample
                data = np.hstack((data, data_copy))
                gt = np.hstack((gt, gt_copy))
        self.data = data
        self.label = gt
        self.classes = hyperparams['n_classes']
        self.dataset_name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.center_pixel = True
        supervision = hyperparams['supervision']
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)
        self.count = len(self.labels) // 2

    @staticmethod
    def ud_flip(data, label):
        data = np.flipud(data)
        label = np.flipud(label)
        return data, label

    @staticmethod
    def lr_flip(data, label):
        data = np.fliplr(data)
        label = np.fliplr(label)
        return data, label

    @staticmethod
    def trans_flip(data, label):
        data = data.transpose((1, 0, 2))
        label = label.transpose((1, 0))
        return data, label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        data0 = self.data[x1:x2, y1:y2]
        label0 = self.label[x1:x2, y1:y2]

        idx = i + 1
        if idx >= len(self.indices):
            idx = i - 1
        w, z = self.indices[idx]
        w1, z1 = w - self.patch_size // 2, z - self.patch_size // 2
        w2, z2 = w1 + self.patch_size, z1 + self.patch_size
        data1 = self.data[x1:x2, y1:y2]
        label1 = self.label[x1:x2, y1:y2]

        if self.flip_augmentation:
            if (i > self.count) & (i <= self.count * 2):
                data1, label1 = self.ud_flip(data0, label0)
            elif (i > self.count * 2) & (i <= self.count * 3):
                data1, label1 = self.lr_flip(data0, label0)
            elif i > self.count * 3:
                data1, label1 = self.trans_flip(data0, label0)

        data0 = np.asarray(
            np.copy(data0).transpose(
                (2, 0, 1)), dtype='float32')
        label0 = np.asarray(np.copy(label0), dtype='int64')
        data1 = np.asarray(
            np.copy(data1).transpose(
                (2, 0, 1)), dtype='float32')
        label1 = np.asarray(np.copy(label1), dtype='int64')

        data0 = torch.from_numpy(data0)
        label0 = torch.from_numpy(label0)
        data1 = torch.from_numpy(data1)
        label1 = torch.from_numpy(label1)
        if self.center_pixel and self.patch_size > 1:
            label0 = label0[self.patch_size // 2, self.patch_size // 2]
            label1 = label1[self.patch_size // 2, self.patch_size // 2]
        elif self.patch_size == 1:
            data0 = data0[:, 0, 0]
            label0 = label0[0, 0]
            data1 = data1[:, 0, 0]
            label1 = label1[0, 0]

        # data augmentation
        if self.dataset_name == 'IndianPines' or self.dataset_name == 'HyRANK':
            data0_std = torch.from_numpy(np.random.normal(1, 0.2, size=data0.size())).float()
            data0 = data0 * data0_std
            data1_std = torch.from_numpy(np.random.normal(1, 0.2, size=data0.size())).float()
            data1 = data1 * data1_std
        elif self.dataset_name == 'Houston18':
            data0_std = torch.from_numpy(np.random.normal(1, 0.1, size=data0.size())).float()
            data0 = data0 * data0_std
            data1_std = torch.from_numpy(np.random.normal(1, 0.1, size=data0.size())).float()
            data1 = data1 * data1_std
        elif self.dataset_name == 'Botswana':
            data0_std = torch.from_numpy(np.random.normal(1, 0.5, size=data0.size())).float()
            data0 = data0 * data0_std
            data1_std = torch.from_numpy(np.random.normal(1, 0.5, size=data0.size())).float()
            data1 = data1 * data1_std

        return data0, label0, data1, label1
