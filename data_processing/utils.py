import itertools
import re
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import logging
import os
import seaborn as sns
import spectral
from sklearn.decomposition import PCA
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore", category=Warning)


def PCA_data(data, dataset, components=10):
    model_dir = 'pca_data/'+dataset+'48.npy'
    if not os.path.isfile(model_dir):
#        os.makedirs(model_dir, exist_ok=True)
        new_datawithlabel_list = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                c2l = list(data[i][j])
                c2l.append(data[i][j])
                new_datawithlabel_list.append(c2l)
        new_datawithlabel_array = np.array(new_datawithlabel_list)
        pca = PCA(n_components=components)
        pca.fit(new_datawithlabel_array[:,:-1],new_datawithlabel_array[:,-1])
        pca_data = pca.transform(new_datawithlabel_array[:,:-1])
        new_pca_data = np.zeros((data.shape[0],data.shape[1],components))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                new_pca_data[i][j] = pca_data[i*data.shape[1]+j]
        np.save(file=model_dir, arr=new_pca_data)
    else:
        new_pca_data = np.load(file=model_dir)
    return new_pca_data


def logger(logfile_name='test_log/logs.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(filename=logfile_name)
    fileHandler.setLevel(logging.DEBUG)

    #    formatter = logging.Formatter("%(asctime)s-%(filename)s-%(message)s", "%Y-%m-%d-%H-%M")
    formatter = logging.Formatter("%(message)s")
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    logger.debug('creating {}'.format(logfile_name))

    return logger


def show_plot(iteration, loss):
    import matplotlib.pyplot as plt
    plt.plot(iteration, loss)
    plt.show()


def show_inference(pre_data, label, args):
    from spectral import imshow
    import matplotlib.pyplot as plt
    pred_matrix = np.zeros((args.size[0], args.size[1]))
    count = 0
    for i in range(args.size[0]):
        for j in range(args.size[1]):
            if label[i][j] != 0:
                pred_matrix[i][j] = pre_data[count]
                count += 1

    # save_rgb(args.save_fig_name, pred_matrix, colors=spy_colors)
    imshow(classes=pred_matrix.astype(int))
    plt.savefig(args.save_fig_name)
    return pred_matrix

def sample_gt(gt, train_size=None, mode='', sample_nums=5):
    indices = np.nonzero(gt) 
    X = list(zip(*indices))  # x,y features
    y = gt[indices].ravel()  # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
        train_size = int(train_size)

    if mode == 'random':
        train_indices, test_indices = train_test_split(X, train_size=train_size, stratify=y)
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[train_indices] = gt[train_indices]
        test_gt[test_indices] = gt[test_indices]
    elif mode == 'fixed':
        train_indices, test_indices = [], []
        for c in np.unique(gt):
            if c == 0:
                continue 
            indices = np.nonzero(gt == c)
            X = list(zip(*indices))  # x,y features
            if sample_nums / len(X) > 1:
                X_copy = X.copy()
                for i in range(sample_nums // len(X)): 
                    X += X_copy
            train, test = train_test_split(X, train_size=sample_nums / len(X))
            train_indices += train
            test_indices += test
        train_indices = [list(t) for t in zip(*train_indices)]
        test_indices = [list(t) for t in zip(*test_indices)]
        train_gt[train_indices] = gt[train_indices]
        test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            # gt.shape[0]为145
            for x in range(gt.shape[0]):
                # 0到145遍历
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True  
    ignored_mask = ~ignored_mask 
    # target = target[ignored_mask] -1
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes
    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results


def build_dataset(mat, gt, ignored_labels=None):
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.array(samples).astype(float), np.array(labels).astype(float)


def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.
    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)
    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format
    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        palette = {0: (0, 0, 0)}
        flatui = [ "#7A80A4", "#0a5757", "#1DA96C", "#9FD06C", "#05B8E1",
            "#7F655E", "#FDA190", "#4A4D68", "#D1E0E9", "#C4C1C5", "#F2D266",
            "#B15546", "#CE7452", "#A59284", "#DFD2A3", "#F9831A","#000000"] 
        for k, color in enumerate(sns.color_palette(flatui)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def display_goundtruth(gt, vis, caption=""):
    color_gt = convert_to_color_(gt)
    vis.images([np.transpose(color_gt, (2, 0, 1))], opts={'caption': caption})
    return color_gt

def display_dataset(img, vis):
    rgb = spectral.get_rgb(img, [29, 19, 9])
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    caption = "Data set ground truth"
    # send to visdom server
    vis.images([np.transpose(rgb, (2, 0, 1))],
               opts={'caption': caption})


def build_dataset(mat, gt, ignored_labels=None):
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.array(samples).astype(float), np.array(labels).astype(float)
    

def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def sliding_window(image, step=1, window_size=(9, 9), with_data=True):
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    # W2, H2 = spa_image.shape[:2]
#    offset_w = (W - w) % step
#    offset_h = (H - h) % step
    for x in range(0, W - w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + 1, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(img, step=1, window_size=(9, 9)):
    sw = sliding_window(img, step, window_size=window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def get_device(logger, ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    n_classes = np.max(ground_truth) if n_classes is None else n_classes # 17
    weights = np.zeros(n_classes) 
    frequencies = np.zeros(n_classes) 
    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)
    frequencies /= np.sum(frequencies) # 307
    # Obtain the median on non-zero frequencies
    #  (array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]),)
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()