import os
from pickle import FALSE
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class BaseDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all paths in memory"""

    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = Image.open(self.images[index]).convert('RGB')
        x = self.transform(x)
        y = self.labels[index]
        return x, y


def get_data(path, num_tasks, nc_first_task, validation, shuffle_classes, class_order=None,lt = False,ltio =False ):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []
    clsanalysis={}

    # read filenames and labels
    trn_lines = np.loadtxt(os.path.join(path, 'train.txt'), dtype=str)
    tst_lines = np.loadtxt(os.path.join(path, 'test.txt'), dtype=str)
    if class_order is None:
        num_classes = len(np.unique(trn_lines[:, 1]))
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    if shuffle_classes:
        np.random.shuffle(class_order)

    # compute classes per task and num_tasks
    if nc_first_task is None:
        cpertask = np.array([num_classes // num_tasks] * num_tasks)
        for i in range(num_classes % num_tasks):
            cpertask[i] += 1
    else:
        assert nc_first_task < num_classes, "first task wants more classes than exist"
        remaining_classes = num_classes - nc_first_task
        assert remaining_classes >= (num_tasks - 1), "at least one class is needed per task"  # better minimum 2
        cpertask = np.array([nc_first_task] + [remaining_classes // (num_tasks - 1)] * (num_tasks - 1))
        for i in range(remaining_classes % (num_tasks - 1)):
            cpertask[i + 1] += 1

    assert num_classes == cpertask.sum(), "something went wrong, the split does not match num classes"
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': []}
        data[tt]['val'] = {'x': [], 'y': []}
        data[tt]['tst'] = {'x': [], 'y': []}
        clsanalysis[tt] = np.zeros(cpertask[tt])
    
    #TRAIN ANALYSIS
    num_per_cls = np.zeros(num_classes)
    for this_image, this_label in trn_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        this_label = class_order.index(this_label)
        this_task = (this_label >= cpertask_cumsum).sum()
        num_per_cls[this_label]+=1
    img_num_per_cls = get_img_num_per_cls(num_per_cls,'exp')
    if lt:
        np.random.shuffle(img_num_per_cls)
    num_per_cls_now = np.zeros(num_classes)
    # ALL OR TRAIN
    for this_image, this_label in trn_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        if num_per_cls_now[this_label] >= img_num_per_cls[this_label] and (ltio or lt):
            continue
        else:
            clsanalysis[this_task][this_label - init_class[this_task]]+=1
            data[this_task]['trn']['x'].append(this_image)
            data[this_task]['trn']['y'].append(this_label - init_class[this_task])
            num_per_cls_now[this_label] += 1
    # ALL OR TEST
    for this_image, this_label in tst_lines:
        if not os.path.isabs(this_image):
            this_image = os.path.join(path, this_image)
        this_label = int(this_label)
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        data[this_task]['tst']['x'].append(this_image)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.round(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][rnd_img[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, class_order


def get_img_num_per_cls(num_per_cls,imb_type,imb_factor = 0.01):
    img_max = num_per_cls[0]
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(len(num_per_cls)):
            num = img_max * (imb_factor**(cls_idx / (len(num_per_cls)- 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(len(num_per_cls)// 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(len(num_per_cls) // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    elif imb_type == 'fewshot':
        for cls_idx in range(len(num_per_cls)):
            if cls_idx<50:
                num = img_max
            else:
                num = img_max*0.01
            img_num_per_cls.append(int(num))
    else:
        img_num_per_cls.extend([int(img_max)] * len(num_per_cls))
    return img_num_per_cls

