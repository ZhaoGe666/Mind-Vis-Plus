
import os
import csv
import json
import copy
import torch
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset
import torchvision.transforms as transforms




class GOD_dataset(Dataset):
    def __init__(self, dir = './data/Kamitani/npz',
                 subset = 'train',
                 roi = 'VC',
                 subjects = ['sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5'], 
                 patch_size=16,
                 return_more_class_info = False,
                 test_category=None, 
                 include_nonavg_test=False):
        super().__init__()

        self.dir = dir
        self.subset = subset
        self.roi = roi
        self.subjects = subjects
        self.patch_size = patch_size
        self.return_more_class_info = return_more_class_info

        self.fmri, self.image, self.label_all, self.sub_idx = self._load_data(dir=self.dir, 
                                                                subset=self.subset, 
                                                                roi = self.roi,
                                                                subjects = self.subjects,
                                                                patch_size = self.patch_size,
                                                                test_category=test_category, 
                                                                include_nonavg_test=include_nonavg_test)
        
        self.num_voxels = self.fmri.shape[-1]
        # self.num_per_sub = self.fmri.shape[] 

        if len(self.image) != len(self.fmri):
            self.image = np.repeat(self.image, 35, axis=0)

        self.img_class = [i[0] for i in self.label_all]  # class_in_1000 
        self.img_class_name = [i[1] for i in self.label_all]  # description
        self.naive_label = [i[2] for i in self.label_all]  # subset_rank


    def _load_data(self, dir, subset, roi, subjects, patch_size,
                   test_category=None, include_nonavg_test=False):
        # 'test'-->'valid'
        img_npz = dict(np.load(os.path.join(dir, 'images_256.npz'))) # image data: 1200 'train_images' and 50 'test_images'
        with open(os.path.join(dir, 'imagenet_class_index.json'), 'r') as f:
            img_class_index = json.load(f)  # a dict of length 1000 (classes) e.g. '34':['n01665541', 'n01664990', 'leatherback_turtle']
        # key 为类别索引, value 中 第1项、第2项（如有）为文件名前缀, 最后一项为类别str
        with open(os.path.join(dir, 'imagenet_training_label.csv'), 'r') as f:
            csvreader = csv.reader(f)  # 0003: a list of length 1200 e.g. ['1518878.014910', 'n01518878_14910.JPEG'], 一个文件民可能对应多个__xx
            img_training_filename = [row for row in csvreader]
        # 两个元素均为文件名
        with open(os.path.join(dir, 'imagenet_testing_label.csv'), 'r') as f:
            csvreader = csv.reader(f)  # length 50
            img_testing_filename = [row for row in csvreader]

        train_img_label, naive_label_set = self.get_img_label(img_class_index, img_training_filename)
        test_img_label, _ = self.get_img_label(img_class_index, img_testing_filename, naive_label_set)
        # 0008: (31, 'tree_frog', 1) label分别为ImageNet 1000类中的(class_in_1000, description, subset_rank)
        test_img = [] # img_npz['test_images']
        train_img = [] # img_npz['train_images']
        train_fmri = []
        test_fmri = []
        train_img_label_all = []
        test_img_label_all = []
        train_sub_idx = []
        test_sub_idx = []
        for sub_idx, sub in enumerate(subjects):
            npz = dict(np.load(os.path.join(dir, f'{sub}.npz')))  # fMRI data
            test_img.append(img_npz['test_images'])
            train_img.append(img_npz['train_images'][npz['arr_3']])  # 'arr_3' 似乎是 fMRI 对应的 train_images 的位置索引 (0~1199)
            train_lb = [train_img_label[i] for i in npz['arr_3']]  # 按照位置索引取label
            test_lb = test_img_label
            
            roi_mask = npz[roi]
            tr = npz['arr_0'][..., roi_mask]  # train fMRI
            tt = npz['arr_2'][..., roi_mask]  # test fMRI ,'arr_1'共35*50=1750条数据, 'arr_2'是每个样本35个数据的平均
            if include_nonavg_test:
                tt = np.concatenate([tt, npz['arr_1'][..., roi_mask]], axis=0)

            # train_fmri.append(tr[..., :tr.shape[-1] - tr.shape[-1] % patch_size])
            # test_fmri.append(tt[..., :tt.shape[-1] - tt.shape[-1] % patch_size])
            tr = self.normalize(self.pad_to_patch_size(tr, patch_size))
            tt = self.normalize(self.pad_to_patch_size(tt, patch_size), np.mean(tr), np.std(tr))  #  用训练集的均值和标准差norm测试集
            train_fmri.append(tr)
            test_fmri.append(tt)
            if test_category is not None:
                train_img_, train_fmri_, test_img_, test_fmri_, train_lb, test_lb = self.reorganize_train_test(train_img[-1], train_fmri[-1], 
                                                                test_img[-1], test_fmri[-1], train_lb, test_lb,
                                                                test_category, npz['arr_3'])
                train_img[-1] = train_img_
                train_fmri[-1] = train_fmri_
                test_img[-1] = test_img_
                test_fmri[-1] = test_fmri_
            
            train_img_label_all += train_lb
            train_sub_idx += [sub_idx]*len(train_lb)
            test_img_label_all += test_lb
            test_sub_idx += [sub_idx]*len(test_lb)

        len_max = max([i.shape[-1] for i in test_fmri])
        test_fmri = [np.pad(i, ((0, 0),(0, len_max-i.shape[-1])), mode='wrap') for i in test_fmri]  # 不同sub可能 num_voxel不一样?
        train_fmri = [np.pad(i, ((0, 0),(0, len_max-i.shape[-1])), mode='wrap') for i in train_fmri]

        # len_min = min([i.shape[-1] for i in test_fmri])
        # test_fmri = [i[:,:len_min] for i in test_fmri]
        # train_fmri = [i[:,:len_min] for i in train_fmri]


        test_fmri = np.concatenate(test_fmri, axis=0)
        train_fmri = np.concatenate(train_fmri, axis=0)
        test_img = np.concatenate(test_img, axis=0)
        train_img = np.concatenate(train_img, axis=0)
        
        # test_img = rearrange(test_img, 'n h w c -> n c h w')
        # train_img = rearrange(train_img, 'n h w c -> n c h w')
        if subset == 'train':
            return train_fmri, train_img, train_img_label_all, train_sub_idx
        elif subset == 'valid':
            return test_fmri, test_img, test_img_label_all, test_sub_idx
        elif subset == 'test':
            raise Exception('fool! it is called validation set!')

    def image_transform(self, img):
        if self.subset == 'train':
            random_crop = transforms.RandomCrop(size=int(0.8*256)) if torch.rand(1)<0.5 else self.identity
            transform_func = transforms.Compose([self.img_normalize,
                                                    random_crop,
                                                    transforms.Resize((256, 256)), 
                                                    self.channel_last])
        elif self.subset == 'valid':
            transform_func = transforms.Compose([self.img_normalize, 
                                            transforms.Resize((256, 256)), 
                                            self.channel_last])
        return transform_func(img)
        # return img

    def img_normalize(self,img):
        if img.shape[-1] == 3:
            img = rearrange(img, 'h w c -> c h w')
        img = torch.tensor(img)
        img = img * 2.0 - 1.0 # to -1 ~ 1
        return img

    def channel_last(self, img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')
    
    def fmri_transform(self, x, sparse_rate=0.2):
        # x: 1, num_voxels
        x_aug = copy.deepcopy(x)
        idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
        x_aug[idx] = 0
        return torch.FloatTensor(x_aug)

    def identity(self, x):
        return x
    
    def get_img_label(self, class_index:dict, img_filename:list, naive_label_set=None):
        img_label = []
        wind = []
        desc = []
        for _, v in class_index.items():  # 在GOD中, class_index来自ImageNet 1000类
            n_list = []
            for n in v[:-1]:
                n_list.append(int(n[1:]))
            wind.append(n_list)  # 来自WordNet的index,由于通用描述(description)的多义词特点,一个大类别可能对应多个子类
            desc.append(v[-1])  # description 通用类别的词汇描述
        # GOD中只有1250张图, 200类(训练集1200图, 150类)
        naive_label = {} if naive_label_set is None else naive_label_set
        for _, file in enumerate(img_filename):  # 给 1200 个文件生成对应label
            name = int(file[0].split('.')[0])
            naive_label[name] = []  # 外层循环更新一个字典，ImageNet类别index为key,重复的类别不增加字典的长度,其实可以用set来实现
            nl = list(naive_label.keys()).index(name)  # 每次循环按照ImageNet类别index寻找其在当前子集(如,training set)中类别的index
            # 所以nl为子集类别index,可以统计类别数量
            for c, (w, d) in enumerate(zip(wind, desc)):
                if name in w:
                    img_label.append((c, d, nl))
                    break
        return img_label, naive_label

    def pad_to_patch_size(self, x, patch_size):
        assert x.ndim == 2
        # voxel 后面 wrap padding到16的倍数
        # warp padding意义不明 https://medium.com/@Orca_Thunder/image-padding-techniques-wrap-padding-part-3-231da96ef20a
        return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap') 

    def normalize(self, x, mean=None, std=None):
        mean = np.mean(x) if mean is None else mean
        std = np.std(x) if std is None else std
        return (x - mean) / (std * 1.0)
    
    def reorganize_train_test(self, train_img, train_fmri, test_img, test_fmri, train_lb, test_lb, 
                        test_category, train_index_lookup):
        test_img_ = []
        test_fmri_ = []
        test_lb_ = []
        train_idx_list = []
        num_per_category = 8
        for c in test_category:
            c_idx = c * num_per_category + np.random.choice(num_per_category, 1)[0]
            train_idx = train_index_lookup[c_idx]
            test_img_.append(train_img[train_idx])
            test_fmri_.append(train_fmri[train_idx])
            test_lb_.append(train_lb[train_idx])
            train_idx_list.append(train_idx)
        
        train_img_ = np.stack([img for i, img in enumerate(train_img) if i not in train_idx_list])
        train_fmri_ = np.stack([fmri for i, fmri in enumerate(train_fmri) if i not in train_idx_list])
        train_lb_ = [lb for i, lb in enumerate(train_lb) if i not in train_idx_list] + test_lb

        train_img_ = np.concatenate([train_img_, test_img], axis=0)
        train_fmri_ = np.concatenate([train_fmri_, test_fmri], axis=0)

        test_img_ = np.stack(test_img_)
        test_fmri_ = np.stack(test_fmri_)
        return train_img_, train_fmri_, test_img_, test_fmri_, train_lb_, test_lb_

    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, index):
        fmri = self.fmri[index]
        if index >= len(self.image):  # TODO: 没有图像对应的fmri?
            img = np.zeros_like(self.image[0])
        else:
            img = self.image[index] / 255.0
        fmri = np.expand_dims(fmri, axis=0) # (1, num_voxels)
        if self.return_more_class_info:
            img_class = self.img_class[index]
            img_class_name = self.img_class_name[index]
            naive_label = torch.tensor(self.naive_label[index])
            return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img), 'subject': self.sub_idx[index],
                    'image_class': img_class, 'image_class_name': img_class_name, 'naive_label':naive_label}
        else:
            return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img), 'subject': self.sub_idx[index]}
        


if __name__ == '__main__':
    train_set = GOD_dataset(subset='train')
