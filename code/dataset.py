
from torch.utils.data import Dataset
import numpy as np
import os
from scipy import interpolate
from einops import rearrange
import json
import csv
import torch
from pathlib import Path
import torchvision.transforms as transforms

def identity(x):
    return x
def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    # voxel 后面 wrap padding到16的倍数
    # FIXME: warp padding意义不明 https://medium.com/@Orca_Thunder/image-padding-techniques-wrap-padding-part-3-231da96ef20a
    return np.pad(x, ((0,0),(0, patch_size-x.shape[1]%patch_size)), 'wrap') 

def pad_to_length(x, length):
    assert x.ndim == 3
    assert x.shape[-1] <= length
    if x.shape[-1] == length:
        return x

    return np.pad(x, ((0,0),(0,0), (0, length - x.shape[-1])), 'wrap')

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def process_voxel_ts(v, p, t=8):
    '''
    处理每一个subject数据, 结果为 1D vector:
    (1) augmentation: 在时间上均匀采样取更小时间序列 8s-->10 frames-->120 倍
    (2) average across time
    (3) padding到16的整数倍
   
    v: voxel timeseries of a subject. (1200, num_voxels)
    p: patch size
    t: time step of the averaging window for v. Kamitani used 8 ~ 12s
    return: voxels_reduced. reduced for the alignment of the patch size (num_samples, num_voxels_reduced)

    '''
    # average the time axis first
    num_frames_per_window = t // 0.75 # ~0.75s per frame in HCP  
    # 8/0.75 ~= 10.667 = 10
    # FIXME: 这里有错误，HCP 1200 release resting-fMRI 的 TR 为 720ms

    v_split = np.array_split(v, len(v) // num_frames_per_window, axis=0) 
    # list-->len=1200/10=120, 共120个split，每个split为(10, num_voxels)
    # FIXME: 均分array可以直接用np.split

    v_split = np.concatenate([np.mean(f,axis=0).reshape(1,-1) for f in v_split],axis=0) # (120, num_voxels)
    # pad the num_voxels
    # v_split = np.concatenate([v_split, np.zeros((v_split.shape[0], p - v_split.shape[1] % p))], axis=-1)
    v_split = pad_to_patch_size(v_split, p)  # num_voxel填充到16的整数倍
    v_split = normalize(v_split)  # Z-score norm 到 0-1分布
    return v_split

def augmentation(data, aug_times=2, interpolation_ratio=0.5):
    '''
    空间插值增广
    data: num_samples, num_voxels_padded
    return: data_aug: num_samples*aug_times, num_voxels_padded
    '''
    num_to_generate = int((aug_times-1)*len(data)) 
    if num_to_generate == 0:
        return data
    pairs_idx = np.random.choice(len(data), size=(num_to_generate, 2), replace=True)
    data_aug = []
    for i in pairs_idx:
        z = interpolate_voxels(data[i[0]], data[i[1]], interpolation_ratio)
        data_aug.append(np.expand_dims(z,axis=0))
    data_aug = np.concatenate(data_aug, axis=0)

    return np.concatenate([data, data_aug], axis=0)

def interpolate_voxels(x, y, ratio=0.5):
    ''''
    x, y: one dimension voxels array
    ratio: ratio for interpolation
    return: z same shape as x and y

    '''
    values = np.stack((x,y))
    points = (np.r_[0, 1], np.arange(len(x)))
    xi = np.c_[np.full((len(x)), ratio), np.arange(len(x)).reshape(-1,1)]
    z = interpolate.interpn(points, values, xi)
    return z

def img_norm(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = (img / 255.0) * 2.0 - 1.0 # to -1 ~ 1
    return img

def channel_first(img):
        if img.shape[-1] == 3:
            return rearrange(img, 'h w c -> c h w')
        return img

class hcp_dataset(Dataset):
    def __init__(self, path='../data/HCP/npz', roi='VC', patch_size=16, transform=identity, aug_times=2, 
                num_sub_limit=None, include_kam=False, include_hcp=True):
        super(hcp_dataset, self).__init__()
        data = []
        images = []
        
        if include_hcp:
            for c, sub in enumerate(os.listdir(path)):  # 有多个subject
                if os.path.isfile(os.path.join(path,sub,'HCP_visual_voxel.npz')) == False:
                    continue 
                if num_sub_limit is not None and c > num_sub_limit:  # 为None时则使用的subject数量没有限制；config里默认为None
                    break
                npz = dict(np.load(os.path.join(path,sub,'HCP_visual_voxel.npz')))
                voxels = np.concatenate([npz['V1'],npz['V2'],npz['V3'],npz['V4']], axis=-1) if roi == 'VC' else npz[roi] # 1200, num_voxels
                voxels = process_voxel_ts(voxels, patch_size) # num_samples, num_voxels_padded
                # Notice: num_samples 是截取后的样本数 
                data.append(voxels)
                # 能直接 padding 说明每个subject 的 voxel数量一致
                
            data = augmentation(np.concatenate(data, axis=0), aug_times) # num_samples, num_voxels_padded
            # 默认不augmentation
            data = np.expand_dims(data, axis=1) # num_samples, 1, num_voxels_padded
            images += [None] * len(data)

        if include_kam:
            kam_path = os.path.join(str(Path(path).parent.parent), 'Kamitani', 'npz')
            k = Kamitani_pretrain_dataset(kam_path, roi, patch_size, transform, aug_times)
            if len(data) != 0:
                padding_len = max([data.shape[-1],  k.data.shape[-1]])
                data = pad_to_length(data, padding_len)
                data_k = pad_to_length(k.data, padding_len)
                data = np.concatenate([data, data_k], axis=0)
            else:
                data = k.data
            images += k.images

        assert len(data) != 0, 'No data found'
        
        self.roi = roi
        self.patch_size = patch_size
        self.num_voxels = data.shape[-1]
        self.data = data
        self.transform = transform
        self.images = images
        self.images_transform = transforms.Compose([
                                            img_norm,
                                            transforms.Resize((112, 112)), 
                                            channel_first
                                        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.images[index]
        images_transform = self.images_transform if img is not None else identity
        img = img if img is not None else torch.zeros(3, 112, 112)

        return {'fmri': self.transform(self.data[index]),
                'image': images_transform(img)}
       
class Kamitani_pretrain_dataset(Dataset):
    def __init__(self, path='../data/Kamitani/npz', roi='VC', patch_size=16, transform=identity, aug_times=2):
        super(Kamitani_pretrain_dataset, self).__init__()
        k1, k2 = create_Kamitani_dataset(path, roi, patch_size, transform, include_nonavg_test=True)
        # data = np.concatenate([k1.fmri, k2.fmri], axis=0)
        # self.images = [img for img in k1.image] + [None] * len(k2.fmri)

        data = k1.fmri
        self.images = [(img*255.0).astype(np.uint8) for img in k1.image]

        # data = augmentation(data, aug_times)
        self.data = np.expand_dims(data, axis=1)
        self.roi = roi
        self.patch_size = patch_size
        self.num_voxels = data.shape[-1]
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.transform(self.data[index])

def get_img_label(class_index:dict, img_filename:list, naive_label_set=None):
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

def create_Kamitani_dataset(path='../data/Kamitani/npz',  roi='VC', patch_size=16, fmri_transform=identity,
            image_transform=identity, subjects = ['sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5'], 
            test_category=None, include_nonavg_test=False):
    img_npz = dict(np.load(os.path.join(path, 'images_256.npz'))) # image data: 1200 'train_images' and 50 'test_images'
    with open(os.path.join(path, 'imagenet_class_index.json'), 'r') as f:
        img_class_index = json.load(f)  # a dict of length 1000 (classes) e.g. '34':['n01665541', 'n01664990', 'leatherback_turtle']
    # key 为类别索引, value 中 第1项、第2项（如有）为文件名前缀, 最后一项为类别str
    with open(os.path.join(path, 'imagenet_training_label.csv'), 'r') as f:
        csvreader = csv.reader(f)  # 0003: a list of length 1200 e.g. ['1518878.014910', 'n01518878_14910.JPEG'], 一个文件民可能对应多个__xx
        img_training_filename = [row for row in csvreader]
    # 两个元素均为文件名
    with open(os.path.join(path, 'imagenet_testing_label.csv'), 'r') as f:
        csvreader = csv.reader(f)  # length 50
        img_testing_filename = [row for row in csvreader]

    train_img_label, naive_label_set = get_img_label(img_class_index, img_training_filename)
    test_img_label, _ = get_img_label(img_class_index, img_testing_filename, naive_label_set)
    # 0008: (31, 'tree_frog', 1) label分别为ImageNet 1000类中的(class_in_1000, description, subset_rank)
    test_img = [] # img_npz['test_images']
    train_img = [] # img_npz['train_images']
    train_fmri = []
    test_fmri = []
    train_img_label_all = []
    test_img_label_all = []
    for sub in subjects:
        npz = dict(np.load(os.path.join(path, f'{sub}.npz')))  # fMRI data
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
        tr = normalize(pad_to_patch_size(tr, patch_size))
        tt = normalize(pad_to_patch_size(tt, patch_size), np.mean(tr), np.std(tr))  #  用训练集的均值和标准差norm测试集
        train_fmri.append(tr)
        test_fmri.append(tt)
        if test_category is not None:
            train_img_, train_fmri_, test_img_, test_fmri_, train_lb, test_lb = reorganize_train_test(train_img[-1], train_fmri[-1], 
                                                            test_img[-1], test_fmri[-1], train_lb, test_lb,
                                                            test_category, npz['arr_3'])
            train_img[-1] = train_img_
            train_fmri[-1] = train_fmri_
            test_img[-1] = test_img_
            test_fmri[-1] = test_fmri_
        
        train_img_label_all += train_lb
        test_img_label_all += test_lb

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
    num_voxels = train_fmri.shape[-1]

    # test_img = rearrange(test_img, 'n h w c -> n c h w')
    # train_img = rearrange(train_img, 'n h w c -> n c h w')

    if isinstance(image_transform, list):
        return (Kamitani_dataset(train_fmri, train_img, train_img_label_all, fmri_transform, image_transform[0], num_voxels, len(npz['arr_0'])), 
                Kamitani_dataset(test_fmri, test_img, test_img_label_all, torch.FloatTensor, image_transform[1], num_voxels, len(npz['arr_2'])))
    else:
        return (Kamitani_dataset(train_fmri, train_img, train_img_label_all, fmri_transform, image_transform, num_voxels, len(npz['arr_0'])), 
                Kamitani_dataset(test_fmri, test_img, test_img_label_all, torch.FloatTensor, image_transform, num_voxels, len(npz['arr_2'])))

def reorganize_train_test(train_img, train_fmri, test_img, test_fmri, train_lb, test_lb, 
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

class Kamitani_dataset(Dataset):
    def __init__(self, fmri, image, img_label, fmri_transform=identity, image_transform=identity, num_voxels=0, num_per_sub=50):
        super(Kamitani_dataset, self).__init__()
        self.fmri = fmri
        self.image = image
        if len(self.image) != len(self.fmri):
            self.image = np.repeat(self.image, 35, axis=0)
        self.fmri_transform = fmri_transform
        self.image_transform = image_transform
        self.num_voxels = num_voxels
        self.num_per_sub = num_per_sub
        self.img_class = [i[0] for i in img_label]  # class_in_1000 
        self.img_class_name = [i[1] for i in img_label]  # description
        self.naive_label = [i[2] for i in img_label]  # subset_rank
        self.return_image_class_info = False

    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, index):
        fmri = self.fmri[index]
        if index >= len(self.image):  # FIXME: 为什么会超出索引?
            img = np.zeros_like(self.image[0])
        else:
            img = self.image[index] / 255.0
        fmri = np.expand_dims(fmri, axis=0) # (1, num_voxels)
        if self.return_image_class_info:
            img_class = self.img_class[index]
            img_class_name = self.img_class_name[index]
            naive_label = torch.tensor(self.naive_label[index])
            return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img),
                    'image_class': img_class, 'image_class_name': img_class_name, 'naive_label':naive_label}
        else:
            return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img)}

class base_dataset(Dataset):
    def __init__(self, x, y=None, transform=identity):
        super(base_dataset, self).__init__()
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        if self.y is None:
            return self.transform(self.x[index])
        else:
            return self.transform(self.x[index]), self.transform(self.y[index])
    
def remove_repeats(fmri, img_lb):
    assert len(fmri) == len(img_lb), 'len error'
    fmri_dict = {}
    for f, lb in zip(fmri, img_lb):
        if lb in fmri_dict.keys():
            fmri_dict[lb].append(f)
        else:
            fmri_dict[lb] = [f]
    lbs = []
    fmris = []
    for k, v in fmri_dict.items():
        lbs.append(k)
        fmris.append(np.mean(np.stack(v), axis=0))
    return np.stack(fmris), lbs

def get_stimuli_list(root, sub):
    sti_name = []
    path = os.path.join(root, 'Stimuli_Presentation_Lists', sub)
    folders = os.listdir(path)
    folders.sort()
    for folder in folders:
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        files = os.listdir(os.path.join(path, folder))
        files.sort()
        for file in files:
            if file.endswith('.txt'):
                sti_name += list(np.loadtxt(os.path.join(path, folder, file), dtype=str))

    sti_name_to_return = []
    for name in sti_name:
        if name.startswith('rep_'):
            name = name.replace('rep_', '', 1)
        sti_name_to_return.append(name)
    return sti_name_to_return

def list_get_all_index(list, value):
    return [i for i, v in enumerate(list) if v == value]
    
def create_BOLD5000_dataset(path='../data/BOLD5000', patch_size=16, fmri_transform=identity,
            image_transform=identity, subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4'], include_nonavg_test=False):
    roi_list = ['EarlyVis', 'LOC', 'OPA', 'PPA', 'RSC']
    fmri_path = os.path.join(path, 'BOLD5000_GLMsingle_ROI_betas/py')
    img_path = os.path.join(path, 'BOLD5000_Stimuli')
    imgs_dict = np.load(os.path.join(img_path, 'Scene_Stimuli/Presented_Stimuli/img_dict.npy'),allow_pickle=True).item()
    repeated_imgs_list = np.loadtxt(os.path.join(img_path, 'Scene_Stimuli', 'repeated_stimuli_113_list.txt'), dtype=str)

    fmri_files = [f for f in os.listdir(fmri_path) if f.endswith('.npy')]
    fmri_files.sort()
    
    fmri_train_major = []
    fmri_test_major = []
    img_train_major = []
    img_test_major = []
    for sub in subjects:
        # load fmri
        fmri_data_sub = []
        for roi in roi_list:
            for npy in fmri_files:
                if npy.endswith('.npy') and sub in npy and roi in npy:
                    fmri_data_sub.append(np.load(os.path.join(fmri_path, npy)))
        fmri_data_sub = np.concatenate(fmri_data_sub, axis=-1) # concatenate all rois
        fmri_data_sub = normalize(pad_to_patch_size(fmri_data_sub, patch_size))
      
        # load image
        img_files = get_stimuli_list(img_path, sub)
        img_data_sub = [imgs_dict[name] for name in img_files]
        
        # split train test
        test_idx = [list_get_all_index(img_files, img) for img in repeated_imgs_list]
        test_idx = [i for i in test_idx if len(i) > 0] # remove empy list for CSI4
        test_fmri = np.stack([fmri_data_sub[idx].mean(axis=0) for idx in test_idx])
        test_img = np.stack([img_data_sub[idx[0]] for idx in test_idx])
        
        test_idx_flatten = []
        for idx in test_idx:
            test_idx_flatten += idx # flatten
        if include_nonavg_test:
            test_fmri = np.concatenate([test_fmri, fmri_data_sub[test_idx_flatten]], axis=0)
            test_img = np.concatenate([test_img, np.stack([img_data_sub[idx] for idx in test_idx_flatten])], axis=0)

        train_idx = [i for i in range(len(img_files)) if i not in test_idx_flatten]
        train_img = np.stack([img_data_sub[idx] for idx in train_idx])
        train_fmri = fmri_data_sub[train_idx]

        fmri_train_major.append(train_fmri)
        fmri_test_major.append(test_fmri)
        img_train_major.append(train_img)
        img_test_major.append(test_img)
    fmri_train_major = np.concatenate(fmri_train_major, axis=0)
    fmri_test_major = np.concatenate(fmri_test_major, axis=0)
    img_train_major = np.concatenate(img_train_major, axis=0)
    img_test_major = np.concatenate(img_test_major, axis=0)

    num_voxels = fmri_train_major.shape[-1]
    if isinstance(image_transform, list):
        return (BOLD5000_dataset(fmri_train_major, img_train_major, fmri_transform, image_transform[0], num_voxels), 
                BOLD5000_dataset(fmri_test_major, img_test_major, torch.FloatTensor, image_transform[1], num_voxels))
    else:
        return (BOLD5000_dataset(fmri_train_major, img_train_major, fmri_transform, image_transform, num_voxels), 
                BOLD5000_dataset(fmri_test_major, img_test_major, torch.FloatTensor, image_transform, num_voxels))

class BOLD5000_dataset(Dataset):
    def __init__(self, fmri, image, fmri_transform=identity, image_transform=identity, num_voxels=0):
        self.fmri = fmri
        self.image = image
        self.fmri_transform = fmri_transform
        self.image_transform = image_transform
        self.num_voxels = num_voxels
    
    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, index):
        fmri = self.fmri[index]
        img = self.image[index] / 255.0
        fmri = np.expand_dims(fmri, axis=0) 
        return {'fmri': self.fmri_transform(fmri), 'image': self.image_transform(img)}
    
    def switch_sub_view(self, sub, subs):
        # Not implemented
        pass