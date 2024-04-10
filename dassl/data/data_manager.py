import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset

from dassl.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform

import numpy as np
import cv2
import scipy.fftpack as fft
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
import sys
sys.path.append('/home/aleksandrmatsun/CCSDG')
from ccsdg.utils.fourier import FDA_source_to_target_np
from ccsdg.datasets.utils.normalize import normalize_image
from ccsdg.datasets.utils.slaug import LocationScaleAugmentation


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    if cfg.DATALOADER.CCSDG:
        if is_train:
            data_loader = torch.utils.data.DataLoader(
                dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
                batch_size=batch_size,
                sampler=sampler,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                drop_last=is_train and len(data_source) >= batch_size,
                pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
                collate_fn=collate_fn_tr_styleaug
            )
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
                batch_size=batch_size,
                sampler=sampler,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                drop_last=is_train and len(data_source) >= batch_size,
                pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
                collate_fn=collate_fn_ts
            )
    else:
        if is_train:
            data_loader = torch.utils.data.DataLoader(
                dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
                batch_size=batch_size,
                sampler=sampler,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                drop_last=is_train and len(data_source) >= batch_size,
                pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
            )
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
                batch_size=batch_size,
                sampler=sampler,
                num_workers=cfg.DATALOADER.NUM_WORKERS,
                drop_last=is_train and len(data_source) >= batch_size,
                pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
            )
        
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        self.ccsdg = cfg.DATALOADER.CCSDG
        self.return_high = cfg.DATALOADER.RETURN_HIGH

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.ToTensor()]
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode, antialias=True)]
        self.to_tensor_high = T.Compose(to_tensor)
        
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,
            "index": idx
        }

        img0 = read_image(item.impath)

        if self.ccsdg:
            img_n = cv2.normalize(np.array(img0), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # img_n = cv2.resize(img_n, self.cfg.INPUT.SIZE)
            output["img_numpy"] = img_n.transpose(2,0,1)

            

        elif self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0)  # without any augmentation
        if self.return_high:
            high = np.array(read_image(item.hpath)).mean(2)
            high = cv2.normalize(np.array(high), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            output["high"] = self.to_tensor_high(high)

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img
    

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

def fourier_augmentation_reverse(data, fda_beta=0.1):
    this_fda_beta = round(0.05+np.random.random() * fda_beta, 2)
    lowf_batch = data[::-1]
    fda_data = FDA_source_to_target_np(data, lowf_batch, L=this_fda_beta)
    return fda_data


def sl_augmentation(image):
    location_scale = LocationScaleAugmentation(vrange=(0., 255.), background_threshold=0.01)
    GLA = location_scale.Global_Location_Scale_Augmentation(image.copy())
    # LLA = location_scale.Local_Location_Scale_Augmentation(image.copy(), mask.copy().astype(np.int32))
    return GLA


def get_train_transform(patch_size=(512, 512)):
    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )
    # tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))
    # tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
    #                                            p_per_channel=0.5, p_per_sample=0.15))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

def collate_fn_tr_styleaug(batch):
    # image, label, name = zip(*batch)
    image = np.stack([d['img_numpy'] for d in batch])
    high_flag = bool('high' in batch[0])
    if high_flag:
        high = torch.stack([d['high'] for d in batch])
    impath = [d['impath'] for d in batch]
    domain = torch.tensor([d['domain'] for d in batch])
    label = torch.tensor([d['label'] for d in batch])

    data_dict = {"data" : image}

    tr_transforms = get_train_transform(patch_size=image.shape[-2:])
    data_dict = tr_transforms(**data_dict)

    fda_data = fourier_augmentation_reverse(data_dict['data'])
    data_dict['fda_img'] = torch.from_numpy(normalize_image(fda_data)).to(dtype=torch.float32)
    GLA = sl_augmentation(data_dict['data'])
    data_dict['data'] = torch.from_numpy(normalize_image(data_dict['data'])).to(dtype=torch.float32)
    data_dict['GLA_img'] = torch.from_numpy(normalize_image(GLA)).to(dtype=torch.float32)
    data_dict['impath'] = impath
    data_dict['domain'] = domain
    data_dict['label'] = label

    if high_flag:
        data_dict['seg'] = high

    return data_dict

def collate_fn_ts(batch):
    image = np.stack([d['img_numpy'] for d in batch])
    label = torch.tensor([d['label'] for d in batch])

    data_dict = {}

    data_dict['img'] = torch.from_numpy(normalize_image(image)).to(dtype=torch.float32)
    data_dict['label'] = label

    return data_dict