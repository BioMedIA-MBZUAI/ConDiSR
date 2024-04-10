import os.path as osp

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase

import glob


@DATASET_REGISTRY.register()
class Cam17(DatasetBase):
    """Cam17.

        TODO
    """

    dataset_dir = "cam17"
    high_dir = "cam17_high"
    domains = ["center_0", "center_1", "center_2", "center_3", "center_4"]
    # data_url = "https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE"
    # the following images contain errors and should be ignored
    _error_paths = ["sketch/dog/n02103406_4068-1.png"]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.high_dir = osp.join(root, self.high_dir)

        # if not osp.exists(self.dataset_dir):
        #     dst = osp.join(root, "pacs.zip")
        #     self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "train")
        val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "crossval")
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, "all")

        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self, input_domains, split):
        items = []

        for domain, dname in enumerate(input_domains):
            if split == "all":
                all_pths = glob.glob(osp.join(self.dataset_dir, dname, '*', '*'))
                high_pths = glob.glob(osp.join(self.high_dir, dname, '*', '*'))
            elif split == 'train':
                all_pths = glob.glob(osp.join(self.dataset_dir, dname, '*', '*0.png'))
                high_pths = glob.glob(osp.join(self.high_dir, dname, '*', '*0.png'))
            elif split == 'crossval':
                all_pths = glob.glob(osp.join(self.dataset_dir, dname, '*', '*1.png'))
                high_pths = glob.glob(osp.join(self.high_dir, dname, '*', '*1.png'))

            all_pths.sort()
            high_pths.sort()
            
            for impath, hpath in zip(all_pths, high_pths):
                classname = impath.split("/")[-2]
                label = int(classname)
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=classname,
                    hpath = hpath
                )
                items.append(item)

        return items

    # def _read_split_pacs(self, split_file):
    #     items = []

    #     with open(split_file, "r") as f:
    #         lines = f.readlines()

    #         for line in lines:
    #             line = line.strip()
    #             impath, label = line.split(" ")
    #             if impath in self._error_paths:
    #                 continue
    #             impath = osp.join(self.image_dir, impath)
    #             label = int(label) - 1
    #             items.append((impath, label))

    #     return items