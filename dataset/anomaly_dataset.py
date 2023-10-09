from torch import zeros
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from typing import Tuple, List
from image_crop import image_crop
import numpy as np

class AnomalyDetecionDataset(Dataset):
    def __init__(
        self,
        img_path_root: str,
        imgfile_list: list,
        label_list: list,
        gtfile_list: list,
        transform: transforms,
        gt_transform: transforms,
        roi: list = None,
    ):
        """Dataset for AnomalyDetection SDKs."""
        self.img_path_root = img_path_root
        self.img_paths = imgfile_list
        self.labels = label_list
        self.gt_paths = gtfile_list
        self.transform = transform
        self.gt_transform = gt_transform
        self.roi = tuple(roi) if roi else roi

        if len(self.img_paths) != len(self.labels):
            print("Error! dataset's size is different!")
            raise RuntimeError

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        gt = self.gt_paths[idx]
        label = self.labels[idx]
        filename, _ = os.path.splitext(
            img_path.replace(self.img_path_root,"")
            .replace("\\", "/")
            .replace("/","_")
            .lstrip("_")
        )

        img = Image.open(img_path).convert("RGB")#.crop(self.roi)
        # l, r, w = image_crop(img, 50)
        # cropsize = 32 * (w // 32)
        # offset = (w % 32) // 2
        # l = l + offset
        # img = img.crop((l, 0, l+cropsize, 416))
        # print(l+cropsize, filename)
        # dac test 230131
        # img = img.crop((936, 0, 3208, 416))

        def img_crop(input_img, crop=False):
            croprange = None
            if crop:
                l, r, w = image_crop(input_img)
                if l != -1 and r != -1 and w > 1000:
                    cropsize = 32 * (w // 32)
                    offset = (w % 32) // 2
                    l = l + offset
                    croprange = (l, l+cropsize)
                    crop_img_np = input_img[:, l:l+cropsize, :]
                else:
                    crop = False
                    crop_img_np = input_img.copy()
            else:
                crop_img_np = input_img.copy()
            return crop_img_np, croprange
        img = img.crop((0, 0, 4096, 416))
        img, croprange = img_crop(np.asarray(img), True)
        img = Image.fromarray(img)
        img = self.transform(img)

        if gt != 0:
            gt = Image.open(gt).convert("1")#.crop(self.roi)
            if croprange:
                gt = gt.crop((croprange[0], 0, croprange[1], 416))
            else:
                gt = gt.crop((0, 0, 4096, 416))
            # gt = gt.crop((l, 0, l+cropsize, 416))
            # dac test 230131
            # gt = gt.crop((936, 0, 3208, 416))
            gt = self.gt_transform(gt)
        else:
            gt = zeros([1, img.size()[-2], img.size()[-1]])

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, filename


def load_dataset_from_path(
    datapath, prefixs, gt_datapath=None, recursive=False, **kwargs
) -> Tuple[List[str], List[str], List[str]]:
    """Load dataset from path.
    label will automatically added by prefixs.

    Examples: (MVTec AD)
    ---------
    >>> img_paths, labels, gt_paths = load_dataset_from_path(
            "./bottle/test", [["good"], []], "./bottle/ground_truth", True
        )
    >>> print( img_paths )
    [ "./bottle/test/good/000.jpg", ...,
      "./bottle/test/broken_large/000.jpg", ...,
      "./bottle/test/broken_small/000.jpg", ...,
      "./bottle/test/contamination/000.jpg", ... ]
    >>> print( labels )
    [ 0, ...,
      1, ...,
      1, ...,
      1, ... ]
    >>> print( gt_paths )
    [ 0, ...,
      "./bottle/ground_truth/broken_large/000.jpg", ...,
      "./bottle/ground_truth/broken_small/000.jpg", ...,
      "./bottle/ground_truth/contamination/000.jpg", ... ]
    """
    if len(prefixs) != 2:
        print("prefix error : it should be 2-ch list")
    else:
        for i in range(len(prefixs)):
            if prefixs[i] is None:
                prefixs[i] = []

    # initialize return buffer
    img_paths = []
    labels = []
    gt_paths = []

    if recursive:
        filelist = []
        for (root, dirs, files) in os.walk(datapath):
            dirs.sort()
            files.sort()
            filelist.extend(
                [
                    os.path.join(root, file)
                    for file in files
                    if file.endswith((".jpg", "jpeg", ".bmp", ".png"))
                ]
            )
    else:
        filelist = [
            os.path.join(datapath, file)
            for file in os.listdir(datapath)
            if file.endswith((".jpg", "jpeg", ".bmp", ".png"))
        ]
    img_paths.extend(filelist)

    for path in img_paths:
        label = -1
        for i in range(len(prefixs)):
            if prefixs[i] and any(prefix in path for prefix in prefixs[i]):
                label = i
                break
        labels.append(label)

    if isinstance(gt_datapath, str):
        for filepath, label in zip(img_paths, labels):
            if label >= 0:
                gt_paths.append(0)
                continue
            new_path = filepath.replace(datapath, gt_datapath)
            name, ext = os.path.splitext(new_path)
            # new_path = "".join([name, "_label", ".png"])
            # new_path = "".join([name, "_mask.png"]) # MVTec AD
            new_path = "".join([name, ".png"])
            if os.path.isfile(new_path):
                gt_paths.append(new_path)
            else:
                print("file is not exist:", new_path)
                gt_paths.append(0)
    elif gt_datapath == None:
        gt_paths = [0 for i in range(len(img_paths))]

    return img_paths, labels, gt_paths
