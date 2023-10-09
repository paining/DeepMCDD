# line image crop
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.signal import medfilt

logger = logging.getLogger(__name__)

def image_crop(img:np.ndarray, criterion=10):
    img = np.array(img)
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 1:
            img_gray = img.reshape(img.shape[:2])
        else:
            raise ValueError(f"Invalid Shape : {img.shape}")
    elif len(img.shape) == 2:
        img_gray = img
    else:
        raise ValueError(f"Invalid Shape : {img.shape}")
    mean = np.mean(img_gray, axis=0)
    mean = medfilt(mean, 5)
    # std = np.std(img_gray, axis=0)
    half = 3

    idx = len(mean)//2
    while idx < len(mean) - half:
        diff = mean[idx + half] - mean[idx - half]
        if idx == len(mean) // 2:
            last_diff = diff
        # if (mean[idx + half] - mean[idx - half]) > criterion:
        if abs(diff - last_diff) >= 15:
            r_idx = idx
            break

        idx += 1
        last_diff = diff
    else:
        r_idx = -1 # print('경계를 찾을 수 없음')

    cnt = 0
    idx = len(mean)//2

    while idx > half:
        diff = mean[idx - half] - mean[idx + half]
        if idx == len(mean) // 2:
            last_diff = diff
        # if (mean[idx - half] - mean[idx + half]) > criterion:
        if abs(diff - last_diff) >= 15:
            l_idx = idx
            break
        idx -= 1
        last_diff = diff
    else:
        l_idx = -1 # print('경계를 찾을 수 없음')

    return l_idx, r_idx, r_idx-l_idx


if __name__ == "__main__":
    # data_path = "C:/Users/YSH/OneDrive - ysh-pleiony/OneDrive - Pleiony,Inc/data/private/donga/20230313/Silver_9_400_60_(0)_m600_ex800_g3_off0/normal/standard"
    # data_path = "C:/Users/YSH/OneDrive - ysh-pleiony/OneDrive - Pleiony,Inc/data/private/donga/20230313/Silver_9_400_60_(0)_m600_ex800_g3_off0/etc"
    # data_path = "C:/Users/YSH/OneDrive - ysh-pleiony/OneDrive - Pleiony,Inc/data/private/donga/20230313/Black_14_325_60_(2)_m600_ex800_g10_off30/normal/standard"
    # data_path = "C:/Users/YSH/OneDrive - ysh-pleiony/OneDrive - Pleiony,Inc/data/private/donga/20230313/Black_14_325_60_(1)_m600_ex800_g10_off-20/etc"

    # data_path = "CameraImage_00011.png"
    data_path = "C:\\Data\\20230310\\Black_13_295_60_(3)_m600_ex800_g10_off70\\normal\\standard"

    logging.basicConfig(level=logging.INFO)
    import os
    if os.path.isdir(data_path):
        filelist = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    elif os.path.isfile(data_path):
        filelist = [data_path]
    else:
        raise ValueError(f"data_path is not file or directory. {data_path}")
    logger.info(f"Number of files : {len(filelist)}")

    for file in filelist:
        logger.info(f"reading {file}")
        img = cv2.imread(file)
        logger.info(f"image size : {img.shape}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        logger.info(f"gray image size : {img.shape}")

        l, r, w = image_crop(img, 20)

        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(3,1,1)
        ax.set_title("gray_img")
        ax.imshow(gray, cmap='gray', vmin=0, vmax=255)

        ax = fig.add_subplot(3,1,2)
        ax.set_title("mean")
        # mean = medfilt(np.mean(gray, axis=0), 5)
        mean = medfilt(np.median(gray, axis=0), 5)
        dmean = np.zeros_like(mean)
        dmean[3:-3] = abs(mean[:-6] - mean[6:])
        ax.plot(mean, "g", label="mean")
        ax.plot(dmean, "b", label="diff")
        ax.vlines([l, r], ymin=0, ymax=255, colors='r')
        ax.legend()

        ax = fig.add_subplot(3,1,3)
        ax.set_title("croped image")
        ax.imshow(gray[:, l:r], cmap="gray", vmin=0, vmax=255)

        fig.tight_layout()
        plt.show()
        logger.info("show!")
        plt.close(fig)

