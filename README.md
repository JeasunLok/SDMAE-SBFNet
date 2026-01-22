# Investigating Very High Resolution Land Cover Mapping in the Pearl River Delta with Remote Sensing Foundation Model and Multi-Source Data Bayesian Fusion

***
## Introduction

<!-- <b> Official implementation of [SDMAE-SBFNet](https://ieeexplore.ieee.org/document/10975807) by [Junshen Luo](https://github.com/JeasunLok), Jiahe Li, Xinlin Chu, Sai Yang, Lingjun Tao and Qian Shi. </b> -->
<b> Official implementation of SDMAE and SBFNet for VHR land cover mapping by [Junshen Luo](https://github.com/JeasunLok), Yikai Zhao, Mingyang Xuan, Jizhou Zhen, Yan Zhou* and Xiaoping Liu. </b>
***
![](images/SDMAE.jpg)
***
![](images/SBFNet.jpg)
***

## How to use it?
### 1. Installation
```
git clone https://github.com/JeasunLok/SDMAE-SBFNet.git && cd SDMAE-SBFNet
conda create -n SDMAE-SBFNet python=3.11
conda activate SDMAE-SBFNet
pip install -r requirements.txt
```

### 2. Download our datasets

Download our datasets and run `generate_list_pretrained.py` or `generate_list_segmentation.py` to generate the data lists. Then place the lists in the correct path of `data` folder:

Zenodo: https://doi.org/10.5281/zenodo.18301135.

### 3. Quick start to use our pretraining model SDMAE

<b> You should change the settings in `sdmae_pretrain_ddp.py` then: </b>
```
torchrun --nproc_per_node=2 sdmae_pretrain_ddp.py
```
You can use tensorboard to visualize the pretraining process:
```
tensorboard --logdir=/path/log_folder/SummaryWriter --port=6061
```
Then link it to your PC through ssh:
```
ssh -NfL (port of your PC):127.0.0.1:6061 username@host -p port
```

### 4. Quick start to use our semantic segmentation model SBFNet with SDMAE
```
python train_SDSBFNet.py
```

<!-- ***
## Citation
<b> Please kindly cite the papers if this code is useful and helpful for your research. </b>

J. Luo, J. Li, X. Chu, S. Yang, L. Tao and Q. Shi, "BTCDNet: Bayesian Tile Attention Network for Hyperspectral Image Change Detection," in IEEE Geoscience and Remote Sensing Letters, vol. 22, pp. 1-5, 2025, Art no. 5504205, doi: 10.1109/LGRS.2025.3563897.

```
@article{luo2025btcdnet,
  title={BTCDNet: Bayesian Tile Attention Network for Hyperspectral Image Change Detection},
  author={Luo, Junshen and Li, Jiahe and Chu, Xinlin and Yang, Sai and Tao, Lingjun and Shi, Qian},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2025},
  publisher={IEEE}
}
``` -->

***
## Contact Information
Junshen Luo: luojsh7@mail2.sysu.edu.cn

Junshen Luo is with School of Geography and Planning, Sun Yat-sen University, Guangzhou 510275, China
***