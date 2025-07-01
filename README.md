![header](https://capsule-render.vercel.app/api?type=Venom&color=gradient&height=200&text=EMM&fontSize=80)

### EMMnet : EEG and MRI Multimodal
<img src="https://img.shields.io/badge/Python-2B2728?style=plastic&logo=Python&logoColor=3776AB"/> <img src="https://img.shields.io/badge/PyTorch-2B2728?style=plastic&logo=PyTorch&logoColor=EE4C2C"/>


## 1. Getting Started

### Requirements

- Installation of Conda (refer to <https://www.anaconda.com/products/distribution>)
- Nvidia GPU with CUDA support

> Note: we tested the code in the following environments.
>
> |    **OS**    | **Python** | **PyTorch** | **CUDA** |
> |:------------:|:----------:|:-----------:|:--------:|
> |  Windows 10  |  not yet   |   not yet   | not yet  |
> | Ubuntu 20.04 |   3.9.21   |    2.2.1    |   12.1   |

### Installation

(optional) Create and activate a Conda environment.

```bash
  conda create -n caueeg python=3.9
  conda activate caueeg
```

Install PyTorch library (refer to <https://pytorch.org/get-started/locally/>).

Install other necessary libraries.

```bash
  pip install -r requirements.txt
```

### Preparation of the [CAUEMM](https://github.com/ipis-mjkim/caueeg-dataset) dataset

> ❗ Note: The use of the CAUEMM dataset is allowed for only academic and research purposes 👩‍🎓👨🏼‍🎓.

- For full access of the CAUEEG dataset, follow the instructions specified in <https://github.com/ipis-mjkim/caueeg-dataset>.
- Download, unzip, and move the whole dataset files into [local/datasets/](local/datasets/).

```
dataset
├── abnormal_EMM.json
├── annotation_EMM.json
├── dementia_EMM.json
├── CAUEEG
│   └── caueeg-dataset
│        ├── abnormal.json
│        ├── annotation.json
│        ├── dementia.json
│        ├── event
│        └── signal
│            ├── edf
│            └── memmap
│                ├── 00001.nii
│                ├── ...
│                └── 01379.nii
└── CAUMRI
    └── caumri-dataset
        └── image
            └── nii
                ├── 00001.nii
                ├── ...
                └── 01379.nii
```

> 💡 Note: We provide `caueeg-dataset-test-only` at [[link 1]](https://drive.google.com/file/d/1P3CbLY7h9O1CoWEWsIZFbUKoGSRUkTA1/view?usp=sharing) or [[link 2]](http://naver.me/xzLCBwFp) to test our research. `caueeg-dataset-test-only` has the 'real' test splits of two benchmarks (*CAUEEG-Dementia* and *CAUEEG-Abnormal*) but includes the 'fake' train and validation splits.

---
## 2. Usage

### Train EMMNet

Train our EMMNet model on the training set of *CAUEMM-Dementia* from scratch using the following command:

```bash
  python train.py data=cauemm-dementia train=base_train model=multi-modal
```

Similarly, train our model on the training set of *CAUEMM-Abnormal* from scratch using:

```bash
  python train.py data=cauemm-abnormal train=base_train model=multi-modal
```

Or, you can check [this Jupyter notebook](notebook/06_Pretrain_Self_Supervision_MAE.ipynb).

If you want to start with pretrained weights:

```bash
  python train.py   data=cauemm-abnormal  train=fine_tune   model=multi-modal
```

If you encounter a GPU memory allocation error or wish to adjust the balance between memory usage and training speed, you can specify the minibatch size by adding the `++model.minibatch=INTEGER_NUMBER` option to the command as shown below:

```bash
  python train.py data=caueeg-abnormal train=base_train model=multi-modal ++model.minibatch=32
```

Thanks to [Hydra](https://hydra.cc/) support, the model, hyperparameters, and other training details are easily tuned using or modifying config files.


### Evaluation

Evaluation can be conducted using [this Jupyter notebook](notebook/03_Evaluate.ipynb) (or [another notebook](notebook/03_Evaluate_Test_Only.ipynb) for `caueeg-dataset-test-only` case)


---


### *CAUEEG-Abnormal* dataset

|                   Model                   | #Params | Model size (MiB) | TTA | Throughput (EEG/s) | Test accuracy |                                             Link 1                                             |                Link 2                |
|:-----------------------------------------:|:-------:|:----------------:|:---:|:------------------:|:-------------:|:----------------------------------------------------------------------------------------------:|:------------------------------------:|
|         K-Nearest Neighbors (K=7)         |    -    |      14015.3     |     |        41.19       |     51.42%    |                                                                                                |                                      |
|       Random Forests (#trees=2000)        |    -    |      1930.5      |     |       830.80       |     72.63%    |                                                                                                |                                      |
|                Linear SVM                 |   0.1M  |        0.3       |     |      10363.76      |     68.00%    |                                                                                                |                                      |
|               Ieracitano-CNN              |   3.5M  |       13.2       |     |       8293.08      |     65.98%    |                                                                                                |                                      |
|            CEEDNet (1D-VGG-19)            |  20.2M  |       77.2       |     |       7660.22      |     72.45%    | [nemy8ikm](https://drive.google.com/file/d/1NpDsxmFMln71d9JEpnGfCxhaRtyWK4su/view?usp=sharing) | [nemy8ikm](http://naver.me/x1gdjONm) |
|            CEEDNet (1D-VGG-19)            |  20.2M  |       77.2       |  ✔  |       998.54       |     74.28%    | [nemy8ikm](https://drive.google.com/file/d/1NpDsxmFMln71d9JEpnGfCxhaRtyWK4su/view?usp=sharing) | [nemy8ikm](http://naver.me/x1gdjONm) |
|          CEEDNet (1D-ResNet-18)           |  11.4M  |       43.5       |  ✔  |       844.65       |     74.85%    | [4439k9pg](https://drive.google.com/file/d/1LH069g2oyO2XvEDzFpJPR9X5xuLmcnq3/view?usp=sharing) | [4439k9pg](http://naver.me/5vYbUTay) |
|          CEEDNet (1D-ResNet-50)           |  26.3M  |       100.7      |  ✔  |       837.66       |     76.37%    | [q1hhkmik](https://drive.google.com/file/d/1U9G0nJ-dYe6RBFxuCsdCkh-LU5AxwqFS/view?usp=sharing) | [q1hhkmik](http://naver.me/xEqsymHV) |
|          CEEDNet (1D-ResNeXt-50)          |  25.7M  |       98.2       |  ✔  |       800.49       |     77.32%    | [tp7qn5hd](https://drive.google.com/file/d/1OR5Z4U-QWDZBlm8A8pnRB2LMU0wTOMVa/view?usp=sharing) | [tp7qn5hd](http://naver.me/GItl9VHH) |
|            CEEDNet (2D-VGG-19)            |  20.2M  |       77.2       |  ✔  |       447.81       |     75.39%    | [ruqd8r7g](https://drive.google.com/file/d/1UUADOHCoBc4wt9LmIn-GitbPzNRbmCTn/view?usp=sharing) | [ruqd8r7g](http://naver.me/GkJzA84q) |
|          CEEDNet (2D-ResNet-18)           |  11.5M  |       43.8       |  ✔  |       410.44       |     75.19%    | [dn10a6bv](https://drive.google.com/file/d/12bsVV0dcVbbjO4eB3vN7ykeFAf6vp-7P/view?usp=sharing) | [dn10a6bv](http://naver.me/51nm4WtS) |
|          CEEDNet (2D-ResNet-50)           |  25.7M  |       98.5       |  ✔  |       187.30       |     74.96%    | [atbhqdgg](https://drive.google.com/file/d/1ZWnK04-o5V1eIDtlE_5Ct83oxaMfHkzX/view?usp=sharing) | [atbhqdgg](http://naver.me/5Lo4eJAa) |
|          CEEDNet (2D-ResNeXt-50)          |  25.9M  |       99.1       |  ✔  |       201.01       |     75.85%    | [0svudowu](https://drive.google.com/file/d/1A8npNb_3ixmS6ui6yTonh95oQXwPjHWp/view?usp=sharing) | [0svudowu](http://naver.me/FEdfcVaz) |
|            CEEDNet (ViT-B-16)             |  86.9M  |       331.6      |  ✔  |        63.99       |     72.70%    | [1cdws3t5](https://drive.google.com/file/d/1OT-xOTJ2kSqYWOG0KWQ6PeSPYWdX52Lo/view?usp=sharing) | [1cdws3t5](http://naver.me/xkqoPaor) |
|            CEEDNet (Ensemble)             |  253.8M |       969.9      |  ✔  |        26.40       |     79.16%    |                                                                                                |                                      |



### *CAUEMM-Abnormal* dataset