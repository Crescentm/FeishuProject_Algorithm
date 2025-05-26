# 本项目是针对Blog课题的图像识别算法代码

## 1.代码分为以下几个重要部分
* 模型文件：`model.py`，其中包含EfficientNet B0-B7的定义与实现，本项目采用的是EfficientNet B7模型。
* 数据加载文件：`dataset.py`，其中包含对输入模型的图像的预处理，包含Train和Val的处理，定义如下，针对不同的数据集有不同的处理方法：
```
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode

def build_transform(dest_image_size, is_train=True):
    if not isinstance(dest_image_size, tuple):
        dest_image_size = (dest_image_size, dest_image_size)

    if is_train:
        # 训练时的变换：包含数据增强
        transform = transforms.Compose([
            transforms.RandomResizedCrop(dest_image_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
        ])
    else:
        # 验证时的变换：仅调整大小和标准化
        transform = transforms.Compose([
            transforms.Resize(dest_image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform

def build_data_set(dest_image_size, data, is_train=True):
    transform = build_transform(dest_image_size, is_train=is_train)
    dataset = datasets.ImageFolder(data, transform=transform, target_transform=None)
    return dataset
```
* 指标计算文件：`metric.py`，包含指标的计算，主要是`F1，Recall，Precision`，使用的是自带的库函数，且设置average分别为`micro`和`macro`两种模式：
```
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(
        tgt_np, pred_np,
        average=average,  # 可以设置不同的方式
        zero_division=0
    )
```
* 模型训练文件：`train.py`，主要是对模型训练代码的设置，设置超参，如`batch_size`，`epoch`，`lr`等。

## 2.模型的训练过程
在当前文件目录下，在终端直接使用`python train.py`即可，可以使用`CUDA_VISIBLE_DEVICES=1`来设置使用那张GPU训练，如果出现显存OOM问题，可以适当调小batch_size。
训练过程中，会将最优的模型保存在`ckpts`文件中，命名为`best_model.pth`。同时会保存模型训练和验证阶段计算的指标，保存到runs文件中，可以使用下面指令，在`tensorbaord`中查看：
```
tensorboard --logdir=Blog/code/runs --port=6006(端口号)
```

## 3.模型的可视化过程
### 1.使用test.ipynb脚本
可以在`jupyternote book`脚本中执行，其中定义了所有需要的脚本代码，包括测试推理延迟，生成图像热力图，限制CPU数量，以及转换模型格式（从`.pth`变成`.onnx`和`.pt`）等。
### 2.使用visual_save.py
可以直接在当前目录下执行`python visual_save.py`，需要先设置数据集地址，执行后，会在当前目录下生成`result`文件夹，具体的组织架构如下：
```
- result
-- class1
--- xxx1_overlay.png
--- xxx2_overlay.png
--- ......
-- class2
-- class3
-- result.csv  # 包含file_path（文件路径）,file_label（图片真实标签）,pred_label（图片预测标签）,pred_prob（图片预测概率）
```
## 4.最后脚本提交文件image_predict.py
最后需要提交的测试脚本，包含模型的加载，数据的读取与处理，结果的保存。