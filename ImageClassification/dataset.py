# from torchvision import transforms
# from torchvision import datasets
# from torchvision.transforms import InterpolationMode

# def _norm_advprop(img):
#     return img * 2.0 - 1.0

# def build_transform(dest_image_size):
#     normalize = transforms.Lambda(_norm_advprop)
#     if not isinstance(dest_image_size, tuple):
#         dest_image_size = (dest_image_size, dest_image_size)
#     else:
#         dest_image_size = dest_image_size

#     transform = transforms.Compose([
#         transforms.Resize(dest_image_size, interpolation=InterpolationMode.BICUBIC),
#         transforms.ToTensor(),
#         normalize
#     ])

#     return transform


# def build_data_set(dest_image_size, data):
#     transform = build_transform(dest_image_size)
#     dataset = datasets.ImageFolder(data, transform=transform, target_transform=None)
#     return dataset

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