import torchvision
from torchvision import transforms

data_transform = {
        "train": transforms.Compose([transforms.Resize(64),
                                     transforms.RandomCrop(64, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


# 加载训练集
train_dataset = torchvision.datasets.CIFAR10(root='./dataset/data', train=True,
                                        download=True, transform=data_transform['train'])

# 加载测试集
val_dataset = torchvision.datasets.CIFAR10(root='./dataset/data', train=False,
                                       download=True, transform=data_transform['val'])

