import torch
import yaml
from PIL import Image
import matplotlib.pyplot as plt 
from dataset.dataset import train_dataset, val_dataset
from model.VisionTransformer import VisionTransformer
from torchvision import transforms

with open('./dataset/data.yaml', 'r') as file:
    data = yaml.safe_load(file)

cifar10_classes = data['cifar10_classes']
print(cifar10_classes)

inverse_transform = transforms.Compose([
    transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1/0.229, 1/0.224, 1/0.225]),  # 反归一化
    transforms.ToPILImage()  # 转回 PIL 图像
])

transform_image = transforms.Compose([transforms.Resize((64,64)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

device = 'cuda' if torch.cuda.is_available() else 'cpu'


vit = VisionTransformer(img_size=64, patch_size=8, in_channel=3, embedding_size=64, q_k_size=64, v_size=64, f_size=128, head=2, nblocks=3, class_num=10).to(device)

vit.load_state_dict(torch.load('./checkpoints/model.pth'))

vit.eval()

image = transform_image(Image.open('inference_image.png').convert('RGB'))

# image,label=val_dataset[9]
# print('正确分类:',cifar10_classes[label])

logits=vit(image.unsqueeze(0).to(device))
print('预测分类:',cifar10_classes[logits.argmax(-1).item()])

plt.imshow(inverse_transform(image))
plt.title(cifar10_classes[logits.argmax(-1).item()])
plt.show()


