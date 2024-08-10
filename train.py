import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.dataset import train_dataset, val_dataset
from model.VisionTransformer import VisionTransformer


device = 'cuda' if torch.cuda.is_available() else 'cpu'


vit = VisionTransformer(img_size=64, patch_size=8, in_channel=3, embedding_size=64, q_k_size=64, v_size=64, f_size=128, head=2, nblocks=3, class_num=10).to(device)

try:
    vit.load_state_dict(torch.load('./checkpoints/model.pth'))
except:
    pass

optimizer = torch.optim.Adam(vit.parameters(), lr=1e-5)

epochs = 300
batch_size = 256


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

iter_count = 0

for epoch in range(epochs):
    vit.train()
    train_loss = 0
    
    # Training loop with progress bar
    for img, label in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}', ncols=100):
        logist = vit(img.to(device))

        loss = F.cross_entropy(logist, label.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if iter_count % 10 == 0:
            torch.save(vit.state_dict(), './checkpoints/model.pth')
            os.replace('./checkpoints/model.pth', './checkpoints/model.pth')

        iter_count += 1

    train_loss /= len(train_dataloader)

    # Validation loop
    vit.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for img, label in val_dataloader:
            logist = vit(img.to(device))
            loss = F.cross_entropy(logist, label.to(device))

            val_loss += loss.item()
            _, predicted = logist.max(1)
            total += label.size(0)
            correct += predicted.eq(label.to(device)).sum().item()

    val_loss /= len(val_dataloader)
    val_accuracy = 100. * correct / total

    print(f'Epoch {epoch+1}/{epochs} - '
          f'Train Loss: {train_loss:.4f} '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
