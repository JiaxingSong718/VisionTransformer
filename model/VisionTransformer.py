import torch
from torch import nn
from model.PatchWithPositionEmbedding import PatchWithPositionEmbedding
from model.Encoder import Encoder


class VisionTransformer(nn.Module):
    def __init__(self,img_size, patch_size, in_channel, embedding_size, q_k_size, v_size, f_size, head, nblocks,class_num) -> None:
        super().__init__()
        self.patch_positionembedding = PatchWithPositionEmbedding(img_size, patch_size, in_channel, embedding_size)
        self.encoder = Encoder(embedding_size, q_k_size, v_size, f_size, head, nblocks)
        self.class_Linear = nn.Linear(embedding_size,class_num)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self,x):
        x = self.patch_positionembedding(x)
        y = self.encoder(x)
        output = self.class_Linear(y[:,0,:])
        return self.softmax(output)
