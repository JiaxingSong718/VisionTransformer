from torch import nn 
import torch 

class PatchWithPositionEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channel, embedding_size) -> None:
        super().__init__()
        # (H, W)
        img_size = (img_size,img_size)
        # (P, P)
        patch_size = (patch_size,patch_size)
        # N = (H // P) * (W // P)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_count = num_patches

        # Patch
        self.patch = nn.Conv2d(in_channels=in_channel,out_channels=self.patch_count,kernel_size=self.patch_size,padding=0,stride=self.patch_size)

        # Patch Embedding
        self.patch_embedding = nn.Linear(in_features=self.patch_count,out_features=embedding_size)
        
        # class head
        self.class_head = nn.Parameter(torch.rand(1,1,embedding_size))

        # Position Embedding
        self.position_embedding = nn.Parameter(torch.rand(1,self.patch_count+1,embedding_size))

    def forward(self,x): # x:(batch_size,channel=3,width=224,height=224)
        x = self.patch(x) # x:(batch_size,channel=patch_count,width=14,height=14)
        x = x.view(x.size(0),x.size(1),self.patch_count) # x:(batch_size,embedding_size,seq_len=14*14)
        x = x.permute(0,2,1) # x:(batch_size,seq_len=14*14,embedding_size)
        # print(x.size())
        x = self.patch_embedding(x) # x:(batch_size,seq_len,embedding_size)

        class_head = self.class_head.expand(x.size(0),1,x.size(2)) # class_head:(batch_size,1,embedding_size)
        x = torch.cat((class_head, x), dim=1)

        x = self.position_embedding + x
        return x

# if __name__ == '__main__':
#     x = torch.rand(1,3,224,224)
#     c = PatchWithPositionEmbedding(img_size=224,patch_size=16,in_channel=3,embedding_size=768)
#     out = c(x)
#     print(out)

