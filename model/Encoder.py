import torch
from torch import nn
from model.Encoder_Block import EncoderBlock

class Encoder(nn.Module):
    def __init__(self, embedding_size, q_k_size, v_size, f_size, head, nblocks) -> None:
        super().__init__()

        self.encoder_blocks = nn.ModuleList()
        for _ in range(nblocks):
            self.encoder_blocks.append(EncoderBlock(embedding_size=embedding_size, q_k_size=q_k_size, v_size=v_size, f_size=f_size, head=head))


    def forward(self,x): #x:(batch_size,seq_len)
        for block in self.encoder_blocks:
            x = block(x) #x:(batch_size,seq_len,embedding_size)
        return x
    


