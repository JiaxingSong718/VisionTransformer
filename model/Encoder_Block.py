from torch import nn
import torch
from model.MultiHeadAttention import MultiHeadAttention

class EncoderBlock(nn.Module):
    def __init__(self, embedding_size, q_k_size, v_size, f_size, head) -> None:
        super().__init__()
        self.Multihead_attention = MultiHeadAttention(embedding_size, q_k_size, v_size, head) #多头注意力
        self.z_linear = nn.Linear(head * v_size, embedding_size) #将多头注意力的输出调整为embedding_size,方便后面的残差连接相加
        self.addnorm1 = nn.LayerNorm(embedding_size) #按照last dim做norm

        #Feed Back
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, f_size),
            nn.ReLU(),
            nn.Linear(f_size, embedding_size)
        )

        self.addnorm2 = nn.LayerNorm(embedding_size) #按照last dim做norm

    def forward(self, x): #x:(batch_size,seq_len,embedding_size)
        z = self.Multihead_attention(x, x) #x: (batch_size, seq_len, head*v_size)
        z = self.z_linear(z) #z:(batch,seq_len,embedding_size)
        output1 = self.addnorm1(z + x) #output:(batch,seq_len,embedding_size)

        output2 = self.feedforward(output1) #output:(batch,seq_len,embedding_size)
        EncoderBlock_output = self.addnorm2(output2 + output1) #output:(batch,seq_len,embedding_size)
        return EncoderBlock_output




