from torch import nn
import torch
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, q_k_size, v_size, head) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.q_k_size = q_k_size
        self.v_size = v_size
        self.head = head

        self.w_q = nn.Linear(embedding_size, head * q_k_size) # head * q_k_size -> 多头
        self.w_k = nn.Linear(embedding_size, head * q_k_size)
        self.w_v = nn.Linear(embedding_size, head * v_size)

    def forward(self, x_q, x_k_v):
        #x_q: (batch_size,seq_max_len,embedding_size)
        q = self.w_q(x_q) #q: (batch_size,seq_max_len,head * q_k_size)
        k = self.w_k(x_k_v) #k: (batch_size,seq_max_len,head * q_k_size)

        # 多头兼容(view作用=reshape)
        q = q.view(q.size()[0], q.size()[1], self.head, self.q_k_size).transpose(1, 2) # (batch_size, head, seq_max_len, q_k_size)
        k = k.view(k.size()[0], k.size()[1], self.head, self.q_k_size).transpose(1, 2).transpose(2, 3) # (batch_size, head, q_k_size, seq_max_len)

        # 注意力矩阵
        attention = torch.matmul(q, k) / math.sqrt(self.q_k_size) #(batch_size, head, seq_max_len, seq_max_len), row是q, col是k
        # print(attention.size())

        # 注意力分值处理
        attention = torch.softmax(attention, dim=-1) #(batch_size,head,seq_max_len,seq_max_len)

        # 注意力与v相乘
        v = self.w_v(x_k_v) #k: (batch_size,seq_max_len,head * v_size)
        v = v.view(v.size()[0], v.size()[1], self.head, self.v_size).transpose(1, 2) # (batch_size, head, seq_max_len, v_size)
        z = torch.matmul(attention, v) # (batch_size, head, seq_max_len, v_size)
        z = z.transpose(1,2) # (batch_size, seq_max_len, head, v_size)
        return z.reshape(z.size()[0], z.size()[1],-1) # (batch_size, seq_max_len, head * v_size)
    


