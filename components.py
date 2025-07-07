# Multi-Head Attention
from torch.nn import Module
import torch.nn as nn
import torch
import math
# 重要的参数：model_dimension
# 重要的参数：head_number
class MyMultiHeadAttention (Module):
    def __init__(self, seq_length, input_dimension, model_dimension, head_number):
        super().__init__()
        if (model_dimension % head_number != 0):
            raise (f"Wrong head_number Of model dimension {0}", model_dimension)

        self.seq_length = seq_length
        self.model_dimension = model_dimension
        self.head_number = head_number
        self.changed_dimension = model_dimension // head_number

        self.q_proj = nn.Linear(input_dimension, model_dimension)
        self.k_proj = nn.Linear(input_dimension, model_dimension)
        self.v_proj = nn.Linear(input_dimension, model_dimension)
        self.o_proj = nn.Linear(model_dimension, model_dimension)

    def forward(self, X):
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        Q = Q.reshape(self.seq_length, self.head_number,
                      self.changed_dimension)
        K = K.reshape(self.seq_length, self.head_number,
                      self.changed_dimension)
        V = V.reshape(self.seq_length, self.head_number,
                      self.changed_dimension)
        # 交换维度
        Q = Q.permute(1, 0, 2)
        K = K.permute(1, 0, 2)
        V = V.permute(1, 0, 2)
        K_T = K.permute(0, 2, 1)
        # (head_number, seq_length, changed_dimension)
        attention_score = torch.matmul(Q, K_T) /  math.sqrt(self.changed_dimension)
        attention_result = torch.matmul(torch.softmax(attention_score, 2), V)
        attention_result = attention_result.permute(1, 0, 2)
        # (seq_length, head_number, changed_dimension)
        attention_result = attention_result.reshape(self.seq_length, self.model_dimension)
        # (seq_length, head_number, model_dimension)
        output = self.o_proj(attention_result)
        return output


