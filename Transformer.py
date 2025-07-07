# 一组/一个权重是一种学习的信息
# 你好
# 1 2 seq_length,
# Embedding =>  seq_length, embedding dimension
# + Position => seq_length, model_dimension
# Q = x W_Q   (seq_length, embedding_dimension)  (embedding_dimension, model_dimension)
# Q : seq_length, model_dimension
# K : seq_length, model_dimension
# V : seq_length, model_dimension
# (QK_T/sqrt(d_k)) * V
# Q K V 维度一致 => Attention 维度一致
# 根据 Head Number 降维 => model_dimension / head number
# 降维 changed_dimension
# (seq_length, model_dimension) => (seq_length, changed_dimension)
# head1 = Attention (Q1, K1, V1)
# head2 = Attention (Q2, K2, V2)
# ...
# head_h = Attention (Q_h, K_h, V_h)
# (seq_length, head_number, changed_dimension)
# (head_number, seq_length, changed_dimension) *
# (head_number, changed_dimension, seq_length) *
# (head_number, seq_length, changed_dimension)
# = (head_number, seq_length, changed_dimension)
# (seq_length, head_number, changed_dimension)
# (seq_length, changed_dimension * head_number )
# (seq_length, model_dimension)
# MultiHead Attention

from components import MyMultiHeadAttention
import torch
seq_length = 5
embedding_dimension = 512
model_dimension = 64
head_num = 4

X = torch.rand(5, embedding_dimension)
MultiHeadAttention = MyMultiHeadAttention(seq_length, embedding_dimension,
                     model_dimension, head_num)
output = MultiHeadAttention(X)
print(output.shape)

