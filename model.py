from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from collections import OrderedDict
import math
import numpy as np
import time




# def binary_cross_entropy_weight(y_pred, y,has_weight=False, weight_length=1, weight_max=10):
#     '''

#     :param y_pred:
#     :param y:
#     :param weight_length: how long until the end of sequence shall we add weight
#     :param weight_value: the magnitude that the weight is enhanced
#     :return:
#     '''
#     if has_weight:
#         weight = torch.ones(y.size(0),y.size(1),y.size(2))
#         weight_linear = torch.arange(1,weight_length+1)/weight_length*weight_max
#         weight_linear = weight_linear.view(1,weight_length,1).repeat(y.size(0),1,y.size(2))
#         weight[:,-1*weight_length:,:] = weight_linear
#         loss = F.binary_cross_entropy(y_pred, y, weight=weight.cuda())
#     else:
#         loss = F.binary_cross_entropy(y_pred, y)
#     return loss

def binary_cross_entropy_weight(y_pred, y, has_weight=False, weight_length=1, weight_max=10):
    '''
    :param y_pred: Model's raw logits (before sigmoid).
    :param y: Ground truth labels.
    :param has_weight: Whether to apply weighting.
    :param weight_length: Length of the sequence for applying weight.
    :param weight_max: Maximum weight value.
    :return: Weighted binary cross entropy loss.
    '''
    # Ensure input tensors are on the same device
    assert torch.all((y == 0) | (y == 1)), "Target values must be binary (0 or 1)"
    device = y_pred.device
    if has_weight:
        
        weight = torch.ones(y.size(0), y.size(1), y.size(2), device=device)
        print(weight.size(), y_pred.size(), y.size())  # Check if all sizes are the same
        weight_linear = torch.arange(1, weight_length + 1, device=device) / weight_length * weight_max
        weight_linear = weight_linear.view(1, weight_length, 1).repeat(y.size(0), 1, y.size(2))
        weight[:, -weight_length:, :] = weight_linear

        # Ensure no NaN or Inf values in y_pred or y
        assert torch.all(torch.isfinite(y_pred)), "y_pred contains NaN or Inf values"
        assert torch.all(torch.isfinite(y)), "y contains NaN or Inf values"
        
        # Apply binary cross entropy loss with the weight
        loss = F.binary_cross_entropy(y_pred, y, weight=weight)
    else:
        # Standard binary cross entropy loss without weight
        loss = F.binary_cross_entropy(y_pred, y)
    
    return loss

def masked_binary_cross_entropy(outputs, targets, lengths):
    """
    outputs: (B, T, F) — predicted
    targets: (B, T, F) — ground truth
    lengths: (B,) — actual lengths per sequence
    """
    B, T, Fe = outputs.shape

    # Create mask (B, T) — valid positions = 1, padding = 0
    mask = torch.zeros((B, T), device=outputs.device)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1

    # Expand mask to match (B, T, F)
    mask = mask.unsqueeze(-1).expand_as(outputs)

    # Compute element-wise BCE loss without reduction
    loss = F.binary_cross_entropy(outputs, targets, reduction='none')

    # Apply mask and average only over valid elements
    loss = loss * mask
    loss = loss.sum() / mask.sum()

    return loss

def sample_tensor(y,sample=True, thresh=0.5):
    # do sampling
    if sample:
        y_thresh = Variable(torch.rand(y.size())).cuda()
        y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size())*thresh).cuda()
        y_result = torch.gt(y, y_thresh).float()
    return y_result

def gumbel_softmax(logits, temperature, eps=1e-9):
    '''

    :param logits: shape: N*L
    :param temperature:
    :param eps:
    :return:
    '''
    # get gumbel noise
    noise = torch.rand(logits.size())
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    noise = Variable(noise).cuda()

    x = (logits + noise) / temperature
    x = F.softmax(x)
    return x

# for i in range(10):
#     x = Variable(torch.randn(1,10)).cuda()
#     y = gumbel_softmax(x, temperature=0.01)
#     print(x)
#     print(y)
#     _,id = y.topk(1)
#     print(id)


def gumbel_sigmoid(logits, temperature):
    '''

    :param logits:
    :param temperature:
    :param eps:
    :return:
    '''
    # get gumbel noise
    noise = torch.rand(logits.size()) # uniform(0,1)
    noise_logistic = torch.log(noise)-torch.log(1-noise) # logistic(0,1)
    noise = Variable(noise_logistic).cuda()

    x = (logits + noise) / temperature
    x = F.sigmoid(x)
    return x

# x = Variable(torch.randn(100)).cuda()
# y = gumbel_sigmoid(x,temperature=0.01)
# print(x)
# print(y)

def s_sample_sigmoid(y, sample=True, sample_time=1):
    if len(y.size()) == 2:
        y_thresh = torch.rand_like(y)
    elif len(y.size()) == 3:
        y_thresh = torch.rand(y.size(0), y.size(1), y.size(2)).to(y.device)
    else:
        raise ValueError(f"Unsupported tensor shape for sampling: {y.shape}")

    if sample:
        y_sampled = (y > y_thresh).float()
        return y_sampled
    else:
        return (y > 0.5).float()

def sample_sigmoid(y, sample, thresh=0.5, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''
    if isinstance(y, tuple):
        y = y[0]  # Extract the first element if y is a tuple
    # do sigmoid first
    y = F.sigmoid(y)
    # do sampling
    if sample:
        if sample_time>1:
            y_result = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).cuda()
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = Variable(torch.rand(y.size(1), y.size(2))).cuda()
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data>0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).cuda()
            y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2))*thresh).cuda()
        y_result = torch.gt(y, y_thresh).float()
    return y_result

def sample_sigmoid_attention(y, sample, thresh=0.5, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''
    if isinstance(y, tuple):
        y = y[0]  # Extract the first element if y is a tuple
    # do sigmoid first
    y = F.sigmoid(y)
    
    # Check dimensions and expand if necessary
    if y.dim() == 2:  # If y has only two dimensions
        y = y.unsqueeze(2)  # Expand to 3 dimensions: [batch_size, hidden_size, 1]

    # do sampling
    if sample:
        if sample_time>1:
            y_result = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).cuda()
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = Variable(torch.rand(y.size(1), y.size(2))).cuda()
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data>0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).cuda()
            y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2))*thresh).cuda()
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def sample_sigmoid_supervised(y_pred, y, current, y_len, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y_pred = F.sigmoid(y_pred)
    # do sampling
    y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2))).cuda()
    # loop over all batches
    for i in range(y_result.size(0)):
        # using supervision
        if current<y_len[i]:
            while True:
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).cuda()
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                # print('current',current)
                # print('y_result',y_result[i].data)
                # print('y',y[i])
                y_diff = y_result[i].data-y[i]
                if (y_diff>=0).all():
                    break
        # supervision done
        else:
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).cuda()
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data>0).any():
                    break
    return y_result

def sample_sigmoid_supervised_simple(y_pred, y, current, y_len, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y_pred = F.sigmoid(y_pred)
    # do sampling
    y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2))).cuda()
    # loop over all batches
    for i in range(y_result.size(0)):
        # using supervision
        if current<y_len[i]:
            y_result[i] = y[i]
        # supervision done
        else:
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).cuda()
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data>0).any():
                    break
    return y_result

################### current adopted model, LSTM+MLP || LSTM+VAE || LSTM+LSTM (where LSTM can be GRU as well)
#####
# definition of terms
# h: hidden state of LSTM
# y: edge prediction, model output
# n: noise for generator
# l: whether an output is real or not, binary

# class GRU_plain_with_attention(nn.Module):
#     def __init__(self, input_size, embedding_size, hidden_size, num_layers, 
#                  has_input=True, has_output=False, output_size=None):
#         super(GRU_plain_with_attention, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.has_input = has_input
#         self.has_output = has_output

#         if has_input:
#             self.input = nn.Linear(input_size, embedding_size)
#             self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, 
#                               num_layers=num_layers, batch_first=True)
#         else:
#             self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
#                               num_layers=num_layers, batch_first=True)
#         if has_output:
#             self.output = nn.Sequential(
#                 nn.Linear(128, embedding_size),  # Input size (128) matches context vector
#                 nn.ReLU(),
#                 nn.Linear(embedding_size, output_size)  # Output size is user-defined
#             )


#         self.attention = nn.Linear(hidden_size, 1)  # For computing attention scores
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#         # initialize
#         self.hidden = None  # need initialize before forward run
        
#         for name, param in self.rnn.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, 0.25)
#             elif 'weight' in name:
#                 nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('sigmoid'))
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

#     def init_hidden(self, batch_size):
#         return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()

#     def forward(self, input_raw, pack=False, input_len=None):
#         if self.has_input:
#             if isinstance(input_raw, PackedSequence):
#                 input_raw, _ = pad_packed_sequence(input_raw, batch_first=True)
#             print("Input to self.input:", input_raw.shape)
            
#             # Ensure input is reshaped correctly before passing to the self.input layer
#             if input_raw.dim() == 2:  # If it's a 2D tensor (batch_size, input_size)
#                 input_raw = input_raw.unsqueeze(1)  # Add sequence length dimension (batch_size, seq_len=1, input_size)
#             elif input_raw.dim() == 3:  # If it's a 3D tensor (batch_size, seq_len, input_size)
#                pass  # Already in the correct shape
            
#             input = self.input(input_raw)
#             input = self.relu(input)
#         else:
#             input = input_raw
#         if pack:
#             input = pack_padded_sequence(input, input_len, batch_first=True)
#         output_raw, self.hidden = self.rnn(input, self.hidden)
#         if pack:
#             output_raw, _ = pad_packed_sequence(output_raw, batch_first=True)

#         # Attention mechanism
#         attention_scores = self.attention(output_raw)  # Shape: (batch_size, seq_len, 1)
#         print("Attention scores shape:", attention_scores.shape)
#         attention_weights = self.softmax(attention_scores)  # Shape: (batch_size, seq_len, 1)
#         print("Attention weights shape:", attention_weights.shape)
#         context_vector = torch.sum(attention_weights * output_raw, dim=1)  # Shape: (batch_size, hidden_size)
#         print("Context vector shape:", context_vector.shape)
#         print("Expected input size for output layer:", self.output[0].in_features)
        
#         print("Context vector shape before output layer:", context_vector.shape)

#         if context_vector.dim() == 1:
#             context_vector = context_vector.unsqueeze(0)  # Add batch dimension if missing

#         if context_vector.size(1) != 128:
#             raise ValueError(f"Expected context vector with size [batch_size, 128], but got {context_vector.shape}")

#         if self.has_output:
#             print("Input to output layer:", context_vector.shape)  # Debug context vector shape
#             output_raw = self.output(context_vector)
#         else:
#             output_raw = context_vector

#         return output_raw, attention_weights  # Returning attention weights for interpretability

# plain LSTM model
class LSTM_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
        super(LSTM_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda(),
                Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda())

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw

# plain GRU model
class aGRU_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
        super(GRU_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            print("Shape of input_raw:", input_raw.shape)  # Should be (batch_size, *, 80)
            print("Expected weight shape:", self.input.weight.shape)  # Should be (80, 64)

            input = self.input(input_raw)
            input = self.relu(input)
        else:
            print("Shape of input_raw:", input_raw.shape)  # Should be (batch_size, *, 80)
            print("Expected weight shape:", self.input.weight.shape)  # Should be (80, 64)

            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw
class OutputModule(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size):
        super(OutputModule, self).__init__()
        self.output = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, output_size)
        )

    def forward(self, x):
        return self.output(x)
    
class GRU_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
        super(GRU_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw
    
    
class OutputModule(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size):
        super(OutputModule, self).__init__()
        self.output = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, output_size)
        )

    def forward(self, x):
        return self.output(x)
    
class GRU_plain_dec(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, output_size, has_input=True, has_output=True):
        super(GRU_plain_dec, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        # Input projection
        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.input_raw = nn.Linear(2, embedding_size)
            self.input_pred = nn.Linear(output_size, embedding_size)
            self.relu = nn.ReLU()
        
        # GRU
        self.rnn = nn.GRU(input_size=embedding_size + hidden_size,  # Concatenating graph_embedding
                          hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Output projection
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

       

        # Parameter initialization
        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('sigmoid'))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, graph_embedding):
        """
        Initializes the GRU hidden state using the graph embedding from GraphEncoder.
        
        Args:
            graph_embedding (Tensor): The output of GraphEncoder (shape: batch_size x hidden_size)
            
        Returns:
            Tensor: Initialized hidden state of shape (num_layers, batch_size, hidden_size)
        """
        # Expand graph_embedding to match GRU's hidden state dimensions
        hidden = graph_embedding.expand(self.num_layers, -1, -1).contiguous()  
        return hidden  # Shape: (num_layers, batch_size, hidden_size)

    def forward(self, edge_seq, graph_embedding, hidden=None):
        # Unpack the sequence
        if isinstance(edge_seq, PackedSequence):
            edge_seq, _ = pad_packed_sequence(edge_seq, batch_first=True)
        batch_size, seq_len, feat_dim = edge_seq.shape  # (32, 327, 2)
        # print(f"Edge seq shape before projection: {edge_seq.shape}")  # Debugging print
        # print(f"Expected input size: {self.input.in_features}")  # Debugging print
        # Ensure input size matches expected feature dimension
        #assert feat_dim == self.input.in_features, f"Expected {self.input.in_features}, got {feat_dim}"

        # Reshape to (batch_size * seq_len, input_size) before applying Linear layer
        edge_seq = edge_seq.view(-1, feat_dim)  # Shape: (32*327, 2) -> (10464, 2)

        # Apply input projection
        edge_seq = self.relu(self.input(edge_seq))  # Shape: (10464, embedding_size)

        # Reshape back to (batch_size, seq_len, embedding_size)
        edge_seq = edge_seq.view(batch_size, seq_len, -1)  # Shape: (32, 327, embedding_size)

        # Concatenate input with graph embedding
        graph_embedding = graph_embedding.squeeze(0)  # Ensure correct shape
        graph_embedding = graph_embedding.unsqueeze(1).expand(edge_seq.shape[0], edge_seq.shape[1], -1)
        
        input_seq = torch.cat([edge_seq, graph_embedding], dim=-1)
        # print(f"input_seq shape: {input_seq.shape}")
        # print(hidden.shape)
        # RNN forward pass
        output_seq, hidden = self.rnn(input_seq, hidden)

        # Pass through output layer if applicable
        if self.has_output:
            # print(f"Output seq shape before Linear layer: {output_seq.shape}")  
            # print(f"Expected Linear layer input features: {self.output[0].in_features}")  

            batch_size, seq_len, hidden_dim = output_seq.shape  # (32, 259, 128)
    
            # Flatten output_seq to (batch_size * seq_len, hidden_dim) before passing to Linear
            output_seq = output_seq.reshape(-1, hidden_dim) 
            # Apply output projection layers
            output_seq = self.output(output_seq)
            
            # Reshape back to original sequence format
            output_seq = output_seq.reshape(batch_size, seq_len, -1)  

        return output_seq, hidden.detach()

class aGRU_flat_dec_multihead(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, 
                 edge_output_size, value_output_size, has_input=True):
        super(GRU_flat_dec_multihead, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.relu = nn.ReLU()

        self.rnn = nn.GRU(input_size=embedding_size + hidden_size, 
                          hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Multi-head outputs
        self.output_edge = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, edge_output_size)
        )

        self.output_value = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, value_output_size)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('sigmoid'))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, graph_embedding):
        return graph_embedding.expand(self.num_layers, -1, -1).contiguous()

    
    def forward(self, edge_seq, graph_embedding, hidden=None, pack=False, input_len=None):
        # 1) cuDNN-friendly flatten
        self.rnn.flatten_parameters()

        # 2) Hidden‐state init if needed
        if hidden is None:
            hidden = self.init_hidden(graph_embedding)  # [num_layers, B, H]

        # 3) Unpack or treat raw
        if pack:
            # if already packed, unpack it first
            if isinstance(edge_seq, PackedSequence):
                seq_unpacked, seq_lengths = pad_packed_sequence(edge_seq, batch_first=True)
            else:
                assert input_len is not None, "Must provide input_len when pack=True"
                seq_unpacked, seq_lengths = edge_seq, input_len
        else:
            seq_unpacked, seq_lengths = edge_seq, None  # no packing

        B, T, feat = seq_unpacked.size()
        assert feat == self.input.in_features, \
            f"Expected feature dim {self.input.in_features}, got {feat}"

        # 4) Input embedding
        flat = self.relu(self.input(seq_unpacked.reshape(-1, feat)))
        embedded = flat.view(B, T, -1)

        # 5) Graph‐context expansion
        if graph_embedding.dim() == 2:
            graph_embedding = graph_embedding.unsqueeze(0)  # [1, B, H]
        context = graph_embedding[-1].unsqueeze(1).expand(B, T, -1)

        # 6) Concatenate and (optionally) pack
        rnn_in = torch.cat([embedded, context], dim=-1)
        if pack:
            rnn_in = pack_padded_sequence(rnn_in, seq_lengths, batch_first=True, enforce_sorted=True)

        # 7) GRU forward
        out, hidden = self.rnn(rnn_in, hidden)

        # 8) Unpack if needed
        if pack:
            out, _ = pad_packed_sequence(out, batch_first=True)

        # 9) Multi‐head projections
        edge_logits = self.output_edge(out)      # [B, T, edge_output_size]
        value_preds = self.output_value(out)     # [B, T, value_output_size]

        # 10) Re‐pack outputs, if in training mode
        if pack:
            packed_logits = pack_padded_sequence(edge_logits, seq_lengths, batch_first=True, enforce_sorted=True)
            packed_values = pack_padded_sequence(value_preds, seq_lengths, batch_first=True, enforce_sorted=True)
            return packed_logits, packed_values, hidden

        # 11) Generation mode: return raw logits
        return edge_logits, value_preds, hidden
    
    # def forward(self, edge_seq, graph_embedding, hidden=None, pack=False, input_len=None):
    #     # 1) flatten for cuDNN
    #     self.rnn.flatten_parameters()
    #     # do:
    #     if hidden is None:
    #         hidden = self.init_hidden(graph_embedding)
    #     # 2) unpack or use raw
    #     if isinstance(edge_seq, PackedSequence):
    #         seq_unpacked, seq_lengths = pad_packed_sequence(edge_seq, batch_first=True)
    #     else:
    #         assert input_len is not None
    #         seq_unpacked, seq_lengths = edge_seq, input_len

    #     B, T, feat = seq_unpacked.size()
    #     assert feat == self.input.in_features

    #     # 3) embed edges
    #     flat = self.relu(self.input(seq_unpacked.view(-1, feat)))
    #     embedded = flat.view(B, T, -1)

    #     # 4) expand graph embedding
    #     if graph_embedding.dim() == 2:
    #         graph_embedding = graph_embedding.unsqueeze(0)
    #     context = graph_embedding[-1].unsqueeze(1).expand(B, T, -1)

    #     # 5) concat & pack
    #     rnn_in = torch.cat([embedded, context], dim=-1)
    #     packed_in = pack_padded_sequence(rnn_in, seq_lengths, batch_first=True, enforce_sorted=True)

    #     # 6) forward GRU
    #     packed_out, hidden = self.rnn(packed_in, hidden)

    #     # 7) unpack & project
    #     out, _ = pad_packed_sequence(packed_out, batch_first=True)
    #     edge_logits = self.output_edge(out)
    #     value_preds = self.output_value(out)

    #     # 8) repack if needed
    #     packed_logits = pack_padded_sequence(edge_logits, seq_lengths, batch_first=True, enforce_sorted=True)
    #     packed_values = pack_padded_sequence(value_preds, seq_lengths, batch_first=True, enforce_sorted=True)

    #     return packed_logits, packed_values, hidden

class GRU_flat_dec_multihead(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers,
                 edge_output_size, value_output_size, has_input=True):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.relu = nn.ReLU()

        self.rnn = nn.GRU(input_size=embedding_size + hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        self.output_edge = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, edge_output_size)
        )
        self.output_value = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, value_output_size)
        )
        self._init_weights()

    def _init_weights(self):
        for n, p in self.rnn.named_parameters():
            if 'bias' in n:
                nn.init.constant_(p, 0.17)
            else:
                nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        
    def init_hidden(self, graph_embedding):
        # graph_embedding: [B, H] → hidden: [num_layers, B, H]
        return graph_embedding.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()

    def forward(self, edge_seq, graph_embedding, hidden=None, pack=False, input_len=None):
        # 1) cuDNN
        self.rnn.flatten_parameters()

        # 2) hidden init
        if hidden is None:
            hidden = self.init_hidden(graph_embedding)

        # 3) unpack/pack logic
        if pack:
            if isinstance(edge_seq, PackedSequence):
                seq_unpacked, seq_lengths = pad_packed_sequence(edge_seq, batch_first=True)
            else:
                assert input_len is not None, "Need input_len when pack=True"
                seq_unpacked, seq_lengths = edge_seq, input_len
        else:
            seq_unpacked, seq_lengths = edge_seq, None

        B, T, feat = seq_unpacked.size()
        assert feat == self.input.in_features, f"Expected feat={self.input.in_features}, got {feat}"

        # 4) embed inputs
        flat = self.relu(self.input(seq_unpacked.view(-1, feat)))
        embedded = flat.view(B, T, -1)

        # 5) expand context
        if graph_embedding.dim() == 2:
            graph_embedding = graph_embedding.unsqueeze(0)
        context = graph_embedding[-1].unsqueeze(1).expand(B, T, -1)

        # 6) concat + optional pack
        rnn_in = torch.cat([embedded, context], dim=-1)
        if pack:
            rnn_in = pack_padded_sequence(rnn_in, seq_lengths, batch_first=True, enforce_sorted=True)

        # 7) GRU
        out, hidden = self.rnn(rnn_in, hidden)

        # 8) unpack if needed
        if pack:
            out, _ = pad_packed_sequence(out, batch_first=True)

        # 9) project
        edge_logits = self.output_edge(out)       # [B, T, E_prev]
        value_preds = self.output_value(out)      # [B, T, 1]

        if pack:
            pl = pack_padded_sequence(edge_logits, seq_lengths, batch_first=True, enforce_sorted=True)
            pv = pack_padded_sequence(value_preds, seq_lengths, batch_first=True, enforce_sorted=True)
            return pl, pv, hidden

        return edge_logits, value_preds, hidden
   
    
    
class GRUWithAttention(nn.Module):
    """
    A GRU decoder with Bahdanau-style (additive) attention.
    """
    def __init__(self, input_size, hidden_size, attn_size, output_size, num_layers=1):
        super().__init__()
        # store these so init_hidden works
        self.num_layers   = num_layers
        self.hidden_size  = hidden_size
        # GRU cell for decoder
        self.gru = nn.GRU(input_size + hidden_size, hidden_size,
                          num_layers=num_layers, batch_first=True)  # :contentReference[oaicite:4]{index=4}

        # Attention: score = v^T tanh(W_a [dec_hidden; enc_outputs])
        self.attn = nn.Linear(hidden_size * 2, attn_size)            # :contentReference[oaicite:5]{index=5}
        self.v = nn.Linear(attn_size, 1, bias=False)

        # Final output projection
        self.out = nn.Linear(hidden_size * 2, output_size)

    def init_hidden(self, batch_size, device=None):
        """
        Initialize hidden state to zeros.
        Returns: tensor of shape (num_layers, batch_size, hidden_size)
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size,
                        device=device)
    
    def forward(self, dec_input, hidden, encoder_outputs, mask):
        """
        dec_input:    (batch, 1, input_size) — current step input
        hidden:       (num_layers, batch, hidden_size) — previous decoder hidden
        encoder_outputs: (batch, seq_len, hidden_size) — all encoder states
        mask:         (batch, seq_len) — 1 for valid, 0 for padding
        """
        # 1) Compute attention scores :contentReference[oaicite:6]{index=6}
        #    Expand decoder hidden to compare with every encoder time step
        dec_h = hidden[-1].unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat([dec_h, encoder_outputs], dim=2)))
        scores = self.v(energy).squeeze(2)  # (batch, seq_len)

        # 2) Mask padding positions and normalize :contentReference[oaicite:7]{index=7}
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # 3) Compute context vector as weighted sum :contentReference[oaicite:8]{index=8}
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        #    context: (batch, 1, hidden_size)

        # 4) Concatenate input and context, run through GRU
        gru_input = torch.cat([dec_input, context], dim=2)
        output, hidden = self.gru(gru_input, hidden)  # (batch, 1, hidden)

        # 5) Predict next token / edge logits
        output = torch.cat([output, context], dim=2)   # (batch, 1, hidden*2)
        output = self.out(output)                      # :contentReference[oaicite:9]{index=9}

        return output, hidden, attn_weights    
    
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)
        # Define the projection layer to match the expected hidden size (128)
        self.projection_layer = nn.Linear(128, 128)  # Transform 16 -> 128

    def forward(self, rnn_output):
        """
        rnn_output: Tensor of shape (batch_size, seq_len, hidden_size)
        Returns:
        context_vector: Weighted sum of RNN outputs (batch_size, hidden_size)
        """
        
        if isinstance(rnn_output, PackedSequence):
            rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)

        # Compute attention scores
        scores = self.attention_weights(rnn_output)  # Shape: [batch_size, seq_len, 1]
        scores = scores.squeeze(-1)  # Shape: [batch_size, seq_len]

        # Normalize attention scores using softmax
        attention_weights = F.softmax(scores, dim=1)  # Shape: [batch_size, seq_len]

        # Compute the context vector by performing a weighted sum of the RNN outputs
        context_vector = torch.sum(attention_weights.unsqueeze(2) * rnn_output, dim=1)  # Shape: [batch_size, hidden_size]

        return context_vector, scores



class GRU_plain_with_attention(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
        super(GRU_plain_with_attention, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #self.batch_size = 32
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )
            
        # Add attention
        self.attention = Attention(hidden_size)
        # Add Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            print("Shape of input_raw:", input_raw.shape)  # Should be (batch_size, *, 80)
            print("Expected weight shape:", self.input.weight.shape)  # Should be (80, 64)
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            print("Shape of input_raw:", input_raw.shape)  # Should be (batch_size, *, 80)
            print("Expected weight shape:", self.input.weight.shape)  # Should be (80, 64)
            input = input_raw
        self.hidden = self.init_hidden(batch_size=32)
        
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        #batch_size, seq_len, hidden_size = output_raw.size()  # Get the correct batch size from output_raw
        #print(f"Shape of self.hidden before RNN: {self.hidden.size()}")

        output_raw, self.hidden = self.rnn(input, self.hidden)
        #print(f"Shape of self.hidden after RNN: {self.hidden.size()}")
        if pack:
        # Unpack the sequence if it was packed
            output_raw, lengths = pad_packed_sequence(output_raw, batch_first=True)
       

        # Now apply attention on the unpacked sequence (output_raw is now a tensor, not PackedSequence)
        context_vector, attention_scores = self.attention(output_raw)  # context_vector: (batch_size, hidden_size)
    
        # Apply Layer Normalization
        output_raw = self.layer_norm(output_raw)
        #print(output_raw.shape)  # Check the shape of rnn_output
        
        # # Apply attention
        # context_vector, attention_scores = self.attention(output_raw)  # context_vector: (batch_size, hidden_size)


        if self.has_output:
            context_vector = self.output(context_vector)  # (batch_size, output_size)
        #print(f"Shape of self.hidden after RNN: {self.hidden.size()}")

        return context_vector, attention_scores


 # Unpack the sequence if it's packed
        #output_raw, lengths = pad_packed_sequence(output_raw, batch_first=True)
        #print(output_raw.shape)  # Check the shape of rnn_output
        # Apply attention
        #context_vector, attention_scores = self.attention(output_raw)  # context_vector: (batch_size, hidden_size)
        # if pack:
        #     output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        
        #print(output_raw.shape)  # Check the shape of rnn_output
        
        # Extract batch_size and seq_len dynamically from output_raw shape
        #batch_size, seq_len, hidden_size = output_raw.size()


    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_conditional_plain, self).__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size)  # mu
        self.encode_12 = nn.Linear(h_size, embedding_size)  # lsgms

        self.decode_1 = nn.Linear(embedding_size+h_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size)  # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.size(0), z_sgm.size(1), z_sgm.size(2))).cuda()
        z = eps * z_sgm + z_mu
        # decoder
        y = self.decode_1(torch.cat((h,z),dim=2))
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms




########### baseline model 1: Learning deep generative model of graphs

class DGM_graphs(nn.Module):
    def __init__(self,h_size):
        # h_size: node embedding size
        # h_size*2: graph embedding size

        super(DGM_graphs, self).__init__()
        ### all modules used by the model
        ## 1 message passing, 2 times
        self.m_uv_1 = nn.Linear(h_size*2, h_size*2)
        self.f_n_1 = nn.GRUCell(h_size*2, h_size) # input_size, hidden_size

        self.m_uv_2 = nn.Linear(h_size * 2, h_size * 2)
        self.f_n_2 = nn.GRUCell(h_size * 2, h_size)  # input_size, hidden_size

        ## 2 graph embedding and new node embedding
        # for graph embedding
        self.f_m = nn.Linear(h_size, h_size*2)
        self.f_gate = nn.Sequential(
            nn.Linear(h_size,1),
            nn.Sigmoid()
        )
        # for new node embedding
        self.f_m_init = nn.Linear(h_size, h_size*2)
        self.f_gate_init = nn.Sequential(
            nn.Linear(h_size,1),
            nn.Sigmoid()
        )
        self.f_init = nn.Linear(h_size*2, h_size)

        ## 3 f_addnode
        self.f_an = nn.Sequential(
            nn.Linear(h_size*2,1),
            nn.Sigmoid()
        )

        ## 4 f_addedge
        self.f_ae = nn.Sequential(
            nn.Linear(h_size * 2, 1),
            nn.Sigmoid()
        )

        ## 5 f_nodes
        self.f_s = nn.Linear(h_size*2, 1)




def message_passing(node_neighbor, node_embedding, model):
    node_embedding_new = []
    for i in range(len(node_neighbor)):
        neighbor_num = len(node_neighbor[i])
        if neighbor_num > 0:
            node_self = node_embedding[i].expand(neighbor_num, node_embedding[i].size(1))
            node_self_neighbor = torch.cat([node_embedding[j] for j in node_neighbor[i]], dim=0)
            message = torch.sum(model.m_uv_1(torch.cat((node_self, node_self_neighbor), dim=1)), dim=0, keepdim=True)
            node_embedding_new.append(model.f_n_1(message, node_embedding[i]))
        else:
            message_null = Variable(torch.zeros((node_embedding[i].size(0),node_embedding[i].size(1)*2))).cuda()
            node_embedding_new.append(model.f_n_1(message_null, node_embedding[i]))
    node_embedding = node_embedding_new
    node_embedding_new = []
    for i in range(len(node_neighbor)):
        neighbor_num = len(node_neighbor[i])
        if neighbor_num > 0:
            node_self = node_embedding[i].expand(neighbor_num, node_embedding[i].size(1))
            node_self_neighbor = torch.cat([node_embedding[j] for j in node_neighbor[i]], dim=0)
            message = torch.sum(model.m_uv_1(torch.cat((node_self, node_self_neighbor), dim=1)), dim=0, keepdim=True)
            node_embedding_new.append(model.f_n_1(message, node_embedding[i]))
        else:
            message_null = Variable(torch.zeros((node_embedding[i].size(0), node_embedding[i].size(1) * 2))).cuda()
            node_embedding_new.append(model.f_n_1(message_null, node_embedding[i]))
    return node_embedding_new



def calc_graph_embedding(node_embedding_cat, model):

    node_embedding_graph = model.f_m(node_embedding_cat)
    node_embedding_graph_gate = model.f_gate(node_embedding_cat)
    graph_embedding = torch.sum(torch.mul(node_embedding_graph, node_embedding_graph_gate), dim=0, keepdim=True)
    return graph_embedding


def calc_init_embedding(node_embedding_cat, model):
    node_embedding_init = model.f_m_init(node_embedding_cat)
    node_embedding_init_gate = model.f_gate_init(node_embedding_cat)
    init_embedding = torch.sum(torch.mul(node_embedding_init, node_embedding_init_gate), dim=0, keepdim=True)
    init_embedding = model.f_init(init_embedding)
    return init_embedding



























################################################## code that are NOT used for final version #############


# RNN that updates according to graph structure, new proposed model
class Graph_RNN_structure(nn.Module):
    def __init__(self, hidden_size, batch_size, output_size, num_layers, is_dilation=True, is_bn=True):
        super(Graph_RNN_structure, self).__init__()
        ## model configuration
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.num_layers = num_layers # num_layers of cnn_output
        self.is_bn=is_bn

        ## model
        self.relu = nn.ReLU()
        # self.linear_output = nn.Linear(hidden_size, 1)
        # self.linear_output_simple = nn.Linear(hidden_size, output_size)
        # for state transition use only, input is null
        # self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # use CNN to produce output prediction
        # self.cnn_output = nn.Sequential(
        #     nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=1, padding=1),
        #     # nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_size, 1, kernel_size=3, dilation=1, padding=1)
        # )

        if is_dilation:
            self.conv_block = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=2**i, padding=2**i) for i in range(num_layers-1)])
        else:
            self.conv_block = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=1, padding=1) for i in range(num_layers-1)])
        self.bn_block = nn.ModuleList([nn.BatchNorm1d(hidden_size) for i in range(num_layers-1)])
        self.conv_out = nn.Conv1d(hidden_size, 1, kernel_size=3, dilation=1, padding=1)


        # # use CNN to do state transition
        # self.cnn_transition = nn.Sequential(
        #     nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=1, padding=1),
        #     # nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=1, padding=1)
        # )

        # use linear to do transition, same as GCN mean aggregator
        self.linear_transition = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU()
        )


        # GRU based output, output a single edge prediction at a time
        # self.gru_output = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # use a list to keep all generated hidden vectors, each hidden has size batch*hidden_dim*1, and the list size is expanding
        # when using convolution to compute attention weight, we need to first concat the list into a pytorch variable: batch*hidden_dim*current_num_nodes
        self.hidden_all = []

        ## initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # print('linear')
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                # print(m.weight.data.size())
            if isinstance(m, nn.Conv1d):
                # print('conv1d')
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                # print(m.weight.data.size())
            if isinstance(m, nn.BatchNorm1d):
                # print('batchnorm1d')
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # print(m.weight.data.size())
            if isinstance(m, nn.GRU):
                # print('gru')
                m.weight_ih_l0.data = init.xavier_uniform(m.weight_ih_l0.data,
                                                                  gain=nn.init.calculate_gain('sigmoid'))
                m.weight_hh_l0.data = init.xavier_uniform(m.weight_hh_l0.data,
                                                                  gain=nn.init.calculate_gain('sigmoid'))
                m.bias_ih_l0.data = torch.ones(m.bias_ih_l0.data.size(0)) * 0.25
                m.bias_hh_l0.data = torch.ones(m.bias_hh_l0.data.size(0)) * 0.25

    def init_hidden(self,len=None):
        if len is None:
            return Variable(torch.ones(self.batch_size, self.hidden_size, 1)).cuda()
        else:
            hidden_list = []
            for i in range(len):
                hidden_list.append(Variable(torch.ones(self.batch_size, self.hidden_size, 1)).cuda())
            return hidden_list

    # only run a single forward step
    def forward(self, x, teacher_forcing, temperature = 0.5, bptt=True,bptt_len=20, flexible=True,max_prev_node=100):
        # x: batch*1*self.output_size, the groud truth
        # todo: current only look back to self.output_size nodes, try to look back according to bfs sequence

        # 1 first compute new state
        # print('hidden_all', self.hidden_all[-1*self.output_size:])
        # hidden_all_cat = torch.cat(self.hidden_all[-1*self.output_size:], dim=2)

        # # # add BPTT, detach the first variable
        # if bptt:
        #     self.hidden_all[0] = Variable(self.hidden_all[0].data).cuda()

        hidden_all_cat = torch.cat(self.hidden_all, dim=2)
        # print(hidden_all_cat.size())

        # print('hidden_all_cat',hidden_all_cat.size())
        # att_weight size: batch*1*current_num_nodes
        for i in range(self.num_layers-1):
            hidden_all_cat = self.conv_block[i](hidden_all_cat)
            if self.is_bn:
                hidden_all_cat = self.bn_block[i](hidden_all_cat)
            hidden_all_cat = self.relu(hidden_all_cat)
        x_pred = self.conv_out(hidden_all_cat)
        # 2 then compute output, using a gru
        # first try the simple version, directly give the edge prediction
        # x_pred = self.linear_output_simple(hidden_new)
        # x_pred = x_pred.view(x_pred.size(0),1,x_pred.size(1))

        # todo: use a gru version output
        # if sample==False:
        #     # when training: we know the ground truth, input the sequence at once
        #     y_pred,_ = self.gru_output(x, hidden_new.permute(2,0,1))
        #     y_pred = self.linear_output(y_pred)
        # else:
        #     # when validating, we need to sampling at each time step
        #     y_pred = Variable(torch.zeros(x.size(0), x.size(1), x.size(2))).cuda()
        #     y_pred_long = Variable(torch.zeros(x.size(0), x.size(1), x.size(2))).cuda()
        #     x_step = x[:, 0:1, :]
        #     for i in range(x.size(1)):
        #         y_step,_ = self.gru_output(x_step)
        #         y_step = self.linear_output(y_step)
        #         y_pred[:, i, :] = y_step
        #         y_step = F.sigmoid(y_step)
        #         x_step = sample(y_step, sample=True, thresh=0.45)
        #         y_pred_long[:, i, :] = x_step
        #     pass


        # 3 then update self.hidden_all list
        # i.e., model will use ground truth to update new node
        # x_pred_sample = gumbel_sigmoid(x_pred, temperature=temperature)
        x_pred_sample = sample_tensor(F.sigmoid(x_pred),sample=True)
        thresh = 0.5
        x_thresh = Variable(torch.ones(x_pred_sample.size(0), x_pred_sample.size(1), x_pred_sample.size(2)) * thresh).cuda()
        x_pred_sample_long = torch.gt(x_pred_sample, x_thresh).long()
        if teacher_forcing:
            # first mask previous hidden states
            hidden_all_cat_select = hidden_all_cat*x
            x_sum = torch.sum(x, dim=2, keepdim=True).float()

        # i.e., the model will use it's own prediction to attend
        else:
            # first mask previous hidden states
            hidden_all_cat_select = hidden_all_cat*x_pred_sample
            x_sum = torch.sum(x_pred_sample_long, dim=2, keepdim=True).float()

        # update hidden vector for new nodes
        hidden_new = torch.sum(hidden_all_cat_select, dim=2, keepdim=True) / x_sum

        hidden_new = self.linear_transition(hidden_new.permute(0, 2, 1))
        hidden_new = hidden_new.permute(0, 2, 1)

        if flexible:
            # use ground truth to maintaing history state
            if teacher_forcing:
                x_id = torch.min(torch.nonzero(torch.squeeze(x.data)))
                self.hidden_all = self.hidden_all[x_id:]
            # use prediction to maintaing history state
            else:
                x_id = torch.min(torch.nonzero(torch.squeeze(x_pred_sample_long.data)))
                start = max(len(self.hidden_all)-max_prev_node+1, x_id)
                self.hidden_all = self.hidden_all[start:]

        # maintaing a fixed size history state
        else:
            # self.hidden_all.pop(0)
            self.hidden_all = self.hidden_all[1:]

        self.hidden_all.append(hidden_new)

        # 4 return prediction
        # print('x_pred',x_pred)
        # print('x_pred_mean', torch.mean(x_pred))
        # print('x_pred_sample_mean', torch.mean(x_pred_sample))
        return x_pred, x_pred_sample

# batch_size = 8
# output_size = 4
# generator = Graph_RNN_structure(hidden_size=16, batch_size=batch_size, output_size=output_size, num_layers=1).cuda()
# for i in range(4):
#     generator.hidden_all.append(generator.init_hidden())
#
# x = Variable(torch.rand(batch_size,1,output_size)).cuda()
# x_pred = generator(x,teacher_forcing=True, sample=True)
# print(x_pred)




# current baseline model, generating a graph by lstm
class Graph_generator_LSTM(nn.Module):
    def __init__(self,feature_size, input_size, hidden_size, output_size, batch_size, num_layers):
        super(Graph_generator_LSTM, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear_input = nn.Linear(feature_size, input_size)
        self.linear_output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        # initialize
        # self.hidden,self.cell = self.init_hidden()
        self.hidden = self.init_hidden()

        self.lstm.weight_ih_l0.data = init.xavier_uniform(self.lstm.weight_ih_l0.data, gain=nn.init.calculate_gain('sigmoid'))
        self.lstm.weight_hh_l0.data = init.xavier_uniform(self.lstm.weight_hh_l0.data, gain=nn.init.calculate_gain('sigmoid'))
        self.lstm.bias_ih_l0.data = torch.ones(self.lstm.bias_ih_l0.data.size(0))*0.25
        self.lstm.bias_hh_l0.data = torch.ones(self.lstm.bias_hh_l0.data.size(0))*0.25
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data,gain=nn.init.calculate_gain('relu'))
    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers,self.batch_size, self.hidden_size)).cuda(), Variable(torch.zeros(self.num_layers,self.batch_size, self.hidden_size)).cuda())


    def forward(self, input_raw, pack=False,len=None):
        input = self.linear_input(input_raw)
        input = self.relu(input)
        if pack:
            input = pack_padded_sequence(input, len, batch_first=True)
        output_raw, self.hidden = self.lstm(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        output = self.linear_output(output_raw)
        return output






# a simple MLP generator output
class Graph_generator_LSTM_output_generator(nn.Module):
    def __init__(self,h_size, n_size, y_size):
        super(Graph_generator_LSTM_output_generator, self).__init__()
        # one layer MLP
        self.generator_output = nn.Sequential(
            nn.Linear(h_size+n_size, 64),
            nn.ReLU(),
            nn.Linear(64, y_size),
            nn.Sigmoid()
        )
    def forward(self,h,n,temperature):
        y_cat = torch.cat((h,n), dim=2)
        y = self.generator_output(y_cat)
        # y = gumbel_sigmoid(y,temperature=temperature)
        return y

# a simple MLP discriminator
class Graph_generator_LSTM_output_discriminator(nn.Module):
    def __init__(self, h_size, y_size):
        super(Graph_generator_LSTM_output_discriminator, self).__init__()
        # one layer MLP
        self.discriminator_output = nn.Sequential(
            nn.Linear(h_size+y_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self,h,y):
        y_cat = torch.cat((h,y),dim=2)
        l = self.discriminator_output(y_cat)
        return l



# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        # self.relu = nn.ReLU()
    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y,self.weight)
        return y


# vanilla GCN encoder
class GCN_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_encoder, self).__init__()
        self.conv1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.conv2 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        # self.bn1 = nn.BatchNorm1d(output_dim)
        # self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                # init_range = np.sqrt(6.0 / (m.input_dim + m.output_dim))
                # m.weight.data = torch.rand([m.input_dim, m.output_dim]).cuda()*init_range
                # print('find!')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self,x,adj):
        x = self.conv1(x,adj)
        # x = x/torch.sum(x, dim=2, keepdim=True)
        x = self.relu(x)
        # x = self.bn1(x)
        x = self.conv2(x,adj)
        # x = x / torch.sum(x, dim=2, keepdim=True)
        return x
# vanilla GCN decoder
class GCN_decoder(nn.Module):
    def __init__(self):
        super(GCN_decoder, self).__init__()
        # self.act = nn.Sigmoid()
    def forward(self,x):
        # x_t = x.view(-1,x.size(2),x.size(1))
        x_t = x.permute(0,2,1)
        # print('x',x)
        # print('x_t',x_t)
        y = torch.matmul(x, x_t)
        return y


# GCN based graph embedding
# allowing for arbitrary num of nodes
class GCN_encoder_graph(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim,num_layers):
        super(GCN_encoder_graph, self).__init__()
        self.num_layers = num_layers
        self.conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        # self.conv_hidden1 = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        # self.conv_hidden2 = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.conv_block = nn.ModuleList([GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for i in range(num_layers)])
        self.conv_last = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        self.act = nn.ReLU()
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                # init_range = np.sqrt(6.0 / (m.input_dim + m.output_dim))
                # m.weight.data = torch.rand([m.input_dim, m.output_dim]).cuda()*init_range
                # print('find!')
    def forward(self,x,adj):
        x = self.conv_first(x,adj)
        x = self.act(x)
        out_all = []
        out, _ = torch.max(x, dim=1, keepdim=True)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            out,_ = torch.max(x, dim=1, keepdim = True)
            out_all.append(out)
        x = self.conv_last(x,adj)
        x = self.act(x)
        out,_ = torch.max(x, dim=1, keepdim = True)
        out_all.append(out)
        output = torch.cat(out_all, dim = 1)
        output = output.permute(1,0,2)
        # print(out)
        return output

# x = Variable(torch.rand(1,8,10)).cuda()
# adj = Variable(torch.rand(1,8,8)).cuda()
# model = GCN_encoder_graph(10,10,10).cuda()
# y = model(x,adj)
# print(y.size())


def preprocess(A):
    # Get size of the adjacency matrix
    size = A.size(1)
    # Get the degrees for each node
    degrees = torch.sum(A, dim=2)

    # Create diagonal matrix D from the degrees of the nodes
    D = Variable(torch.zeros(A.size(0),A.size(1),A.size(2))).cuda()
    for i in range(D.size(0)):
        D[i, :, :] = torch.diag(torch.pow(degrees[i,:], -0.5))
    # Cholesky decomposition of D
    # D = np.linalg.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    # D = np.linalg.inv(D)
    # Create an identity matrix of size x size
    # Create A hat
    # Return A_hat
    A_normal = torch.matmul(torch.matmul(D,A), D)
    # print(A_normal)
    return A_normal



# a sequential GCN model, GCN with n layers
class GCN_generator(nn.Module):
    def __init__(self, hidden_dim):
        super(GCN_generator, self).__init__()
        # todo: add an linear_input module to map the input feature into 'hidden_dim'
        self.conv = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.act = nn.ReLU()
        # initialize
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self,x,teacher_force=False,adj_real=None):
        # x: batch * node_num * feature
        batch_num = x.size(0)
        node_num = x.size(1)
        adj = Variable(torch.eye(node_num).view(1,node_num,node_num).repeat(batch_num,1,1)).cuda()
        adj_output = Variable(torch.eye(node_num).view(1,node_num,node_num).repeat(batch_num,1,1)).cuda()

        # do GCN n times
        # todo: try if residual connections are plausible
        # todo: add higher order of adj (adj^2, adj^3, ...)
        # todo: try if norm everytim is plausible

        # first do GCN 1 time to preprocess the raw features

        # x_new = self.conv(x, adj)
        # x_new = self.act(x_new)
        # x = x + x_new

        x = self.conv(x, adj)
        x = self.act(x)

        # x = x / torch.norm(x, p=2, dim=2, keepdim=True)
        # then do GCN rest n-1 times
        for i in range(1, node_num):
            # 1 calc prob of a new edge, output the result in adj_output
            x_last = x[:,i:i+1,:].clone()
            x_prev = x[:,0:i,:].clone()
            x_prev = x_prev
            x_last = x_last
            prob = x_prev @ x_last.permute(0,2,1)
            adj_output[:,i,0:i] = prob.permute(0,2,1).clone()
            adj_output[:,0:i,i] = prob.clone()
            # 2 update adj
            if teacher_force:
                adj = Variable(torch.eye(node_num).view(1, node_num, node_num).repeat(batch_num, 1, 1)).cuda()
                adj[:,0:i+1,0:i+1] = adj_real[:,0:i+1,0:i+1].clone()
            else:
                adj[:, i, 0:i] = prob.permute(0,2,1).clone()
                adj[:, 0:i, i] = prob.clone()
            adj = preprocess(adj)
            # print(adj)
            # print(adj.min().data[0],adj.max().data[0])
            # print(x.min().data[0],x.max().data[0])
            # 3 do graph conv, with residual connection
            # x_new = self.conv(x, adj)
            # x_new = self.act(x_new)
            # x = x + x_new

            x = self.conv(x, adj)
            x = self.act(x)

            # x = x / torch.norm(x, p=2, dim=2, keepdim=True)
        # one = Variable(torch.ones(adj_output.size(0), adj_output.size(1), adj_output.size(2)) * 1.00).cuda().float()
        # two = Variable(torch.ones(adj_output.size(0), adj_output.size(1), adj_output.size(2)) * 2.01).cuda().float()
        # adj_output = (adj_output + one) / two
        # print(adj_output.max().data[0], adj_output.min().data[0])
        return adj_output


# #### test code ####
# print('teacher forcing')
# # print('no teacher forcing')
#
# start = time.time()
# generator = GCN_generator(hidden_dim=4)
# end = time.time()
# print('model build time', end-start)
# for run in range(10):
#     for i in [500]:
#         for batch in [1,10,100]:
#             start = time.time()
#             torch.manual_seed(123)
#             x = Variable(torch.rand(batch,i,4)).cuda()
#             adj = Variable(torch.eye(i).view(1,i,i).repeat(batch,1,1)).cuda()
#             # print('x', x)
#             # print('adj', adj)
#
#             # y = generator(x)
#             y = generator(x,True,adj)
#             # print('y',y)
#             end = time.time()
#             print('node num', i, '  batch size',batch, '  run time', end-start)




class CNN_decoder(nn.Module):
    def __init__(self, input_size, output_size, stride = 2):

        super(CNN_decoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.deconv1_1 = nn.ConvTranspose1d(in_channels=int(self.input_size), out_channels=int(self.input_size/2), kernel_size=3, stride=stride)
        self.bn1_1 = nn.BatchNorm1d(int(self.input_size/2))
        self.deconv1_2 = nn.ConvTranspose1d(in_channels=int(self.input_size/2), out_channels=int(self.input_size/2), kernel_size=3, stride=stride)
        self.bn1_2 = nn.BatchNorm1d(int(self.input_size/2))
        self.deconv1_3 = nn.ConvTranspose1d(in_channels=int(self.input_size/2), out_channels=int(self.output_size), kernel_size=3, stride=1, padding=1)

        self.deconv2_1 = nn.ConvTranspose1d(in_channels=int(self.input_size/2), out_channels=int(self.input_size / 4), kernel_size=3, stride=stride)
        self.bn2_1 = nn.BatchNorm1d(int(self.input_size / 4))
        self.deconv2_2 = nn.ConvTranspose1d(in_channels=int(self.input_size / 4), out_channels=int(self.input_size/4), kernel_size=3, stride=stride)
        self.bn2_2 = nn.BatchNorm1d(int(self.input_size / 4))
        self.deconv2_3 = nn.ConvTranspose1d(in_channels=int(self.input_size / 4), out_channels=int(self.output_size), kernel_size=3, stride=1, padding=1)

        self.deconv3_1 = nn.ConvTranspose1d(in_channels=int(self.input_size / 4), out_channels=int(self.input_size / 8), kernel_size=3, stride=stride)
        self.bn3_1 = nn.BatchNorm1d(int(self.input_size / 8))
        self.deconv3_2 = nn.ConvTranspose1d(in_channels=int(self.input_size / 8), out_channels=int(self.input_size / 8), kernel_size=3, stride=stride)
        self.bn3_2 = nn.BatchNorm1d(int(self.input_size / 8))
        self.deconv3_3 = nn.ConvTranspose1d(in_channels=int(self.input_size / 8), out_channels=int(self.output_size), kernel_size=3, stride=1, padding=1)



        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.dataset.normal_(0, math.sqrt(2. / n))
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self, x):
        '''

        :param
        x: batch * channel * length
        :return:
        '''
        # hop1
        x = self.deconv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        # print(x.size())
        x_hop1 = self.deconv1_3(x)
        # print(x_hop1.size())

        # hop2
        x = self.deconv2_1(x)
        x = self.bn2_1(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)
        x_hop2 = self.deconv2_3(x)
        # print(x_hop2.size())

        # hop3
        x = self.deconv3_1(x)
        x = self.bn3_1(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)
        # print(x.size())
        x_hop3 = self.deconv3_3(x)
        # print(x_hop3.size())



        return x_hop1,x_hop2,x_hop3

        # # reference code for doing residual connections
        # def _make_layer(self, block, planes, blocks, stride=1):
        #     downsample = None
        #     if stride != 1 or self.inplanes != planes * block.expansion:
        #         downsample = nn.Sequential(
        #             nn.Conv2d(self.inplanes, planes * block.expansion,
        #                       kernel_size=1, stride=stride, bias=False),
        #             nn.BatchNorm2d(planes * block.expansion),
        #         )
        #
        #     layers = []
        #     layers.append(block(self.inplanes, planes, stride, downsample))
        #     self.inplanes = planes * block.expansion
        #     for i in range(1, blocks):
        #         layers.append(block(self.inplanes, planes))
        #
        #     return nn.Sequential(*layers)





class CNN_decoder_share(nn.Module):
    def __init__(self, input_size, output_size, stride, hops):
        super(CNN_decoder_share, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hops = hops

        self.relu = nn.ReLU()
        self.deconv = nn.ConvTranspose1d(in_channels=int(self.input_size), out_channels=int(self.input_size), kernel_size=3, stride=stride)
        self.bn = nn.BatchNorm1d(int(self.input_size))
        self.deconv_out = nn.ConvTranspose1d(in_channels=int(self.input_size), out_channels=int(self.output_size), kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.dataset.normal_(0, math.sqrt(2. / n))
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self, x):
        '''

        :param
        x: batch * channel * length
        :return:
        '''

        # hop1
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x_hop1 = self.deconv_out(x)
        # print(x_hop1.size())

        # hop2
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x_hop2 = self.deconv_out(x)
        # print(x_hop2.size())

        # hop3
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x_hop3 = self.deconv_out(x)
        # print(x_hop3.size())



        return x_hop1,x_hop2,x_hop3



class CNN_decoder_attention(nn.Module):
    def __init__(self, input_size, output_size, stride=2):

        super(CNN_decoder_attention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.deconv = nn.ConvTranspose1d(in_channels=int(self.input_size), out_channels=int(self.input_size),
                                         kernel_size=3, stride=stride)
        self.bn = nn.BatchNorm1d(int(self.input_size))
        self.deconv_out = nn.ConvTranspose1d(in_channels=int(self.input_size), out_channels=int(self.output_size),
                                             kernel_size=3, stride=1, padding=1)
        self.deconv_attention = nn.ConvTranspose1d(in_channels=int(self.input_size), out_channels=int(self.input_size),
                                             kernel_size=1, stride=1, padding=0)
        self.bn_attention = nn.BatchNorm1d(int(self.input_size))
        self.relu_leaky = nn.LeakyReLU(0.2)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.dataset.normal_(0, math.sqrt(2. / n))
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        '''

        :param
        x: batch * channel * length
        :return:
        '''
        # hop1
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x_hop1 = self.deconv_out(x)
        x_hop1_attention = self.deconv_attention(x)
        # x_hop1_attention = self.bn_attention(x_hop1_attention)
        x_hop1_attention = self.relu(x_hop1_attention)
        x_hop1_attention = torch.matmul(x_hop1_attention,
                                        x_hop1_attention.view(-1,x_hop1_attention.size(2),x_hop1_attention.size(1)))
        # x_hop1_attention_sum = torch.norm(x_hop1_attention, 2, dim=1, keepdim=True)
        # x_hop1_attention = x_hop1_attention/x_hop1_attention_sum


        # print(x_hop1.size())

        # hop2
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x_hop2 = self.deconv_out(x)
        x_hop2_attention = self.deconv_attention(x)
        # x_hop2_attention = self.bn_attention(x_hop2_attention)
        x_hop2_attention = self.relu(x_hop2_attention)
        x_hop2_attention = torch.matmul(x_hop2_attention,
                                        x_hop2_attention.view(-1, x_hop2_attention.size(2), x_hop2_attention.size(1)))
        # x_hop2_attention_sum = torch.norm(x_hop2_attention, 2, dim=1, keepdim=True)
        # x_hop2_attention = x_hop2_attention/x_hop2_attention_sum


        # print(x_hop2.size())

        # hop3
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x_hop3 = self.deconv_out(x)
        x_hop3_attention = self.deconv_attention(x)
        # x_hop3_attention = self.bn_attention(x_hop3_attention)
        x_hop3_attention = self.relu(x_hop3_attention)
        x_hop3_attention = torch.matmul(x_hop3_attention,
                                        x_hop3_attention.view(-1, x_hop3_attention.size(2), x_hop3_attention.size(1)))
        # x_hop3_attention_sum = torch.norm(x_hop3_attention, 2, dim=1, keepdim=True)
        # x_hop3_attention = x_hop3_attention / x_hop3_attention_sum


        # print(x_hop3.size())



        return x_hop1, x_hop2, x_hop3, x_hop1_attention, x_hop2_attention, x_hop3_attention






#### test code ####
# x = Variable(torch.randn(1, 256, 1)).cuda()
# decoder = CNN_decoder(256, 16).cuda()
# y = decoder(x)

class Graphsage_Encoder(nn.Module):
    def __init__(self, feature_size, input_size, layer_num):
        super(Graphsage_Encoder, self).__init__()

        self.linear_projection = nn.Linear(feature_size, input_size)

        self.input_size = input_size

        # linear for hop 3
        self.linear_3_0 = nn.Linear(input_size*(2 ** 0), input_size*(2 ** 1))
        self.linear_3_1 = nn.Linear(input_size*(2 ** 1), input_size*(2 ** 2))
        self.linear_3_2 = nn.Linear(input_size*(2 ** 2), input_size*(2 ** 3))
        # linear for hop 2
        self.linear_2_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))
        self.linear_2_1 = nn.Linear(input_size * (2 ** 1), input_size * (2 ** 2))
        # linear for hop 1
        self.linear_1_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))
        # linear for hop 0
        self.linear_0_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))

        self.linear = nn.Linear(input_size*(2+2+4+8), input_size*(16))


        self.bn_3_0 = nn.BatchNorm1d(self.input_size * (2 ** 1))
        self.bn_3_1 = nn.BatchNorm1d(self.input_size * (2 ** 2))
        self.bn_3_2 = nn.BatchNorm1d(self.input_size * (2 ** 3))

        self.bn_2_0 = nn.BatchNorm1d(self.input_size * (2 ** 1))
        self.bn_2_1 = nn.BatchNorm1d(self.input_size * (2 ** 2))

        self.bn_1_0 = nn.BatchNorm1d(self.input_size * (2 ** 1))

        self.bn_0_0 = nn.BatchNorm1d(self.input_size * (2 ** 1))

        self.bn = nn.BatchNorm1d(input_size*(16))

        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data,gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, nodes_list, nodes_count_list):
        '''

        :param nodes: a list, each element n_i is a tensor for node's k-i hop neighbours
                (the first nodes_hop is the furthest neighbor)
                where n_i = N * num_neighbours * features
               nodes_count: a list, each element is a list that show how many neighbours belongs to the father node
        :return:
        '''


        # 3-hop feature
        # nodes original features to representations
        nodes_list[0] = Variable(nodes_list[0]).cuda()
        nodes_list[0] = self.linear_projection(nodes_list[0])
        nodes_features = self.linear_3_0(nodes_list[0])
        nodes_features = self.bn_3_0(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_count = nodes_count_list[0]
        # print(nodes_count,nodes_count.size())
        # aggregated representations placeholder, feature dim * 2
        nodes_features_farther = Variable(torch.Tensor(nodes_features.size(0), nodes_count.size(1), nodes_features.size(2))).cuda()
        i = 0
        for j in range(nodes_count.size(1)):
            # mean pooling for each father node
            # print(nodes_count[:,j][0],type(nodes_count[:,j][0]))
            nodes_features_farther[:,j,:] = torch.mean(nodes_features[:, i:i+int(nodes_count[:,j][0]), :], 1, keepdim = False)
            i += int(nodes_count[:,j][0])
        # assign node_features
        nodes_features = nodes_features_farther
        nodes_features = self.linear_3_1(nodes_features)
        nodes_features = self.bn_3_1(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_count = nodes_count_list[1]
        # aggregated representations placeholder, feature dim * 2
        nodes_features_farther = Variable(torch.Tensor(nodes_features.size(0), nodes_count.size(1), nodes_features.size(2))).cuda()
        i = 0
        for j in range(nodes_count.size(1)):
            # mean pooling for each father node
            nodes_features_farther[:,j,:] = torch.mean(nodes_features[:, i:i+int(nodes_count[:,j][0]), :], 1, keepdim = False)
            i += int(nodes_count[:,j][0])
        # assign node_features
        nodes_features = nodes_features_farther
        # print('nodes_feature',nodes_features.size())
        nodes_features = self.linear_3_2(nodes_features)
        nodes_features = self.bn_3_2(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        # nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_features_hop_3 = torch.mean(nodes_features, 1, keepdim=True)
        # print(nodes_features_hop_3.size())

        # 2-hop feature
        # nodes original features to representations
        nodes_list[1] = Variable(nodes_list[1]).cuda()
        nodes_list[1] = self.linear_projection(nodes_list[1])
        nodes_features = self.linear_2_0(nodes_list[1])
        nodes_features = self.bn_2_0(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_count = nodes_count_list[1]
        # aggregated representations placeholder, feature dim * 2
        nodes_features_farther = Variable(torch.Tensor(nodes_features.size(0), nodes_count.size(1), nodes_features.size(2))).cuda()
        i = 0
        for j in range(nodes_count.size(1)):
            # mean pooling for each father node
            nodes_features_farther[:,j,:] = torch.mean(nodes_features[:, i:i+int(nodes_count[:,j][0]), :], 1, keepdim = False)
            i += int(nodes_count[:,j][0])
        # assign node_features
        nodes_features = nodes_features_farther
        nodes_features = self.linear_2_1(nodes_features)
        nodes_features = self.bn_2_1(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        # nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_features_hop_2 = torch.mean(nodes_features, 1, keepdim=True)
        # print(nodes_features_hop_2.size())


        # 1-hop feature
        # nodes original features to representations
        nodes_list[2] = Variable(nodes_list[2]).cuda()
        nodes_list[2] = self.linear_projection(nodes_list[2])
        nodes_features = self.linear_1_0(nodes_list[2])
        nodes_features = self.bn_1_0(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        # nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_features_hop_1 = torch.mean(nodes_features, 1, keepdim=True)
        # print(nodes_features_hop_1.size())


        # own feature
        nodes_list[3] = Variable(nodes_list[3]).cuda()
        nodes_list[3] = self.linear_projection(nodes_list[3])
        nodes_features = self.linear_0_0(nodes_list[3])
        nodes_features = self.bn_0_0(nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1)))
        nodes_features_hop_0 = nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        # print(nodes_features_hop_0.size())



        # concatenate
        nodes_features = torch.cat((nodes_features_hop_0, nodes_features_hop_1, nodes_features_hop_2, nodes_features_hop_3),dim=2)
        nodes_features = self.linear(nodes_features)
        # nodes_features = self.bn(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        # print(nodes_features.size())
        return(nodes_features)




