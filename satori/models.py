import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# import Parameter to create custom activations with learnable parameters
from torch.nn.parameter import Parameter
import numpy as np
from einops.layers.torch import Rearrange
from torch import nn, einsum
import torch.nn.init as init
from satori.sei_model import Sei_light, BSplineTransformation, Sei_model2


class CNNAttentionBlock(nn.Module):
    def __init__(self, input_dim, cnn_channels, kernel_size, num_heads, head_dim):
        super(CNNAttentionBlock, self).__init__()

        # 1D CNN layer
        self.cnn_layer = nn.Conv1d(
            in_channels=input_dim, out_channels=cnn_channels, kernel_size=kernel_size, padding="same")
        self.relu = nn.ReLU()

        # Multihead Attention layer
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=cnn_channels, num_heads=num_heads)

    def forward(self, x):
        # Input x: (batch_size, sequence_length, input_dim)

        # Apply CNN layer
        # Permute for CNN input format
        cnn_output = self.relu(self.cnn_layer(x.permute(0, 2, 1)))
        cnn_output = cnn_output.permute(0, 2, 1)

        # Apply Multihead Attention
        attention_output, _ = self.multihead_attention(
            cnn_output, cnn_output, cnn_output)
        # print(attention_output.shape, cnn_output.shape)

        # Add skip connection
        output = cnn_output + attention_output  # Permute back for addition

        return output


class CNNAttentionModel(nn.Module):
    def __init__(self, num_blocks, input_dim, cnn_channels, kernel_size, num_heads, head_dim):
        super(CNNAttentionModel, self).__init__()
        self.inp_dim = [input_dim, cnn_channels, cnn_channels]
        self.blocks = nn.ModuleList([
            CNNAttentionBlock(
                self.inp_dim[i], cnn_channels, kernel_size, num_heads, head_dim)
            for i in range(num_blocks)
        ])

    def forward(self, x):
        # Input x: (batch_size, sequence_length, input_dim)

        # Apply each block sequentially
        for block in self.blocks:
            x = block(x)

        return x


class PositionalEncoding(nn.Module):
    # Taken from: https://nlp.seas.harvard.edu/2018/04/03/attention.html
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Exponential(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position

    def forward(self, length_q=200, length_k=200):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = torch.LongTensor(distance_mat_clipped).cuda()
        distances = final_mat.repeat(self.num_units, 1, 1)
        distances = torch.abs(distances)

        return distances


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool_fn = Rearrange('b (n p) d-> b n p d', n=1)
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x):
        attn_logits = einsum('b n d, d e -> b n e', x, self.to_attn_logits)
        x = self.pool_fn(x)
        logits = self.pool_fn(attn_logits)

        attn = logits.softmax(dim=-2)
        return (x * attn).sum(dim=-2).squeeze()


# for the model that uses CNN, RNN (optionally), and MH attention
class AttentionNet(nn.Module):
    def __init__(self, numLabels, params, device=None, seq_len=300, genPAttn=True, reuseWeightsQK=False):
        super(AttentionNet, self).__init__()
        self.numofAttnLayers = params['num_attnlayers']
        self.numMultiHeads = params['num_multiheads']
        self.SingleHeadSize = params['singlehead_size']  # SingleHeadSize
        self.MultiHeadSize = params['multihead_size']  # MultiHeadSize
        self.usepooling = params['use_pooling']
        self.pooling_val = params['pooling_val']
        self.readout_strategy = params['readout_strategy']
        self.kmerSize = params['embd_kmersize']
        self.useRNN = params['use_RNN']
        self.useCNN = params['use_CNN']
        self.CNN1useExponential = params['CNN1_useexponential']
        self.usePE = params['use_posEnc']
        self.useCNNpool = params['use_CNNpool']
        self.RNN_hiddenSize = params['RNN_hiddensize']
        self.numCNNfilters = params['CNN_filters']
        self.filterSize = params['CNN_filtersize']
        self.CNNpoolSize = params['CNN_poolsize']
        self.CNNpadding = params['CNN_padding']
        self.numClasses = numLabels.numLabels  # change when not using model selection
        self.device = device
        self.reuseWeightsQK = reuseWeightsQK
        # number of channels, one hot encoding
        self.numInputChannels = params['input_channels']
        self.genPAttn = genPAttn
        self.relativeAttn = params['relativeAttn']
        self.learnable_relativeAttn = params['Learnable_relativeAttn']
        self.mixed_attn = params['mixed_attn']
        self.mixed_attn_config = params['mixed_attn_config']
        self.batch_size = params['batch_size']
        self.sqrt_dist = params['sqrt_dist']
        self.seqLength = seq_len
        self.attn_skip = params['residual_connection']
        self.cnn_residual = False
        self.last_channels = self.numInputChannels
        self.multiple_linear = params['multiple_linear']
        self.linear_size = params['linear_layer_size']
        self.cnnlayer = []
        print(params)
        self.cnn_attention_block = params["cnn_attention_block"]
        self.use_sei = params["sei"]
        self.exp_name = params["exp_name"]


        if self.cnn_attention_block:
            num_blocks = 3
            input_dim = 128
            cnn_channels = 128
            kernel_size = 3
            num_heads = 4
            head_dim = cnn_channels // num_heads

            self.cnn_attention_sub_model = CNNAttentionModel(
                num_blocks, input_dim, cnn_channels, kernel_size, num_heads, head_dim)

        if self.useCNN and self.useCNNpool:

            self.layer1 = nn.Sequential(nn.Conv1d(in_channels=self.numInputChannels, out_channels=self.numCNNfilters[0],
                                                  kernel_size=self.filterSize[0], padding=self.CNNpadding, bias=False), nn.BatchNorm1d(num_features=self.numCNNfilters[0]),
                                        nn.ReLU() if self.CNN1useExponential == False else Exponential(),
                                        nn.MaxPool1d(kernel_size=self.CNNpoolSize))
            self.dropout1 = nn.Dropout(p=0.2)
            self.last_channels = self.numCNNfilters[0]

        if self.useCNN and self.useCNNpool == False:

            self.layer1 = nn.Sequential(nn.Conv1d(in_channels=self.numInputChannels, out_channels=self.numCNNfilters[0],
                                                  kernel_size=self.filterSize[0], padding=self.CNNpadding, bias=False),
                                        nn.BatchNorm1d(
                num_features=self.numCNNfilters[0]),
                nn.ReLU() if self.CNN1useExponential == False else Exponential())
            self.dropout1 = nn.Dropout(p=0.2)
            self.maxpool1 = nn.MaxPool1d(kernel_size=self.CNNpoolSize)

            self.last_channels = self.numCNNfilters[0]

        if len(self.numCNNfilters) > 1:

            for i, num_filters in enumerate(self.numCNNfilters):

                # Add next layers of cnn
                if i > 0:

                    self.cnnlayer.append(nn.Sequential(nn.Conv1d(in_channels=self.last_channels, out_channels=self.numCNNfilters[i],
                                                                 kernel_size=self.filterSize[i], padding="same", bias=False),
                                                       nn.ReLU()))

                    self.last_channels = self.numCNNfilters[i]

            # change shape of Q, K ,V

        def outputsize(n_in, p, k, s): return int(((n_in + (2*p) - k) / s) + 1)
        
        n_out = outputsize(seq_len, self.CNNpadding, self.filterSize[0], 1)
        print("Here", n_out)
        if self.useCNNpool:
            # by default stride is same as kernalsize
            n_out = outputsize(n_out, 0, self.CNNpoolSize, self.CNNpoolSize)
            print("Here cnn pool", n_out, self.CNNpoolSize)

        for i in range(len(self.cnnlayer)):
            same_pad = int(self.filterSize[i+1]/2)
            n_out = outputsize(n_out, same_pad, self.filterSize[i+1], 1)
            print("Here cnn", n_out)

        print(seq_len, n_out)
        
        if self.relativeAttn:
            desired_output_size = (n_out, n_out)
            input_size = (seq_len, seq_len)
            # Calculate the stride
            # Calculate the maximum stride that keeps the output size equal or larger than the desired output size
            max_stride = (
                input_size[0] - desired_output_size[0]) // (desired_output_size[0] - 1)

            # Calculate the stride to ensure the output size is equal or larger than the desired output size
            stride = max(max_stride, 1)

            # Calculate the kernel size to achieve the desired output size
            kernel_size = ((input_size[0] - 1) - (desired_output_size[0] - 1) * stride + 1,
                           (input_size[1] - 1) - (desired_output_size[1] - 1) * stride + 1)

            # Calculate the required padding to achieve the desired output size
            padding = ((desired_output_size[0] - 1) * stride + kernel_size[0] - input_size[0],
                       (desired_output_size[1] - 1) * stride + kernel_size[1] - input_size[1])

            print("Stride:", stride)
            print("Kernel Size:", kernel_size)
            print("Padding:", padding)

            self.rd1 = nn.MaxPool2d(
                kernel_size, stride=stride, padding=padding)
            self.rd2 = nn.MaxPool2d(kernel_size=self.CNNpoolSize)
            self.relative_position_k_q = RelativePosition(
                self.batch_size, self.seqLength - 1)
            self.relative_dist = self.relative_position_k_q(
                self.seqLength, self.seqLength) / (self.seqLength - 1)

        if self.useRNN:
            self.RNN = nn.LSTM(self.numInputChannels if self.useCNN == False else self.last_channels,
                               self.RNN_hiddenSize, num_layers=2, bidirectional=True)
            self.dropoutRNN = nn.Dropout(p=0.4)
            self.Q = nn.ModuleList()
            self.Q.append(nn.ModuleList([nn.Linear(in_features=2*self.RNN_hiddenSize,
                                   out_features=self.SingleHeadSize) for i in range(0, self.numMultiHeads)]))
            self.K = nn.ModuleList()
            self.K.append(nn.ModuleList([nn.Linear(in_features=2*self.RNN_hiddenSize,
                                   out_features=self.SingleHeadSize) for i in range(0, self.numMultiHeads)]))
            self.V = nn.ModuleList()
            self.V.append(nn.ModuleList([nn.Linear(in_features=2*self.RNN_hiddenSize,
                                   out_features=self.SingleHeadSize) for i in range(0, self.numMultiHeads)]))

        if self.useRNN == False and self.useCNN == False:
            self.Q = nn.ModuleList([nn.Linear(in_features=self.numInputChannels,
                                   out_features=self.SingleHeadSize) for i in range(0, self.numMultiHeads)])
            self.K = nn.ModuleList([nn.Linear(in_features=self.numInputChannels,
                                   out_features=self.SingleHeadSize) for i in range(0, self.numMultiHeads)])
            self.V = nn.ModuleList([nn.Linear(in_features=self.numInputChannels,
                                   out_features=self.SingleHeadSize) for i in range(0, self.numMultiHeads)])

        if self.useRNN == False and self.useCNN == True:
            self.Q = nn.ModuleList()
            self.Q.append(nn.ModuleList([nn.Linear(
                in_features=self.last_channels, out_features=self.SingleHeadSize) for i in range(0, self.numMultiHeads)]))
            for j in range(1, self.numofAttnLayers):
                self.Q.append(nn.ModuleList([nn.Linear(
                    in_features=self.MultiHeadSize, out_features=self.SingleHeadSize) for i in range(0, self.numMultiHeads)]))
            self.K = nn.ModuleList()
            self.K.append(nn.ModuleList([nn.Linear(
                in_features=self.last_channels, out_features=self.SingleHeadSize) for i in range(0, self.numMultiHeads)]))
            for j in range(1, self.numofAttnLayers):
                self.K.append(nn.ModuleList([nn.Linear(
                    in_features=self.MultiHeadSize, out_features=self.SingleHeadSize) for i in range(0, self.numMultiHeads)]))
            self.V = nn.ModuleList()
            self.V.append(nn.ModuleList([nn.Linear(
                in_features=self.last_channels, out_features=self.SingleHeadSize) for i in range(0, self.numMultiHeads)]))
            for j in range(1, self.numofAttnLayers):
                self.V.append(nn.ModuleList([nn.Linear(
                    in_features=self.MultiHeadSize, out_features=self.SingleHeadSize) for i in range(0, self.numMultiHeads)]))

        # reuse weights between Query (Q) and Key (K)
        if self.reuseWeightsQK:
            for i in range(0, self.numMultiHeads):
                self.K[i].weight = Parameter(self.Q[i].weight.t())

        self.RELU = nn.ModuleList([nn.ReLU()
                                  for i in range(0, self.numMultiHeads)])
        self.MultiHeadLinearDropout = nn.ModuleList(
            [nn.Dropout(0.2) for j in range(0, self.numofAttnLayers)])
        if self.multiple_linear: #Asa said it does not make difference 

            self.MultiHeadLinear_1 = nn.ModuleList([nn.Linear(in_features=self.SingleHeadSize*self.numMultiHeads, out_features=self.linear_size)
                                                   for j in range(0, self.numofAttnLayers)])  # self.numMultiHeads*self.SingleHeadSize
            self.MultiHeadLinear_2 = nn.ModuleList([nn.Linear(
                in_features=self.linear_size, out_features=self.MultiHeadSize) for j in range(0, self.numofAttnLayers)])
            self.fc3 = nn.Linear(
                in_features=self.MultiHeadSize, out_features=self.numClasses)
            self.MultiHeadLinearDropout2 = nn.ModuleList(
                [nn.Dropout(0.2) for j in range(0, self.numofAttnLayers)])
            self.MultiHeadLayerNorm = nn.ModuleList([nn.LayerNorm(self.SingleHeadSize * self.numMultiHeads) for i in range(0, self.numofAttnLayers)])
            self.MHGeLU = nn.ModuleList([nn.ReLU()
                                    for i in range(0, self.numofAttnLayers)])


        else:
            self.MultiHeadLinear = nn.ModuleList([nn.Linear(
                in_features=self.SingleHeadSize*self.numMultiHeads, out_features=self.MultiHeadSize) for j in range(0, self.numofAttnLayers)])
            
            if self.use_sei:
                if self.exp_name == "simulated":
                    self.sei_light = Sei_light(n_out, big=False, bspline = False) # trying to see if without reshaping self.MultiHeadSize)
                    self.fc3 = nn.Linear(
                    in_features=self.MultiHeadSize, out_features=self.numClasses) #self.sei_light.output_dim * 16,
            
                else:
                    self.sei_light = Sei_light(n_out, big=False) # trying to see if without reshaping self.MultiHeadSize)

                    
                    self.fc3 = nn.Linear(
                    in_features=self.MultiHeadSize, out_features=self.numClasses) #sei_light.output_dim * 16,self.MultiHeadSize
                
            else:
                self.fc3 = nn.Linear(in_features=self.MultiHeadSize, out_features=self.numClasses)
            # 480 * 16 for Sei output, otherwise MultiheadSize


        if self.readout_strategy == "attention_pool":
            self.attention_pool = AttentionPool(self.MultiHeadSize)

        if self.readout_strategy == "bspline":
            self._spline_df = int(128/8)
            self.spline_tr = nn.Sequential(
                BSplineTransformation(self._spline_df, scaled=False))

        maxpool_out = 16
        if self.usepooling:
            self.MAXPOOL = nn.ModuleList([nn.MaxPool2d(
                kernel_size=self.pooling_val) for i in range(0, self.numMultiHeads)])  # 50
            self.MultiHeadLinear = nn.Linear(
                in_features=maxpool_out*self.numMultiHeads, out_features=self.MultiHeadSize)  # 50

        self.MHReLU = nn.ModuleList([nn.ReLU()
                                    for i in range(0, self.numofAttnLayers)])


        if len(self.cnnlayer) > 0:
            self.cnn_residual = True
            self.CNNlayers = nn.ModuleList(self.cnnlayer)


        # self.initialize_weights() # did not work
    def initialize_weights(self):
        print("Initilizing weights with HE...")
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def attention(self, query, key, value, mask=None, dropout=0.0):
        # based on: https://nlp.seas.harvard.edu/2018/04/03/attention.html
        d_k = query.size(-1)
        # print(query.shape, d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn

    def relative_attention(self, query, key, value, RPE, sqrt, mask=None, dropout=0.0):
        # relative attention
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(d_k)  # , diff)
        RPE = RPE[:scores.shape[0], :, :]
        if RPE.shape[0] < scores.shape[0]:
            RPE = RPE[0, :, :].repeat(scores.shape[0], 1, 1)
        if sqrt:
            RPE = torch.sqrt(RPE)
        scores = torch.div(scores, RPE)  # RPE
        p_attn = F.softmax(scores, dim=-1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, inputs):
        output = inputs

        if self.useCNN:
            cnn1_out = self.layer1(output)
            output = self.dropout1(cnn1_out)

        if len(self.cnnlayer) > 0:
            for i in range(len(self.cnnlayer)):
                output = self.CNNlayers[i][0](output)

        if self.cnn_residual:
            output += cnn1_out
            output = self.CNNlayers[i][1](output)
            output = self.maxpool1(output)

        output = output.permute(0, 2, 1)
        
        #print(output.shape)
        
        
        if self.useRNN:
            output, _ = self.RNN(output)
            F_RNN = output[:, :, :self.RNN_hiddenSize]
            R_RNN = output[:, :, self.RNN_hiddenSize:]
            output = torch.cat((F_RNN, R_RNN), 2)
            output = self.dropoutRNN(output)

        pAttn_concat = torch.Tensor([]).to(self.device)
        no_relativeAttn = 0
        for j in range(0, self.numofAttnLayers):

            attn_concat = torch.Tensor([]).to(self.device)
            #pAttn_concat_h = torch.Tensor([]).to(self.device)

            for i in range(0, self.numMultiHeads):

                query, key, value = self.Q[j][i](
                    output), self.K[j][i](output), self.V[j][i](output)
                #temp_value = value
                if self.relativeAttn:
                    if self.mixed_attn:
                        if self.numMultiHeads == 2:
                            no_relativeAttn = 1
                        else:
                            no_relativeAttn = self.mixed_attn_config * self.numMultiHeads
                    else:
                        no_relativeAttn = self.numMultiHeads

                if i < no_relativeAttn and j == 0:
                    if self.conv_residual:
                        RPE1_output = self.rd2(self.rd1(self.relative_dist))
                    else:
                        RPE1_output = self.rd1(self.relative_dist)
                    attnOut, p_attn = self.relative_attention(
                        query, key, value, RPE1_output, self.sqrt_dist, dropout=0.2)
                else:
                    attnOut, p_attn = self.attention(
                        query, key, value, dropout=0.2)

                attnOut = self.RELU[i](attnOut)
                if self.usepooling:
                    attnOut = self.MAXPOOL[i](
                        attnOut.permute(0, 2, 1)).permute(0, 2, 1)
                # if self.attn_skip: #????? use concatenation
                #     attnOut = attnOut + value

                attn_concat = torch.cat((attn_concat, attnOut), dim=2)

                if self.genPAttn and j == 0:
                    pAttn_concat = torch.cat((pAttn_concat, p_attn), dim=2)
                    #pAttn_concat_h = torch.cat((pAttn_concat_h, p_attn.unsqueeze(dim = 1)), dim=1)

            if self.attn_skip:
                    attn_concat = attn_concat + output
            if self.multiple_linear:
                # output = self.MultiHeadLinearDropout[j](attn_concat)
                output = self.MultiHeadLinear_2[j](self.MultiHeadLinearDropout[j](self.MHGeLU[j](self.MultiHeadLinear_1[j]
                                                                                                 (self.MultiHeadLayerNorm[j](attn_concat)))))
                output = self.MHReLU[j](output)
                output = self.MultiHeadLinearDropout2[j](output)
            else:
                #print("attn ", attn_concat.shape)
                output = self.MultiHeadLinearDropout[j](attn_concat)
                output = self.MultiHeadLinear[j](attn_concat)
                output = self.MHReLU[j](output)


            if self.cnn_attention_block:
                output = self.cnn_attention_sub_model(output)

        #output = output.permute(0, 2, 1)
        if self.use_sei:
            output = self.sei_light(output)
        #print("sei output ", output.shape)

        if self.readout_strategy == 'normalize':
            output = output.sum(axis=1)
            output = (output-output.mean())/output.std()
        elif self.readout_strategy == 'bspline':
            spline_out = self.spline_tr(output)
            #print(spline_out.shape)
            output = spline_out.view(
                spline_out.size(0), 480 * self._spline_df)
        elif self.readout_strategy == 'attention_pool':
            output = self.attention_pool(output)
        else:
            pass
            #

        output = self.fc3(output)
        #output = torch.nan_to_num(output)
        assert not torch.isnan(output).any()
        if self.genPAttn:
            return output, pAttn_concat  # , RPE1_output
        else:
            return output
