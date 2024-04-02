import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import numpy as np
from einops.layers.torch import Rearrange
from torch import nn, einsum

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
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = torch.LongTensor(distance_mat_clipped).cuda()
        distances = final_mat.repeat(self.num_units, 1, 1)
        distances = torch.abs(distances)


        return distances


# class CNNAttentionBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding =0,CNNpoolSize=4. stride=1, num_heads =1,SingleHeadSize =64 ):
#         super(CNNAttentionBlock, self).__init__()
#         self.layer1  = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
#                                          kernel_size=kernel_size, padding=CNNpadding, bias=False),nn.BatchNorm1d(num_features=self.numCNNfilters[0]),
#                                          nn.ReLU(),
#                                          nn.MaxPool1d(kernel_size=CNNpoolSize))
#         self.dropout1 = nn.Dropout(p=0.2)
        
#         self.Q = nn.ModuleList([nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
#         self.K = nn.ModuleList([nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
#         self.V = nn.ModuleList([nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
#         #self.self_attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=1)

#     def attention(self, query, key, value, mask=None, dropout=0.0):
#             #based on: https://nlp.seas.harvard.edu/2018/04/03/attention.html
#         d_k = query.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#         p_attn = F.softmax(scores, dim = -1)
#         p_attn = F.dropout(p_attn, p=dropout, training=self.training)
#         return torch.matmul(p_attn, value), p_attn
    
#     def forward(self, x):
#         out = self.conv1d(x)
#         out = F.relu(out)
#         out = out.permute(2, 0, 1)  # Reshape for self-attention
#         out, _ = self.self_attention(out, out, out)
#         out = out.permute(1, 2, 0)  # Reshape back
#         return out

# class CNNAttentionClassifier(nn.Module):
#     def __init__(self, num_blocks, input_dim, num_classes):
#         super(CNNAttentionClassifier, self).__init()
#         self.num_blocks = num_blocks
#         self.blocks = self._create_blocks(num_blocks, input_dim)
#         self.fc = nn.Linear(input_dim, num_classes)

#     def _create_blocks(self, num_blocks, input_dim):
#         blocks = []
#         in_channels = input_dim
#         for _ in range(num_blocks):
#             block = CNNAttentionBlock(in_channels, 64)  # You can adjust the number of output channels
#             blocks.append(block)
#             in_channels = 64  # Update the number of input channels for the next block
#         return nn.Sequential(*blocks)

#     def forward(self, x):
#         out = self.blocks(x)
#         out = F.avg_pool1d(out, out.size(-1)).squeeze(-1)
#         out = self.fc(out)
#         return out

# # Create an instance of the model
# num_blocks = 3  # You can choose the number of blocks
# input_dim = 4  # Replace with the appropriate input dimension (e.g., one-hot encoding of DNA sequences)
# num_classes = 2  # Number of output classes

# model = CNNAttentionClassifier(num_blocks, input_dim, num_classes)

# # Example usage:
# input_data = torch.randn(32, input_dim, 100)  # Replace with your input shape
# output = model(input_data)
# print(output.shape)

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool_fn = Rearrange('b (n p) d-> b n p d', n=1)
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x):
        attn_logits = einsum('b n d, d e -> b n e', x, self.to_attn_logits)
        x = self.pool_fn(x)
        logits = self.pool_fn(attn_logits)

        attn = logits.softmax(dim = -2)
        return (x * attn).sum(dim = -2).squeeze()

class AttentionNet(nn.Module): #for the model that uses CNN, RNN (optionally), and MH attention
    def __init__(self, numLabels, params, device=None, seq_len = 300, genPAttn=True, reuseWeightsQK=False):
        super(AttentionNet, self).__init__()
        self.numofAttnLayers = params['num_attnlayers']
        self.numMultiHeads = params['num_multiheads']
        self.SingleHeadSize = params['singlehead_size']#SingleHeadSize
        self.MultiHeadSize = params['multihead_size']#MultiHeadSize
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
        self.numClasses = numLabels.numLabels #change when not using model selection
        self.device = device
        self.reuseWeightsQK = reuseWeightsQK
        self.numInputChannels = params['input_channels'] #number of channels, one hot encoding
        self.genPAttn = genPAttn
        self.relativeAttn = params['relativeAttn']
        self.learnable_relativeAttn = params['Learnable_relativeAttn']
        self.mixed_attn = params['mixed_attn']
        self.mixed_attn_config = params['mixed_attn_config']
        self.batch_size = params['batch_size']
        self.sqrt_dist = params['sqrt_dist']
        self.seqLength = seq_len
        self.residual = params['residual_connection']

        self.last_channels = self.numInputChannels
        self.multiple_linear = params['multiple_linear']
        self.linear_size = params['linear_layer_size']
        self.cnnlayer = []
        print(params)


        if self.usePE:
            self.pe = PositionalEncoding(d_model = self.numInputChannels, dropout=0.1)

        if self.useCNN and self.useCNNpool:

            #print(self.numCNNfilters)
            
            self.layer1  = nn.Sequential(nn.Conv1d(in_channels=self.numInputChannels, out_channels=self.numCNNfilters[0],
                                         kernel_size=self.filterSize[0], padding=self.CNNpadding, bias=False),nn.BatchNorm1d(num_features=self.numCNNfilters[0]),
                                         nn.ReLU() if self.CNN1useExponential==False else Exponential(),
                                         nn.MaxPool1d(kernel_size=self.CNNpoolSize))
            self.dropout1 = nn.Dropout(p=0.2)

            self.last_channels = self.numCNNfilters[0]
        
        if self.useCNN and self.useCNNpool == False:
                
            self.layer1  = nn.Sequential(nn.Conv1d(in_channels=self.numInputChannels, out_channels=self.numCNNfilters[0],
                                         kernel_size=self.filterSize[0], padding=self.CNNpadding, bias=False),
                                         nn.BatchNorm1d(num_features=self.numCNNfilters[0]),
                                         nn.ReLU() if self.CNN1useExponential==False else Exponential())
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

                    # self.cnnlayer.append(nn.Sequential(nn.Conv1d(in_channels=self.last_channels, out_channels=self.numCNNfilters[i],
                    #                       kernel_size=self.filterSize[i], padding=self.CNNpadding, bias=False),
                    #                       nn.BatchNorm1d(num_features=self.numCNNfilters[i]),
                    #                       nn.ReLU()))                    
                    self.last_channels = self.numCNNfilters[i]
    
            # change shape of Q, K ,V

        
        outputsize = lambda n_in, p, k, s : int(((n_in + (2*p) - k) / s) + 1)
        n_out = outputsize(seq_len, self.CNNpadding, self.filterSize[0], 1)
        if self.useCNNpool:
                n_out = outputsize(n_out, 0, self.CNNpoolSize, self.CNNpoolSize) #by default stride is same as kernalsize
                
        for i in range(len(self.cnnlayer)):
                same_pad = int(self.filterSize[i+1]/2)
                n_out = outputsize(n_out, same_pad, self.filterSize[i+1], 1)
            
        print(seq_len, n_out)
        if self.relativeAttn:        
            #maxpool_stride = int(np.around(200 / float(n_out)))
            desired_output_size = (n_out, n_out)
            input_size = (seq_len, seq_len)
            #print(maxpool_stride, n_out)
            
            # Calculate the stride
            # Calculate the maximum stride that keeps the output size equal or larger than the desired output size
            max_stride = (input_size[0] - desired_output_size[0]) // (desired_output_size[0] - 1)

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
            #mx_stride = ((seq_len - n_out) // (n_out - 1), (seq_len - n_out) // (n_out - 1))

            # Calculate the kernel size
            #mx_kernel_size = (seq_len - mx_stride[0] * (n_out - 1), seq_len - mx_stride[1] * (n_out - 1))
            #moved to forward
            #self.rd = nn.MaxPool2d((9, 9), stride=(6, 6), padding=1) #33 x 33
            #self.rd = nn.MaxPool2d((maxpool_stride+1, maxpool_stride+1), stride=(maxpool_stride, maxpool_stride), padding=0) #15 x 15
            #print(mx_kernel_size, mx_stride)
            self.rd1 = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
            self.rd2 = nn.MaxPool2d(kernel_size=self.CNNpoolSize)
            self.relative_position_k_q = RelativePosition(self.batch_size, self.seqLength - 1)
            self.relative_dist = self.relative_position_k_q(self.seqLength, self.seqLength)/ (self.seqLength - 1)
            #print(self.relative_dist)

            # weight distance matrix
            if self.learnable_relativeAttn:
                self.d_act_1 = nn.GELU()
                self.d_act_2 = nn.GELU()

                self.WRD_1 = nn.ModuleList([nn.Linear(in_features=int(np.around(n_out)**2), out_features=(int(np.around(n_out)) ** 2)//2) for i in range(0,self.numMultiHeads)])
                self.WRD_2 = nn.ModuleList([nn.Linear(in_features=(int(np.around(n_out)) ** 2)//2 , out_features=int(np.around(n_out)) ** 2) for i in range(0,self.numMultiHeads)])
                
                # self.WRDj_1 = nn.ModuleList([nn.Linear(in_features=int(np.around(n_out)), out_features=int(np.around(n_out)) * 2) for i in range(0,self.numMultiHeads)])
                # self.WRDj_2 = nn.ModuleList([nn.Linear(in_features=int(np.around(n_out)) * 2 , out_features=int(np.around(n_out))) for i in range(0,self.numMultiHeads)])
                
                self.dconv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)  # Input: 1 channel (1 matrix), Output: 64 channels
                self.drelu = nn.ReLU()
                self.dconv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)  # Output: 1 channel (1 matrix)
                #self.dsig = torch.sigmoid()





        if self.useRNN:
            self.RNN = nn.LSTM(self.numInputChannels if self.useCNN==False else self.last_channels, self.RNN_hiddenSize, num_layers=2, bidirectional=True)
            self.dropoutRNN = nn.Dropout(p=0.4)
            self.Q = nn.ModuleList([nn.Linear(in_features=2*self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.K = nn.ModuleList([nn.Linear(in_features=2*self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.V = nn.ModuleList([nn.Linear(in_features=2*self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])

        if self.useRNN == False and self.useCNN == False:
            self.Q = nn.ModuleList([nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.K = nn.ModuleList([nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.V = nn.ModuleList([nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])

        if self.useRNN == False and self.useCNN == True:
            self.Q = nn.ModuleList()
            self.Q.append(nn.ModuleList([nn.Linear(in_features=self.last_channels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)]))
            for j in range(1,self.numofAttnLayers):             
                self.Q.append(nn.ModuleList([nn.Linear(in_features=self.SingleHeadSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)]))
            self.K = nn.ModuleList()
            self.K.append(nn.ModuleList([nn.Linear(in_features=self.last_channels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)]))
            for j in range(1,self.numofAttnLayers):             
                self.K.append(nn.ModuleList([nn.Linear(in_features=self.SingleHeadSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)]))
            self.V = nn.ModuleList()
            self.V.append(nn.ModuleList([nn.Linear(in_features=self.last_channels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)]))
            for j in range(1,self.numofAttnLayers):             
                self.V.append(nn.ModuleList([nn.Linear(in_features=self.SingleHeadSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)]))

        #reuse weights between Query (Q) and Key (K)
        if self.reuseWeightsQK:
            for i in range(0, self.numMultiHeads):
                self.K[i].weight = Parameter(self.Q[i].weight.t())	

        # for i in range(0, self.numMultiHeads):
        #         self.K[i].weight = Parameter(self.Q[i].weight.t())	

        self.RELU = nn.ModuleList([nn.ReLU() for i in range(0,self.numMultiHeads)])
        if self.multiple_linear:
            
                self.MultiHeadLinear_1 = nn.ModuleList([nn.Linear(in_features=self.SingleHeadSize*self.numMultiHeads, out_features=self.linear_size) for j in range(0,self.numofAttnLayers) ])#self.numMultiHeads*self.SingleHeadSize
                self.MultiHeadLinear_2 = nn.ModuleList([nn.Linear(in_features=self.linear_size, out_features=self.SingleHeadSize * self.numMultiHeads) for j in range(0,self.numofAttnLayers) ])#50
                #self.linear_cnn = nn.Linear(in_features=self.last_channels, out_features=self.SingleHeadSize*self.numMultiHeads)

                self.fc3 = nn.Linear(in_features=self.SingleHeadSize * self.numMultiHeads, out_features=self.numClasses)
        else:
                self.MultiHeadLinear = nn.Linear(in_features=self.SingleHeadSize*self.numMultiHeads, out_features=self.MultiHeadSize)#50
                #self.linear_cnn = nn.Linear(in_features=self.last_channels, out_features=self.SingleHeadSize*self.numMultiHeads)
                self.fc3 = nn.Linear(in_features=self.MultiHeadSize, out_features=self.numClasses)
        
        self.attention_pool = AttentionPool(self.SingleHeadSize * self.numMultiHeads)


        
        self.MultiHeadLinearDropout = nn.Dropout(0.1)
        self.MultiHeadLinearDropout2 = nn.Dropout(0.1)

        #self.MultiHeadLayerNorm = nn.LayerNorm(self.SingleHeadSize)



        self.MultiHeadDropout = nn.ModuleList([nn.Dropout(0.1) for i in range(0,self.numofAttnLayers)])
        self.MultiHeadLayerNorm = nn.ModuleList([nn.LayerNorm(self.SingleHeadSize * self.numMultiHeads) for i in range(0,self.numofAttnLayers)])


        maxpool_out = 16
        if self.usepooling:
            self.MAXPOOL = nn.ModuleList([nn.MaxPool2d(kernel_size=self.pooling_val) for i in range(0,self.numMultiHeads)])#50
            self.MultiHeadLinear = nn.Linear(in_features=maxpool_out*self.numMultiHeads, out_features=self.MultiHeadSize)#50

            


        self.MHReLU = nn.ReLU()
        self.MHGeLU = nn.ModuleList([nn.GELU() for i in range(0,self.numofAttnLayers)])

        if len(self.cnnlayer) > 0:

            self.CNNlayers = nn.ModuleList(self.cnnlayer)
        



    def attention(self, query, key, value, mask=None, dropout=0.0):
        #based on: https://nlp.seas.harvard.edu/2018/04/03/attention.html
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim = -1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn
    
    def relative_attention(self, query, key, value, RPE, sqrt, mask=None, dropout=0.0): 
        #relative attention  
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #, diff)
        RPE = RPE[:scores.shape[0], :, :]
        if RPE.shape[0] < scores.shape[0]:
            RPE = RPE[0,:,:].repeat(scores.shape[0], 1, 1)
            
        if sqrt:
            RPE = torch.sqrt(RPE)
        #print(scores.shape, RPE.shape)
        scores = torch.div(scores, RPE) #RPE
        p_attn = F.softmax(scores, dim = -1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn
    
        # Distance mask
    def get_dist_mask_tile(self,sentence_len, device):
        mask = torch.FloatTensor(sentence_len, sentence_len).to(torch.device(device))
        for i in range(sentence_len):
            for j in range(sentence_len):
                mask[i, j] = -abs(i-j)
        mask.unsqueeze_(0)
        return mask
    
    def masked_attention(self, query, key, value, mask=None, dropout=0.0):

        d_k = query.size(-1)

        alpha = 1.5
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        dist_mask_tile = self.get_dist_mask_tile(scores.shape[1], self.device)

        scores += alpha * dist_mask_tile


        p_attn = F.softmax(scores, dim = -1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn

    
    def distance_attention(self, query, key, value, mask=None, dropout=0.0): 
        #distance attention from https://www.sciencedirect.com/science/article/pii/S0951832022002733
        d_k = query.size(-1)
        #scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #, diff)

        x = torch.pow(torch.sqrt(torch.square(torch.subtract(query, key))/ math.sqrt(d_k)), 2)
        #print(x)

        scores = 1 / x.transpose(-2,-1)

        #print(scores.shape)

        p_attn = F.softmax(scores, dim = -1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn
    



    def forward(self, inputs):
        output = inputs

        if self.usePE:
            output = self.pe(output)

        if self.useCNN:
            output = self.layer1(output)
            cnn_out = output
            output = self.dropout1(output)
            #output = self.layer2(output)
            
        if len(self.cnnlayer) > 0:
            for i in range(len(self.cnnlayer)):
                output = self.CNNlayers[i][0](output)
        #print("2nd CNN out", output.shape)
        
        #print(output[0,0,0], self.residual)
        
        if self.residual:
            output += cnn_out
            #print(output[0,0,0])

            output = self.CNNlayers[i][1](output)
            output = self.maxpool1(output)
        #print("after maxpool ", output.shape)


        output = output.permute(0,2,1)
        identity = output

        #print("CNN after linear ", identity.shape)

        if self.useRNN:
            output, _ = self.RNN(output)
            F_RNN = output[:,:,:self.RNN_hiddenSize]
            R_RNN = output[:,:,self.RNN_hiddenSize:] 
            output = torch.cat((F_RNN,R_RNN),2)
            output = self.dropoutRNN(output)


        pAttn_concat = torch.Tensor([]).to(self.device)
        

        

        no_relativeAttn = 0
        for j in range(0,self.numofAttnLayers):
            

            attn_concat = torch.Tensor([]).to(self.device)
            
            for i in range(0,self.numMultiHeads):
                
                query, key, value = self.Q[j][i](output), self.K[j][i](output), self.V[j][i](output)

        
                if self.relativeAttn:
                    if self.mixed_attn:             
                        if self.numMultiHeads == 2:
                            no_relativeAttn = 1              
                        else:  
                            no_relativeAttn = self.mixed_attn_config * self.numMultiHeads
                    else:
                        no_relativeAttn = self.numMultiHeads             
                        
                if i < no_relativeAttn and j == 0:
                    # temporarily use distance masked attention
                    #attnOut,p_attn = self.masked_attention(query, key, value, dropout=0.2)

                    if self.residual:
                        RPE1_output = self.rd2(self.rd1(self.relative_dist))
                    else : 
                        RPE1_output = self.rd1(self.relative_dist)

                        
                    
                    if self.learnable_relativeAttn: 
                            RPE_shape = RPE1_output.shape 
                            RPE1_output = RPE1_output.view(RPE_shape[0], 1, RPE_shape[1], RPE_shape[2])               
                            
                            RPE1_output = self.dconv1(RPE1_output)
                            #RPE1_output = self.drelu(RPE1_output)
                            #RPE1_output = self.dconv2(RPE1_output)
                            RPE1_output = torch.sigmoid(RPE1_output)
                            RPE1_output = RPE1_output.view(RPE_shape[0], RPE_shape[1], RPE_shape[2])               

                            #print("Learnable weights...")
                            #print(self.WRD_2[i])
                            #print(RPE_shape) # removed the flattened verion 
                            #RPE1_flat_output = RPE1_output.view(RPE_shape[0], -1)

                            #RPE1_flat_output = self.WRD_1[0](RPE1_flat_output) # use joint fn
                            #RPE1_flat_output = self.d_act_1(RPE1_flat_output)
                            # RPE1_output = self.WRDj_1[0](RPE1_output, ) # use joint fn
                            # RPE1_output = self.d_act(RPE1_output)
                            
                            #RPE1_flat_output = self.WRD_2[0](RPE1_flat_output)
                            #RPE1_output = torch.reshape(RPE1_output, (RPE_shape[0], RPE_shape[1], RPE_shape[2]))
                            #RPE1_flat_output = self.d_act_2(RPE1_flat_output)
                            #RPE1_output = RPE1_flat_output.view(RPE_shape[0], RPE_shape[1], RPE_shape[2])

                    
                    attnOut,p_attn = self.relative_attention(query, key, value, RPE1_output, self.sqrt_dist, dropout=0.2)            
                else:
                    attnOut,p_attn = self.attention(query, key, value, dropout=0.2)

                attnOut = self.RELU[i](attnOut)
                #print(attnOut.shape)
                if self.usepooling:
                    attnOut = self.MAXPOOL[i](attnOut.permute(0,2,1)).permute(0,2,1)
                #print(p_attn.shape)

                attn_concat = torch.cat((attn_concat,attnOut),dim=2)
                if self.genPAttn and j == 0:
                    pAttn_concat = torch.cat((pAttn_concat, p_attn), dim=2)
                    
            if self.multiple_linear:
		    #print(attn_concat.shape)
		    #output = self.MultiHeadLinear_2[j](self.MultiHeadDropout(self.MHGeLU[j](self.MultiHeadLinear_1[j](attn_concat))))
                    attn_concat += identity
                    
                    output = self.MultiHeadLinearDropout(attn_concat) 
    

                    output = self.MultiHeadLinear_2[j](self.MHGeLU[j](self.MultiHeadLinear_1[j]
                                                                      (self.MultiHeadLayerNorm[j](output))))
                    output = self.MultiHeadLinearDropout2(output)

		    #output = output.permute(0,2,1)          
                    #output = (output)
		    #print(output.shape)
            else:
                    #("Attention concat ", attn_concat.shapeprint )

                    attn_concat += identity
                    output = self.MultiHeadLinearDropout(attn_concat)
                    output = self.MultiHeadLinear(output)
                    #print("After MultiheadLinear" , output.shape)
                    output = self.MHReLU(output)

            
            # if j == 0 and self.numofAttnLayers > 1:
            #     #print(output.shape, identity.shape)
            #     print("No residual connection in between attention layers")
            #     #output += identity
            #print(j)
            


        if self.readout_strategy == 'normalize':
            output = output.sum(axis=1)
            output = (output-output.mean())/output.std()
        else:
            output = self.attention_pool(output)
            #print("using this block??")


        #print("before fc3" , output.shape)	
        output = self.fc3(output)
        #print("fc3" , output.shape)	
        output = torch.nan_to_num(output)
        assert not torch.isnan(output).any()
        if self.genPAttn:
            return output,pAttn_concat#, RPE1_output
        else:
            return output
        

class AttentionNet_selection(nn.Module): #for the model that uses CNN, RNN (optionally), and MH attention
    def __init__(self, numLabels, params, device=None, seq_len = 200, genPAttn=True, reuseWeightsQK=False):
        super(AttentionNet_selection, self).__init__()
        self.numMultiHeads = params['num_multiheads']
        self.SingleHeadSize = params['singlehead_size']#SingleHeadSize
        self.MultiHeadSize = params['multihead_size']#MultiHeadSize
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
        self.numClasses = numLabels
        self.device = device
        self.reuseWeightsQK = reuseWeightsQK
        self.numInputChannels = params['input_channels'] #number of channels, one hot encoding
        self.genPAttn = genPAttn
        self.relativeAttn = params['relativeAttn']
        self.batch_size = params['batch_size']
        self.seqLength = seq_len

        self.last_channels = self.numInputChannels
        self.cnnlayer = []


        if self.usePE:
            self.pe = PositionalEncoding(d_model = self.numInputChannels, dropout=0.1)

        if self.useCNN and self.useCNNpool:

            print(self.numCNNfilters[0])
            
            self.layer1  = nn.Sequential(nn.Conv1d(in_channels=self.numInputChannels, out_channels=self.numCNNfilters[0],
                                         kernel_size=self.filterSize[0], padding=self.CNNpadding, bias=False),nn.BatchNorm1d(num_features=self.numCNNfilters[0]),
                                         nn.ReLU() if self.CNN1useExponential==False else Exponential(),
                                         nn.MaxPool1d(kernel_size=self.CNNpoolSize))
            self.dropout1 = nn.Dropout(p=0.2)

            self.last_channels = self.numCNNfilters[0]
        
        if self.useCNN and self.useCNNpool == False:
                
            self.layer1  = nn.Sequential(nn.Conv1d(in_channels=self.numInputChannels, out_channels=self.numCNNfilters[0],
                                         kernel_size=self.filterSize[0], padding=self.CNNpadding, bias=False),
                                         nn.BatchNorm1d(num_features=self.numCNNfilters),
                                         nn.ReLU() if self.CNN1useExponential==False else Exponential())
            self.dropout1 = nn.Dropout(p=0.2)

            self.last_channels = self.numCNNfilters[0]

        if len(self.numCNNfilters) > 1:

            for i, num_filters in enumerate(self.numCNNfilters):

                # Add next layers of cnn
                if i > 0:

                    self.cnnlayer.append(nn.Sequential(nn.Conv1d(in_channels=self.last_channels, out_channels=self.numCNNfilters[i],
                                          kernel_size=self.filterSize[i], padding=self.CNNpadding, bias=False),
                                          nn.BatchNorm1d(num_features=self.numCNNfilters[i]),
                                          nn.ReLU()))
                    
                    self.last_channels = self.numCNNfilters[i]
    
            # change shape of Q, K ,V

        if self.relativeAttn:
            self.rd = nn.MaxPool2d((9, 9), stride=(6, 6), padding=1) #33 x 33
            self.relative_position_k_q = RelativePosition(self.batch_size, self.seqLength - 1)
            self.relative_dist = self.relative_position_k_q(self.seqLength, self.seqLength)/ (self.seqLength - 1)

            # weight distance matrix
            self.WRD = nn.ModuleList([nn.Linear(in_features=(self.SingleHeadSize+1)  ** 2, out_features=(self.SingleHeadSize+1)  **2) for i in range(0,self.numMultiHeads)])

        if self.useRNN:
            self.RNN = nn.LSTM(self.numInputChannels if self.useCNN==False else self.numCNNfilters, self.RNN_hiddenSize, num_layers=2, bidirectional=True)
            self.dropoutRNN = nn.Dropout(p=0.4)
            self.Q = nn.ModuleList([nn.Linear(in_features=2*self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.K = nn.ModuleList([nn.Linear(in_features=2*self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.V = nn.ModuleList([nn.Linear(in_features=2*self.RNN_hiddenSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])

        if self.useRNN == False and self.useCNN == False:
            self.Q = nn.ModuleList([nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.K = nn.ModuleList([nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.V = nn.ModuleList([nn.Linear(in_features=self.numInputChannels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])

        if self.useRNN == False and self.useCNN == True:
            self.Q = nn.ModuleList([nn.Linear(in_features=self.last_channels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.K = nn.ModuleList([nn.Linear(in_features=self.last_channels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.V = nn.ModuleList([nn.Linear(in_features=self.last_channels, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])

        #reuse weights between Query (Q) and Key (K)
        if self.reuseWeightsQK:
            for i in range(0, self.numMultiHeads):
                self.K[i].weight = Parameter(self.Q[i].weight.t())	

        # for i in range(0, self.numMultiHeads):
        #         self.K[i].weight = Parameter(self.Q[i].weight.t())	

        self.RELU = nn.ModuleList([nn.ReLU() for i in range(0,self.numMultiHeads)])
        self.MultiHeadLinear = nn.Linear(in_features=self.SingleHeadSize*self.numMultiHeads, out_features=self.MultiHeadSize)#50
        self.MHReLU = nn.ReLU()

        if len(self.cnnlayer) > 0:

            self.CNNlayers = nn.ModuleList(self.cnnlayer)

        self.fc3 = nn.Linear(in_features=self.MultiHeadSize, out_features=self.numClasses)

    def attention(self, query, key, value, mask=None, dropout=0.0):
        #based on: https://nlp.seas.harvard.edu/2018/04/03/attention.html
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim = -1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn
    
    def relative_attention(self, query, key, value, RPE, mask=None, dropout=0.0): 
        #relative attention  
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #, diff)
        RPE = RPE[:scores.shape[0], :, :]
        if RPE.shape[0] < scores.shape[0]:
            RPE = RPE[0,:,:].repeat(scores.shape[0], 1, 1)
        #print(scores.shape, RPE.shape)
        scores = torch.div(scores, RPE) #RPE
        p_attn = F.softmax(scores, dim = -1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn
    
    def distance_attention(self, query, key, value, mask=None, dropout=0.0): 
        #distance attention from https://www.sciencedirect.com/science/article/pii/S0951832022002733
        d_k = query.size(-1)
        #scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #, diff)

        x = torch.pow(torch.sqrt(torch.square(torch.subtract(query, key))/ math.sqrt(d_k)), 2)
        #print(x)

        scores = 1 / x.transpose(-2,-1)

        #print(scores.shape)

        p_attn = F.softmax(scores, dim = -1)
        p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.matmul(p_attn, value), p_attn


    def forward(self, inputs):
        output = inputs

        if self.usePE:
            output = self.pe(output)

        if self.useCNN:
            output = self.layer1(output)
            output = self.dropout1(output)
            #output = self.layer2(output)
        if len(self.cnnlayer) > 0:
            for i in range(len(self.cnnlayer)):
                output = self.CNNlayers[i](output)


        output = output.permute(0,2,1)

        if self.useRNN:
            output, _ = self.RNN(output)
            F_RNN = output[:,:,:self.RNN_hiddenSize]
            R_RNN = output[:,:,self.RNN_hiddenSize:] 
            output = torch.cat((F_RNN,R_RNN),2)
            output = self.dropoutRNN(output)

        pAttn_concat = torch.Tensor([]).to(self.device)
        attn_concat = torch.Tensor([]).to(self.device)

        for i in range(0,self.numMultiHeads):

            query, key, value = self.Q[i](output), self.K[i](output), self.V[i](output)

            if self.relativeAttn:
                
                RPE1_output = self.rd(self.relative_dist)
                RPE_shape = RPE1_output.shape
                RPE1_output = self.WRD[i](torch.flatten(RPE1_output , start_dim=1))
                RPE1_output = torch.reshape(RPE1_output, (RPE_shape[0], RPE_shape[1], RPE_shape[2]))
                attnOut,p_attn = self.relative_attention(query, key, value, RPE1_output, dropout=0.2)

            else:
                attnOut,p_attn = self.attention(query, key, value, dropout=0.2)
            
                
            attnOut = self.RELU[i](attnOut)
            if self.usepooling:
                attnOut = self.MAXPOOL[i](attnOut.permute(0,2,1)).permute(0,2,1)
            print(attnOut.shape)
            attn_concat = torch.cat((attn_concat,attnOut),dim=2)
            if self.genPAttn:
                pAttn_concat = torch.cat((pAttn_concat, p_attn), dim=2)

        output = self.MultiHeadLinear(attn_concat)
        output = self.MHReLU(output)

        if self.readout_strategy == 'normalize':
            output = output.sum(axis=1)
            output = (output-output.mean())/output.std()

        output = self.fc3(output)	
        #assert not torch.isnan(output).any()
        if self.genPAttn:
            return output,pAttn_concat
        else:
            return output
