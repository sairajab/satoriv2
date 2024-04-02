import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters


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


class AttentionNet(nn.Module): #for the model that uses CNN, RNN (optionally), and MH attention
    def __init__(self, argSpace, params, device=None, seq_len = 200, genPAttn=True, reuseWeightsQK=False):
        super(AttentionNet, self).__init__()
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
        self.numClasses = argSpace.numLabels
        self.device = device
        self.reuseWeightsQK = reuseWeightsQK
        self.numInputChannels = params['input_channels'] #number of channels, one hot encoding
        self.genPAttn = genPAttn
        self.relativeAttn = params['relativeAttn']
        self.batch_size = params['batch_size']
        self.seqLength = seq_len


        if self.usePE:
            self.pe = PositionalEncoding(d_model = self.numInputChannels, dropout=0.1)

        if self.useCNN and self.useCNNpool:
            self.layer1  = nn.Sequential(nn.Conv1d(in_channels=self.numInputChannels, out_channels=self.numCNNfilters,
                                         kernel_size=self.filterSize, padding=self.CNNpadding, bias=False),nn.BatchNorm1d(num_features=self.numCNNfilters),
                                         nn.ReLU() if self.CNN1useExponential==False else Exponential(),
                                         nn.MaxPool1d(kernel_size=self.CNNpoolSize))
            self.dropout1 = nn.Dropout(p=0.2)

            # Add second layer of cnn
            self.layer2  = nn.Sequential(nn.Conv1d(in_channels=self.numCNNfilters, out_channels=self.RNN_hiddenSize,
                                          kernel_size=7, padding=self.CNNpadding, bias=False),
                                          nn.BatchNorm1d(num_features=self.RNN_hiddenSize),
                                          nn.ReLU())
            # change shape of Q, K ,V

        if self.relativeAttn:
            self.rd = nn.MaxPool2d((9, 9), stride=(6, 6), padding=1) #33 x 33
            #self.rd = nn.MaxPool2d((9, 9), stride=(6, 6)) #32 x 32
            self.relative_position_k_q = RelativePosition(self.batch_size, self.seqLength - 1)
            self.relative_dist = self.relative_position_k_q(self.seqLength, self.seqLength)/ (self.seqLength - 1)

            # weight distance matrix
            #self.WRD = nn.ModuleList([nn.Linear(in_features=self.SingleHeadSize ** 2, out_features=self.SingleHeadSize **2) for i in range(0,self.numMultiHeads)])
            #self.WRD = nn.ModuleList([nn.Linear(in_features=33 ** 2, out_features=33 **2) for i in range(0,self.numMultiHeads)])

        if self.useCNN and self.useCNNpool == False:
            self.layer1  = nn.Sequential(nn.Conv1d(in_channels=self.numInputChannels, out_channels=self.numCNNfilters,
                                         kernel_size=self.filterSize, padding=self.CNNpadding, bias=False),
                                         nn.BatchNorm1d(num_features=self.numCNNfilters),
                                         nn.ReLU() if self.CNN1useExponential==False else Exponential())
            self.dropout1 = nn.Dropout(p=0.2)

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
            self.Q = nn.ModuleList([nn.Linear(in_features=self.numCNNfilters, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.K = nn.ModuleList([nn.Linear(in_features=self.numCNNfilters, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.V = nn.ModuleList([nn.Linear(in_features=self.numCNNfilters, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])

            self.WK = nn.ModuleList([nn.Linear(in_features=self.MultiHeadSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.WQ = nn.ModuleList([nn.Linear(in_features=self.MultiHeadSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])
            self.WV = nn.ModuleList([nn.Linear(in_features=self.MultiHeadSize, out_features=self.SingleHeadSize) for i in range(0,self.numMultiHeads)])

        #reuse weights between Query (Q) and Key (K)
        if self.reuseWeightsQK:
            for i in range(0, self.numMultiHeads):
                self.K[i].weight = Parameter(self.Q[i].weight.t())	

        # for i in range(0, self.numMultiHeads):
        #         self.K[i].weight = Parameter(self.Q[i].weight.t())	

        self.RELU = nn.ModuleList([nn.ReLU() for i in range(0,self.numMultiHeads)])
        self.MultiHeadLinear = nn.Linear(in_features=self.SingleHeadSize*self.numMultiHeads, out_features=self.MultiHeadSize)#50
        self.MHReLU = nn.ReLU()

        self.MultiHeadLinear1 = nn.Linear(in_features=self.SingleHeadSize*self.numMultiHeads, out_features=self.MultiHeadSize) 
        self.MHReLU1 = nn.ReLU()

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


    def forward(self, inputs):
        output = inputs

        if self.usePE:
            output = self.pe(output)

        if self.useCNN:
            output = self.layer1(output)
            output = self.dropout1(output)
            #output = self.layer2(output)
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
                #RPE1_output = self.WRD[i](torch.flatten(RPE1_output , start_dim=1))
                #RPE1_output = torch.reshape(RPE1_output, (RPE_shape[0], RPE_shape[1], RPE_shape[2]))

                attnOut,p_attn = self.relative_attention(query, key, value, RPE1_output, dropout=0.2)
            else:
                attnOut,p_attn = self.attention(query, key, value, dropout=0.2)
                
            attnOut = self.RELU[i](attnOut)
            if self.usepooling:
                attnOut = self.MAXPOOL[i](attnOut.permute(0,2,1)).permute(0,2,1)
            attn_concat = torch.cat((attn_concat,attnOut),dim=2)
            if self.genPAttn:
                pAttn_concat = torch.cat((pAttn_concat, p_attn), dim=2)

        output = self.MultiHeadLinear(attn_concat)
        output = self.MHReLU(output)

        pAttn_concat1 = torch.Tensor([]).to(self.device)
        attn_concat1 = torch.Tensor([]).to(self.device)
        for i in range(0,self.numMultiHeads):

            query, key, value = self.WQ[i](output), self.WK[i](output), self.WV[i](output)
            attnOut,p_attn = self.attention(query, key, value, dropout=0.2) #self.relativeAttention(query, key, value,RPE1_output, dropout=0.2)

            attnOut = self.RELU[i](attnOut)
            #print(attnOut.shape)
            attn_concat1 = torch.cat((attn_concat1,attnOut),dim=2)
            if self.genPAttn:
                pAttn_concat1 = torch.cat((pAttn_concat1, p_attn), dim=2)

        output = self.MultiHeadLinear1(attn_concat1)
        output = self.MHReLU1(output)
        #print(output.shape)
            

        if self.readout_strategy == 'normalize':
            output = output.sum(axis=1)
            output = (output-output.mean())/output.std()

        output = self.fc3(output)	
        assert not torch.isnan(output).any()
        if self.genPAttn:
            return output,pAttn_concat1
        else:
            return output
