import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class VerMaxPool(nn.Module):
    """
    This max pool function takes the max of forward and rc 
    of motif scores.
    
    Input dim: (batch_size, seq_len, num_motifs)
    Output dim: (batch_size, seq_len, num_motifs // 2)
    """
    def __init__(self):
        super().__init__()
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        
        x = x.permute(0, 2, 1)  
        x = self.maxpool1d(x)  # Apply max pooling along motif dimension
        x = x.permute(0, 2, 1)  # Restore original shape (n, l, m/2)
        return x

class MyMaxPool1D(nn.Module):
    def __init__(self, pool_size=4, stride=4):
        super(MyMaxPool1D, self).__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size, stride=stride)

    def forward(self, x):
    # Compute padding to maintain same output size
        input_length = x.shape[-1]  # seq_len
        pad_needed = max(0, (input_length - 1) // self.stride * self.stride + self.pool_size - input_length)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        
        # Apply padding (F.pad expects (left, right) for 1D tensors)
        x = F.pad(x, (pad_left, pad_right), mode='constant', value=0)

        # Apply max pooling
        x = self.maxpool(x)
        return x
    
class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, embed_dim, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros( max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        # Handle odd d_model
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]  # Exclude the last column for odd d_model
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #print(f"Positional Encoding Input shape: {x.shape} : {self.pe.shape}")  # Debugging
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0.0):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)

#         self.dropout = nn.Dropout(dropout)

#     def attention(self, query, key, value, mask=None, dropout=0.0, attn_prob=None):
#         d_k = query.size(-1)
#         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#         p_attn = F.softmax(scores, dim=-1)
#         p_attn = F.dropout(p_attn, p=dropout, training=self.training)
#         if attn_prob is not None:
#             p_attn = attn_prob
#         return torch.matmul(p_attn, value), p_attn

#     def forward(self, query, key, value, mask=None):
#         batch_size = query.size(0)

#         # Linear projections
#         query = self.q_proj(query)
#         key = self.k_proj(key)
#         value = self.v_proj(value)

#         # Reshape to (batch_size, num_heads, seq_len, head_dim)
#         query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

#         # Apply attention
#         x, attn_weights = self.attention(query, key, value, mask, self.dropout.p)

#         # Concatenate heads and project
#         x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
#         x = self.out_proj(x)

#         return x, attn_weights

# class AttentionBlock(nn.Module):
#     def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, return_attn_coef=True, layer_or_batch="batch"):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.ff_dim = ff_dim
#         self.dropout = dropout
#         self.return_attn_coef = return_attn_coef
#         self.layer_or_batch = layer_or_batch

#         self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, ff_dim),
#             nn.ReLU(),
#             nn.Linear(ff_dim, embed_dim)
#         )
        
#         self.norm1 = nn.LayerNorm(embed_dim) if layer_or_batch == "layer" else nn.BatchNorm1d(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim) if layer_or_batch == "layer" else nn.BatchNorm1d(embed_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self, x, mask=None):
#         attn_output, attn_weights = self.attn(x, x, x, mask)
#         attn_output = self.dropout1(attn_output)
#         out1 = self.norm1((x + attn_output).permute(0, 2, 1)).permute(0, 2, 1)

#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output)
#         out2 = self.norm2((out1 + ffn_output).permute(0, 2, 1)).permute(0, 2, 1)    

#         if self.return_attn_coef:
#             return out2, attn_weights
#         else:
#             return out2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(query, key, value, mask=None, dropout=None, return_attn_coef=False):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    attn_weights = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    
    output = torch.matmul(attn_weights, value)
    
    if return_attn_coef:
        return output, attn_weights
    return output

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, batch_norm=True, return_attn_coef=False):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.return_attn_coef = return_attn_coef
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        # self.norm1 = nn.BatchNorm1d(embed_dim) if batch_norm else nn.LayerNorm(embed_dim)
        # self.norm2 = nn.BatchNorm1d(embed_dim) if batch_norm else nn.LayerNorm(embed_dim)
        
        # self.ffn = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim * 2, embed_dim)
        # )
    
    def forward(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output, attn_weights = attention(q, k, v, mask, self.dropout, return_attn_coef=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        # Residual Connection + Normalization
        # x = x + self.dropout(attn_output)
        # #x = self.norm1(x.transpose(1, 2)).transpose(1, 2)  # Adjust for batch norm
        
        # # Feed Forward Network + Residual Connection
        # ffn_output = self.ffn(x)
        # x = x + self.dropout(ffn_output)
        #x = self.norm2(x.transpose(1, 2)).transpose(1, 2)  # Adjust for batch norm
        #x = self.dropout(attn_output)
        #x = attn_output
        if self.return_attn_coef:
            return attn_output, attn_weights
        return attn_output


def truncated_normal_(tensor, mean=0.0, std=1e-2):
    """
    Apply Truncated Normal initialization (similar to Keras).
    Values are drawn from a normal distribution, but constrained to be within 2 std devs.
    """
    tensor.normal_(mean, std)
    while True:
        invalid = (tensor < mean - 2 * std) | (tensor > mean + 2 * std)
        if not torch.sum(invalid):
            break
        tensor[invalid] = torch.normal(mean, std, size=tensor[invalid].shape)
    return tensor

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, l2_lambda=5e-7, l1_lambda=1e-8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        # Initialize weights with Truncated Normal
        truncated_normal_(self.linear.weight, std=1e-2)

        # Initialize bias with constant value 0
        nn.init.constant_(self.linear.bias, 0.0)

        # Store regularization values
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda

    def forward(self, x):
        out = self.linear(x)
        
        # L2 Regularization (Weight Decay)
        l2_reg = self.l2_lambda * torch.sum(self.linear.weight ** 2)
        
        # L1 Regularization
        l1_reg = self.l1_lambda * torch.sum(torch.abs(self.linear.weight))
        
        # Add to loss (You may need to manually add this during training)
        return out, l1_reg + l2_reg

# Example Usage
# linear_layer = CustomLinear(in_features=512, out_features=256)
# input_tensor = torch.randn(10, 512)  # Batch of 10
# output, reg_loss = linear_layer(Linearinput_tensor)

# print("Output shape:", output.shape)  # Should be [10, 256]
# print("Regularization Loss:", reg_loss.item())


class TianaModel(nn.Module):
        def __init__(self, num_tf, pad_size, max_len, pssm_path, seq_size=300, output_bias=None, lr=5e-5, numClasses=164, getAttnGrad = False):
            super().__init__()
            self.seq_size = seq_size
            self.n_channels = 4
            self.dropout_rate = 0.4
            self.filter_size = max_len
            self.filter_stride = 1
            self.first_filter_num = num_tf * 2 + pad_size * 2
            self.getAttnGrad = getAttnGrad
            self.conv1 = nn.Conv1d(self.n_channels, self.first_filter_num, kernel_size=self.filter_size, stride=1, padding=0, bias=False)
            nn.init.uniform_(self.conv1.weight, a=-1e-6, b=1e-6)

            self.maxpool = VerMaxPool()
            self.bn1 = nn.BatchNorm1d(self.first_filter_num//2)
            self.relu = nn.ReLU()
            self.maxpool1d = MyMaxPool1D(pool_size=4, stride=4)
            self.dropout = nn.Dropout(self.dropout_rate)

            self.pos_encoding = PositionalEncoding(self.first_filter_num // 2, self.dropout_rate)
            self.transformer1 = AttentionBlock(self.first_filter_num // 2, 4, self.dropout_rate, return_attn_coef=True) #, layer_or_batch="batch")self.first_filter_num, 

            self.flatten = nn.Flatten()
            # flatten size = 145 * 184 --- 70 for 300
            self.fc1 = nn.Linear(self.first_filter_num // 2 * (70), 512)
                    # Weight initialization (Truncated Normal with stddev=1e-2)
            nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=1e-2)
            # Bias initialization (Constant value=0)
            nn.init.constant_(self.fc1.bias, 0.0)
            self.fc2 = nn.Linear(512, 256)
            # Weight initialization (Truncated Normal with stddev=1e-2)
            nn.init.trunc_normal_(self.fc2.weight, mean=0.0, std=1e-2)
            # Bias initialization (Constant value=0)
            nn.init.constant_(self.fc2.bias, 0.0)
            self.fc3 = nn.Linear(256, numClasses)#numClasses

            # Load PSSM weights
            with open(pssm_path, 'rb') as f:
                padded_motif_np_swap = np.load(f)
                padded_motif_np_swap = np.transpose(padded_motif_np_swap, (2, 1, 0))  

            self.conv1.weight.data = torch.from_numpy(padded_motif_np_swap).float() #[:, :num_tf * 2]
            self.conv1.weight.requires_grad = False
            # Regularization parameters
            self.weight_decay = 5e-7  # L2 regularization (kernel_regularizer)
            self.activity_decay = 1e-8  # L1 regularization (activity_regularizer)

        def forward(self, x):
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool1d(x)
            x = self.dropout(x)
            x = x.permute(0, 2, 1)  # (batch_size, seqlen, num_motifs)
            
            x = self.pos_encoding(x)  # This is where the error occurs
            x, pAttn = self.transformer1(x)

            x = self.flatten(x)
            x = self.dropout(x)
            self.fc1_activation = self.fc1(x)
            x = self.relu(self.fc1_activation)
            x = self.dropout(x)
            self.fc2_activation = self.fc2(x)
            x = self.relu(self.fc2_activation)
            x = self.fc3(x)
            if self.getAttnGrad:
                pAttnList = []
                return x, pAttn.view(-1, 280, 70), pAttn #580, 145 for 600bp (280,70)
            # x = self.sigmoid(x)
            # print(f"After sigmoid: {x.shape}")  # Debugging
            return x, pAttn.view(-1, 280, 70)
        
        def regularization_loss(self):
            # L2 regularization for weights
            l2_fc1_loss = self.weight_decay * torch.norm(self.fc1.weight, p=2)
            
            # L1 regularization for activations (assuming you have stored the activations)
            l1_fc1_loss = self.activity_decay * torch.norm(self.fc1_activation, p=1)
            
            # L2 regularization for weights
            l2_fc2_loss = self.weight_decay * torch.norm(self.fc2.weight, p=2)
            
            # L1 regularization for activations (assuming you have stored the activations)
            l1_fc2_loss = self.activity_decay * torch.norm(self.fc2_activation, p=1)
            
            return l2_fc1_loss + l1_fc1_loss + l2_fc2_loss + l1_fc2_loss

    # model = Model()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # return model, optimizer
    
    
    
    

if __name__ == "__main__":
    import torch
    import torch.nn as nn

    def get_padding(kernel_size, stride=1, dilation=1):
        """Calculates padding for 'same' padding in MaxPool2d."""
        padding = ((stride - 1) * (kernel_size - 1) + 2 * (dilation - 1)) // 2
        return padding
    
    

    print(f"Input shape: {input_tensor.shape}")   # Expected: [2, 368, 10]
    print(f"Output shape: {output.shape}") # Expected: [2, 368, 10]

    # Example usage
    max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
    padding = get_padding(kernel_size=3, stride=2)

    # Apply padding to input before passing it through MaxPool2d
    input_tensor = torch.randn(1, 3, 224, 224)
    padded_input = torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding))

    output = max_pool(padded_input)
    print(output.shape)
    # # load pssm
    # with open(pssm, 'rb') as f:
    #     motif_array = np.load(f)
        
    # #number of total tf (half of pssm)
    # ntf = motif_array.shape[-1]//2
        
    # # motif_size
    # motif_size = motif_array.shape[0]
        
    # # padding size
    # if ntf%4 ==0:
    #     npad = 0
    # elif ntf%4 !=0:
    #     npad = 4 - ntf%4
    # # Example usage
    # model, optimizer = make_model_attn_cluster(num_tf=ntf, pad_size=npad, max_len=motif_size, pssm_path='../motif_pssm.npy')
    # print(model)