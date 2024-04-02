import pickle
from scipy.fft import fft, ifft, fftfreq
import numpy as np
import torch
def fourier_weight(T):

    w =  []
    s = 0.2
    freqs = np.arange(500)

    for i in freqs:

        if i <= T:

            w.append(1)
        
        else:

            w.append(1 / (1 + np.power((i -T), s)))
    return w 


if __name__ == "__main__":

    dir_path = "weights/Pos:1-6331(+)_batch.pckl"
    with open(dir_path ,'rb') as f:
        PAttn_1 = pickle.load(f)

    print(PAttn_1.shape)

    y = fft(PAttn_1)
    b = torch.tensor(PAttn_1+4).unsqueeze(dim=0)

    two_fft = torch.concat((torch.tensor(PAttn_1).unsqueeze(dim=0),b))

    print("fft batch")

    batch_f = torch.fft.fft2(two_fft)

    pos_fft = torch.fft.rfft(torch.tensor(PAttn_1), 33)


    torch_y = torch.fft.fft(torch.tensor(PAttn_1), 33)

    print(torch_y.shape)
    print(pos_fft.shape)


    #print(y[:, 0])
    #print(torch_y.detach().numpy()[:, 0])
    m = torch.abs(torch_y[1:PAttn_1.shape[0]//2, 1:PAttn_1.shape[1]//2]) #* (4.0/(PAttn_1.shape[0] * PAttn_1.shape[1])
    #m = torch.abs(torch_y)
    m = torch.abs(batch_f[:, 1:batch_f.shape[1]//2, 1:batch_f.shape[2]//2])
    print(m.shape)
    l1_norm = torch.linalg.norm(torch.linalg.norm(m, ord = 1, dim = 2), ord = 1, dim = 1)
    print(m[:,0,0])
    temp = torch.tensor([2,8], dtype=torch.float32)

    norm_m = (m.permute(2,1,0) / temp).permute(2,1,0)


    print(l1_norm)
    print(norm_m[:,0,0])

    print("SUM")

    print(np.sum(norm_m.detach().numpy()))

    

    T = 1.0 / 800.0
    N = 100

    x = np.linspace(0.0, N*T, N, endpoint=False)

    y = 2 + np.sin(100.0 * 2.0*np.pi*x)
   

    yf = fft(y)
    print("hello")
    print(np.sum(np.abs(yf[0:N//2]) / np.linalg.norm(np.abs(yf[0:N//2]), 1)))
    print(len(yf), yf[0], yf[50], yf.shape)
    #print(2.0/N * np.abs(yf[0:N//2]))
    #print(np.sum(2.0/N * np.abs(yf[1:N//2])))
    xf = fftfreq(N, T)[1:N//2]

    #print(fft(np.arange(4)))
    import matplotlib.pyplot as plt
    plt.plot(xf, 2.0/N * np.abs(yf[1:N//2]))
    plt.grid()

    weights = np.array(fourier_weight(7))

    plt.figure()
    #plt.plot(range(0, len(y)),y)
    plt.plot(range(0, 500), weights)
    #plt.show()
