import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigm

from keras.datasets import mnist

rng = np.random.RandomState()

#def sigm(x, t=1): 
#    return 1./(1 + np.exp(-x/t))

(x_train, y_train), (x_test, y_test) = mnist.load_data()
data_size, input_side,_ = x_train.shape
x_train = x_train.reshape(data_size, input_side**2) 
x_train = 2*(x_train/255)-1

batch_num = 10
batch_size = data_size//batch_num


class RBM:

    def __init__(self, nv = 28*28, nh = 10*10, eta=0.0001):

        self.nv = nv
        self.nh = nh
        self.eta = eta

        self.w = np.sqrt(6.0/(nv+nh))*rng.uniform(
                -1,1,size=(nv, nh))
        self.vbias = np.zeros(self.nv)
        self.hbias = np.zeros(self.nh)


    def step(self, inp):
        
        vp0 = sigm(inp)
        
        # gibbs
        v0 = 1*(vp0>0.5)
        hp0 = sigm(np.dot(v0, self.w) + self.hbias) 
        h0 = rng.binomial(1,hp0) 
        
        vp1 = sigm(np.dot(h0, self.w.T) + self.vbias)
        v1 = rng.binomial(1,vp1) 
        
        # forward
        hp1 = sigm(np.dot(v1, self.w) + self.hbias)
        h1 = rng.binomial(1,hp1)
 
        # learn
        self.w += self.eta*(np.matmul(v0.T, hp0) - np.matmul(v1.T, hp1))
        self.vbias += self.eta*np.mean(v0 - v1, 0)
        self.hbias += self.eta*np.mean(hp0 - hp1, 0)
       
        # error
        return np.mean((v1 - v0)**2)

    def test(self, inp, k=10):
        
        vp0 = sigm(inp)
        
        # gibbs
        v0 = 1*(vp0>0.5)
        
        vs = [v0]
        hs = []

        for k in range(k):
                
            hp0 = sigm(np.dot(v0, self.w) + self.hbias) 
            h0 = rng.binomial(1,hp0) 
            hs.append(h0)

            vp1 = sigm(np.dot(h0, self.w.T) + self.vbias)
            v1 = rng.binomial(1,vp1) 
            vs.append(v1)
             
            v0 = v1.copy()

        return vs, hs

    def format_w(self):

        nv, nh = self.nv, self.nh

        ww =  self.w.reshape(
                int(np.sqrt(nv)),
                int(np.sqrt(nv)),
                int(np.sqrt(nh)),
                int(np.sqrt(nh)))
        ww = ww.transpose(2,3,0,1)

        return ww


if __name__ == "__main__":

    plt.ion()

    imgs = []
    for x in range(10):
        imgs.append([])
        for y in range(10):
            plt.subplot(4, 4, x*4 + y + 1)
            imgs[-1].append(plt.imshow(np.zeros([10, 10])))
    
    rbm = RBM()
    
    for k in range(200):

        rng.shuffle(x_train)  

        for batch in range(batch_num):

            curr_data = x_train[(batch*batch_size):((batch+1)*batch_size)]

            err = rbm.step(curr_data)
            
            print(err)
            if k%1 == 0: 

                w = rbm.format_w()
                
                for x in range(10):
                    for y in range(10):
                        ww = w[x][y]
                        imgs[x][y].set_array(ww)
                        imgs[x][y].set_clim([ww.min(), ww.max()])

                plt.pause(0.1)
