import numpy as np
import matplotlib.pyplot as plt

from rbm import *


if __name__ == "__main__":

    plt.ion()
    ax1 = plt.subplot(1,2,1)
    iv = plt.imshow(np.zeros([28, 28]))
    ax2 = plt.subplot(1,2,2)
    ih = plt.imshow(np.zeros([8, 8]))
     

    rbm = RBM()
    for k in range(200):
        print("epoca: ", k)
        rng.shuffle(x_train)  
        errs = []
        for batch in range(batch_num):

            curr_data = x_train[(batch*batch_size):((batch+1)*batch_size)]

            err = rbm.step(curr_data)
            errs.append(err)
        
        print(np.mean(err))
        if k%30 == 0: 
            im = x_train[np.random.randint(0, len(x_train))].copy()
            im[rng.randint(0, 28*28, 10*10)] = 0
            v, h = rbm.test(im)

            for i in range(10):
                iv.set_array(v[i].reshape(28, 28))
                iv.set_clim([np.min(v[i]), np.max(v[i])])
                plt.pause(0.1)
                ih.set_array(h[i].reshape(10,10))
                ih.set_clim([np.min(h[i]), np.max(h[i])])
                plt.pause(0.1)
                
            iv.set_array(v[i].reshape(28, 28))
            iv.set_clim([np.min(v[i]), np.max(v[i])])
            plt.pause(1)

