import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from rbm import *


class vidManager:
    def __init__(self, name="vid", duration=300):
        self.name = name
        self.dir = "frames"
        self.duration = duration
        self.clear()

    def clear(self):
        self.t = 0
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        files = glob.glob(self.dir + os.sep +"*")
        for f in files:
            if(os.path.isfile(f)):
                os.remove(f)

    def save_frame(self, fig):

        fig.savefig(self.dir + os.sep + 
                self.name +"%08d.png" % self.t)
        self.t += 1

    def mk_video(self):

        # Create the frames
        frames = []
        imgs = glob.glob(self.dir + os.sep + self.name + "*.png")
        for i in sorted(imgs):
            new_frame = Image.open(i)
            frames.append(new_frame)

        # Save into a GIF file that loops forever
        frames[0].save(self.dir + os.sep + self.name +'.gif', 
                format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=self.duration, loop=0)

if __name__ == "__main__":

    vman = vidManager("recon")
   
    fig = plt.figure()
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
            vman.clear()
            im = x_train[np.random.randint(0, len(x_train))].copy()
            im[rng.randint(0, 28*28, 10*10)] = 0
            v, h = rbm.test(im)

            for i in range(10):
                iv.set_array(v[i].reshape(28, 28))
                iv.set_clim([np.min(v[i]), np.max(v[i])])
                vman.save_frame(plt.gcf())
                ih.set_array(h[i].reshape(10,10))
                ih.set_clim([np.min(h[i]), np.max(h[i])])
                vman.save_frame(plt.gcf())
                
            iv.set_array(v[i].reshape(28, 28))
            iv.set_clim([np.min(v[i]), np.max(v[i])])
            vman.save_frame(plt.gcf())
            vman.mk_video()


