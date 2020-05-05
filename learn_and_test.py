import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from rbm import *


class vidManager:
    def __init__(self, fig, name="vid", dirname="frames", duration=300):
        self.name = name
        self.fig = fig
        self.dir = dirname
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

    def save_frame(self):

        self.fig.savefig(self.dir + os.sep + 
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

def sim(plot_clear=None, plot_display=None, kk=5):

    fig = plt.figure()
    ax1 = plt.subplot(1,2,1)
    iv = plt.imshow(np.zeros([28, 28]))
    ax2 = plt.subplot(1,2,2)
    ih = plt.imshow(np.zeros([8, 8]))
     
    fig1 = plt.figure()
    ax3 = plt.subplot(111)
    err_plot, = plt.plot(0,0)

    vman = vidManager(fig, "recon")
    eman = vidManager(fig1, "error", dirname="eframes")
    
    rbm = RBM()
    errors = np.zeros(200)
    for k in range(200):
        rng.shuffle(x_train)  
        errs = []
        for batch in range(batch_num):

            curr_data = x_train[(batch*batch_size):((batch+1)*batch_size)]

            err = rbm.step(curr_data)
            errs.append(err)
        errors[k] = np.mean(err)
        if k%kk == 0: 
            if plot_clear: plot_clear()
            vman.clear()
            eman.clear()
            im = x_train[np.random.randint(0, len(x_train))].copy()
            im[rng.randint(0, 28*28, 10*10)] = 0
            v, h = rbm.test(im)

            for i in range(10):
                iv.set_array(v[i].reshape(28, 28))
                iv.set_clim([np.min(v[i]), np.max(v[i])])
                vman.save_frame()
                ih.set_array(h[i].reshape(10,10))
                ih.set_clim([np.min(h[i]), np.max(h[i])])
                vman.save_frame()
                
            iv.set_array(v[i].reshape(28, 28))
            iv.set_clim([np.min(v[i]), np.max(v[i])])
            vman.save_frame()
            vman.mk_video()
           
            err_plot.set_data(np.arange(k+1), errors[:(k+1)])
            ax3.set_xlim([-k*(0.1),k*1.1])
            eman.save_frame()
            eman.mk_video()
            if plot_display: plot_display()




