import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
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
        files = glob.glob(self.dir + os.sep +"*.png")
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

def sim():
    
    seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
    rng = np.random.RandomState(seed) 
    np.savetxt("seed",[seed])

    
    epochs = 300
    test_num = 10
    eta = 0.000005

    # train ----
    rbm = RBM() 
    errors = np.zeros(epochs)
    for k in range(epochs):
        print(k)
        rng.shuffle(x_train)  
        errs = []
        for batch in range(batch_num):

            curr_data = x_train[(batch*batch_size):((batch+1)*batch_size)]

            err = rbm.step(curr_data)
            errs.append(err)
        errors[k] = np.mean(err)

    # test ----
    tests = []
    for test in range(test_num):
        im = x_train[rng.randint(0, len(x_train))].copy()
        imm = im[rng.randint(0, 28*28, int(28*28*0.2))] 
        im[rng.randint(0, 28*28, int(28*28*0.2))]  = 1 - imm 
        v, h = rbm.test(im)
        tests.append([v, h])

    # plot ----
    fig = plt.figure(figsize=(6,3))
    ax1 = plt.subplot(1,3,1, aspect="equal")
    ax1.set_axis_off()
    iv = plt.imshow(np.zeros([28, 28]), cmap=plt.cm.viridis)
    ax2 = plt.subplot(1,3,3, aspect="equal")
    ax2.set_axis_off()
    ih = plt.imshow(np.zeros([8, 8]), cmap=plt.cm.viridis)
    
    ax_3 = plt.subplot(1,3,2)
    ax_3.set_axis_off()
    arr = ax_3.arrow(0.1, 3.5, 0.6, 0, head_width=0.1, head_length=0.2, fc='#aa3333', ec='#aa3333')
    fig.tight_layout(h_pad=0.2, w_pad=0.2)

    fig1 = plt.figure()
    ax3 = plt.subplot(111)
    err_plot, = plt.plot(0,0)
    ax3.set_xlim([-epochs*0.1, epochs*1.1])
    ax3.set_ylim([-0.25*0.1, 0.25*1.1])
    
    fig2 = rbm.get_weight_graphs()
    
    vman = vidManager(fig, "recon", "recon")
    eman = vidManager(fig1, "error", dirname="error")
    wman = vidManager(fig2, "weights", dirname="weights")
   
    name = vman.name
    for t in range(test_num):
        
        v, h = tests[t]
        vman.name = name +"_s%08d" % t
        for i in range(10):
            iv.set_array(v[i].reshape(28, 28))
            iv.set_clim([np.min(v[i]), np.max(v[i])])
            vman.save_frame()
            arr.remove()
            arr = ax_3.arrow(0.0, 0.5, 0.8, 0, lw=3, head_width=0.1, head_length=0.2, fc='#aa3333', ec='#aa3333')

            ih.set_array(h[i].reshape(10,10))
            ih.set_clim([np.min(h[i]), np.max(h[i])])
            vman.save_frame()
            arr.remove()
            arr = ax_3.arrow(1.0, 0.5, -0.8, 0, lw=3, head_width=0.1, head_length=0.2, fc='#aa3333', ec='#aa3333')
        
        arr.remove()
        arr = ax_3.arrow(0.0, 0.5, 0.8, 0, lw=3, head_width=0.1, head_length=0.2, fc='#aa3333', ec='#aa3333')            
        iv.set_array(v[i].reshape(28, 28))
        iv.set_clim([np.min(v[i]), np.max(v[i])])

        vman.save_frame()
        vman.mk_video()
    
    err_plot.set_data(np.arange(k+1), errors[:(k+1)])
    eman.save_frame()
    eman.mk_video()
    
    rbm.get_weight_graphs()
    wman.save_frame()
    wman.mk_video()


# main ----
if __name__ == "__main__": sim()

