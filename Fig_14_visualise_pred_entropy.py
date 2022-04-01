import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from keras.models import load_model
from skimage.filters.rank import entropy
from skimage.morphology import disk, square

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def RG_decimation(config):
    #config=config.reshape(config.shape[0] * config.shape[1])
    half_root= int(np.sqrt(config.shape[0]/2))
    RG_config = np.zeros([int(config.shape[0]),int(config.shape[1]/2)])
    #print(config.shape,RG_config.shape)
    for i in range(config.shape[0]):
        for j in range(config.shape[1]):
            if ((i+j)%2==0): RG_config[int(i),int(j/2)] = config[i,j]
    #print("Initial",config)
    #print("RG:",RG_config.T)
    return RG_config

def RG_decimation_2(config):
    #config=config.reshape(config.shape[0] * config.shape[1])
    RG_config = np.zeros([int(config.shape[0]/2),int(config.shape[1])])
    #print(config.shape,RG_config.shape)
    for i in range(config.shape[0]):
        for j in range(config.shape[1]):
            if ((i)%2==0): RG_config[int(i/2),int(j)] = config[i,j]
    #print("Initial",config)
    #print("RG:",RG_config.T)
    return RG_config

lattice_size=32
kernal=3
in_dim=int(lattice_size*lattice_size)
out_dim=int(in_dim/4)

cmap = colors.ListedColormap(['black','white'])
bounds=[-1.5,0, 1.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

# This is the size of our encoded representations
encoding_dim = out_dim # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
model_name1="Final_Ising_Half_TL_Deci_32x32.h5"
model_name2="Final_Ising_Full_TL_Deci_32x32.h5"
model_name3="Full_Ising_AE_Deci_32x32.h5"
Half_RG_encoder = load_model(model_name1)
Full_RG_encoder = load_model(model_name2)
Auto_encoder = load_model(model_name3)
#RG_encoder.summary()

x_train = np.load("Ising_32_E_25-500_25.npz")["Lattice"]
x_train = x_train.astype('float32')
x_train = x_train.reshape(x_train.shape[0],in_dim)
print("Input shape:",x_train.shape)

y_train = np.load("Ising_32_E_25-500_25.npz")["beta"]
y_train = y_train.astype('float32')
y_train = y_train.reshape(y_train.shape[0],1)

#encoded_imgs = RG_encoder.predict(x_train[2000:2999])

'''w_in = np.array(RG_encoder.get_weights())
w_in_gen = np.array(Auto_encoder.get_weights())'''

HRG_img = Half_RG_encoder.predict(x_train[0:9999:4999])
FRG_img = Full_RG_encoder.predict(x_train[0:9999:4999])
AE_img = Auto_encoder.predict(x_train[0:9999:4999])
beta=y_train[0:9999:4999]
print(beta)
#print("Weights Shape:",w_in[1])
#print("Weights Shape Generic:",w_in_gen.shape)

i=0
lattice=x_train[i].reshape(lattice_size, lattice_size)
rg_lattice=RG_decimation_2(RG_decimation(lattice))
ae_lattice=AE_img[i].reshape(int(lattice_size), lattice_size)

#np.save("lattice_ising",lattice)
#np.save("lattice_rg",rg_lattice)

ae_lattice= np.sign(ae_lattice)
rg_lattice = (rg_lattice+1)/2
ae_lattice = (ae_lattice+1)/2
lattice = (lattice+1)/2
print("Lattice:", ae_lattice)

lattice = x_train[i].reshape(int(lattice_size), lattice_size)
hrg_lattice = HRG_img[i].reshape(int(lattice_size), lattice_size)
frg_lattice = FRG_img[i].reshape(int(lattice_size), lattice_size)
ae_lattice = AE_img[i].reshape(int(lattice_size), lattice_size)

np.save("lattice_ising",lattice)
np.save("lattice_hrg",hrg_lattice)
np.save("lattice_frg",frg_lattice)
np.save("lattice_ae",ae_lattice)

hrg_lattice= np.sign(hrg_lattice)
frg_lattice= np.sign(frg_lattice)
ae_lattice= np.sign(ae_lattice)

frg_lattice = (frg_lattice+1)/2
hrg_lattice = (hrg_lattice+1)/2
ae_lattice = (ae_lattice+1)/2
lattice = (lattice+1)/2

start=0
n = 3  # How many digits we will display
plt.figure(figsize=(30, 10))
for i in range(n):
    lattice = x_train[i*4999].reshape(int(lattice_size), lattice_size)
    hrg_lattice = HRG_img[i].reshape(int(lattice_size), lattice_size)
    frg_lattice = FRG_img[i].reshape(int(lattice_size), lattice_size)
    ae_lattice = AE_img[i].reshape(int(lattice_size), lattice_size)
    rg_lattice = RG_decimation_2(RG_decimation(lattice))
    print(rg_lattice)

    hrg_lattice = np.sign(hrg_lattice)
    rg_lattice = np.sign(rg_lattice)
    ae_lattice = np.sign(ae_lattice)

    rg_lattice = (rg_lattice + 1) / 2
    hrg_lattice = (hrg_lattice + 1) / 2
    ae_lattice = (ae_lattice + 1) / 2
    lattice = (lattice + 1) / 2

    #Original Lattice
    #pred=RG_encoder.predict(x_train[i+start])
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(lattice,cmap=cmap)
    plt.xlabel(np.average(entropy(lattice, square(kernal))))
    # plt.gray()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    # Display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(ae_lattice,cmap=cmap)
    plt.xlabel(np.average(entropy(ae_lattice, square(kernal))))
    #plt.gray()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    # Display reconstruction
    ax = plt.subplot(3, n, i + 1 + n+ n)
    plt.imshow(rg_lattice,cmap=cmap)
    plt.xlabel(np.average(entropy(rg_lattice, square(kernal))))
    # plt.gray()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    '''# Display reconstruction
    ax = plt.subplot(4, n, i + 1 + n+ n+ n)
    plt.imshow(frg_lattice,cmap=cmap)
    plt.xlabel(shannon_entropy(frg_lattice))
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)'''
plt.show()

