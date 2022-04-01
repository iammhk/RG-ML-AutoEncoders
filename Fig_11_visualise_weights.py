import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from keras.models import load_model
from matplotlib import cbook
import matplotlib.gridspec as gridspec
#(x_train, _), (x_test, _) = mnist.load_data()
from keras import layers, Model, regularizers,Input

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
    RG_config = np.zeros([int(config.shape[0]/2),config.shape[1]])
    #print(config.shape,RG_config.shape)
    for i in range(config.shape[0]):
        for j in range(config.shape[1]):
            if ((i+j)%2==0): RG_config[int(i/2),j] = config[i,j]
    #print("Initial",config)
    #print("RG:",RG_config.T)
    return RG_config

def get_demo_image():
    z = cbook.get_sample_data("axes_grid/bivariate_normal.npy", np_load=True)
    # z is a numpy array of 15x15
    return z, (-3, 4, -4, 3)


lattice_size=32
in_dim=int(lattice_size*lattice_size)
out_dim=int(in_dim/4)

cmap = colors.ListedColormap(['black','white'])
bounds=[-1.5,0, 1.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

# This is the size of our encoded representations
encoding_dim = out_dim # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
model_name1="Final_Ising_Half_TL_Deci_32x32.h5"
model_name2="Final_Ising_Full_TL_Deci_32x32.h5"
model_name3="Final_Ising_AE_Deci_32x32_generic.h5"
Half_RG_encoder = load_model(model_name1)
Full_RG_encoder = load_model(model_name2)
Auto_encoder = load_model(model_name3)
#RG_encoder.summary()

x_train = np.load("Ising_32_Deci_1.npz")["Lattice"]
x_train = x_train.astype('float32')
x_train = x_train.reshape(x_train.shape[0],in_dim)
print("Input shape:",x_train.shape)
#encoded_imgs = RG_encoder.predict(x_train[2000:2999])

w_HRG = np.array(Half_RG_encoder.get_weights())
w_AE = np.array(Auto_encoder.get_weights())

#print("Weights Shape:",w_in[1])
#print("Weights Shape Generic:",w_in_gen.shape)

'''w1=np.load("Ising_Encoder32x32-32x16Deci.npy")
w2=np.load("Ising_Encoder32x16-16x16Deci.npy")
w1 = w1.reshape(w1.shape[1],w1.shape[2])
w2 = w2.reshape(w2.shape[1],w2.shape[2])'''
#print("WEIGHTS:", w)


'''pred_RG = RG_encoder.predict(x_train[2000:2021])
pred_AE = Auto_encoder.predict(x_train[2000:2021])'''

w1= np.matmul(w_HRG[0],w_HRG[0].T)
w2= np.matmul(w_AE[0],w_AE[0].T)
w3= np.matmul(w_HRG[1],w_HRG[1].T)
w4= np.matmul(w_AE[1],w_AE[1].T)

print(w1[:16,:16])

'''Z, extent = get_demo_image()

fig, ax = plt.subplots(figsize=[5, 4])
ax.imshow(w1)

# inset axes....
axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
axins.imshow(w1, extent=extent)
# sub region of the original image
x1, x2, y1, y2 = 1, 1, 5, 5
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')

ax.indicate_inset_zoom(axins, edgecolor="black")

plt.show()'''

fig1, ax1 = plt.subplots(figsize=[5, 4])

f1=ax1.imshow(w1[:16,:16])
#plt.figure(figsize=(30, 30))
#plt.pcolor(np.array(w_HRG[0]))
#ax1.set_yticks([0, 512, 1024])
#ax1.set_xticks([0, 256, 512])
fig1.colorbar(f1, pad = 0.02)
plt.show()

fig1, ax1 = plt.subplots(figsize=[5, 4])

f1=ax1.imshow(w_AE[0])
#plt.figure(figsize=(30, 30))
#plt.pcolor(np.array(w_HRG[0]))
ax1.set_yticks([0, 512, 1024])
ax1.set_xticks([0, 256, 512])
fig1.colorbar(f1, pad = 0.02)
plt.show()

fig1, ax1 = plt.subplots(figsize=[5, 4])

f1=ax1.imshow(w_HRG[1])
#plt.figure(figsize=(30, 30))
#plt.pcolor(np.array(w_HRG[0]))
ax1.set_yticks([0, 256, 512])
ax1.set_xticks([0, 128, 256])
fig1.colorbar(f1, pad = 0.02)
plt.show()

fig1, ax1 = plt.subplots(figsize=[5, 4])

f1=ax1.imshow(w_AE[1])
#plt.figure(figsize=(30, 30))
#plt.pcolor(np.array(w_HRG[0]))
ax1.set_yticks([0, 256, 512])
ax1.set_xticks([0, 128, 256])
fig1.colorbar(f1, pad = 0.02)
plt.show()

'''fig = plt.figure(figsize = (2,2))
gs1 = gridspec.GridSpec(2, 2)
gs1.update(wspace=0.005, hspace=0.05)

ax = plt.subplot(gs1[0])
ax.imshow(w1)
adjust_spines(ax, ['left'])
setp = plt.setp(ax, yticks=[0, 512,1024])

ax = plt.subplot(gs1[1])
ax.imshow(w2)
adjust_spines(ax, [])


ax = plt.subplot(gs1[2])
ax.imshow(w3)
adjust_spines(ax, ['left','bottom'])
plt.setp(ax, xticks=[0, 512,1024],yticks=[0, 512,1024])

ax = plt.subplot(gs1[3])
ax.imshow(w4)
adjust_spines(ax, [ 'bottom'])
plt.setp(ax, xticks=[0, 512,1024])

plt.show()'''

'''start=2000
n = 5  # How many digits we will display
plt.figure(figsize=(40, 10))
for i in range(n):
    #Original Lattice
    #pred=RG_encoder.predict(x_train[i+start])
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(x_train[i+start].reshape(lattice_size, lattice_size), cmap=cmap, norm=norm)
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(4, n, i + 1 + n)
    plt.imshow(AE_img[i].reshape(lattice_size,lattice_size),cmap=cmap, norm=norm)
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(4, n, i + 1 + n+ n)
    plt.imshow(HRG_img[i].reshape(lattice_size, lattice_size), cmap=cmap, norm=norm)
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(4, n, i + 1 + n+ n+ n)
    plt.imshow(FRG_img[i].reshape(lattice_size, lattice_size), cmap=cmap, norm=norm)
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()'''

