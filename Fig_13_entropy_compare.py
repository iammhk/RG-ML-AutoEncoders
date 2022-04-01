from skimage.measure.entropy import shannon_entropy
from skimage.filters.rank import entropy
from skimage.morphology import disk, square
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from keras.models import load_model
import matplotlib.lines as mlines
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
    #half_root= int(np.sqrt(config.shape[0]/2))
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
in_dim=int(lattice_size*lattice_size)
out_dim=int(in_dim/4)
kernal=4
cmap = colors.ListedColormap(['black','white'])
bounds=[-1.5,0, 1.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

# This is the size of our encoded representations
encoding_dim = out_dim # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
model_name3="Full_Ising_AE_Deci_32x32.h5"

Auto_encoder = load_model(model_name3)
#RG_encoder.summary()

x_train = np.load("Ising_32_E_25-500_25.npz")["Lattice"]
x_train = x_train.astype('float32')
x_train = x_train.reshape(x_train.shape[0],in_dim)


y_train = np.load("Ising_32_E_25-500_25.npz")["beta"]
y_train = y_train.astype('float32')
y_train = y_train.reshape(y_train.shape[0],1)
#print("Input shape:",x_train.shape)
#encoded_imgs = RG_encoder.predict(x_train[2000:2999])'''

entropy_lattice, entropy_ae, entropy_ae1, entropy_ae2, entropy_frg, entropy_hrg = [],[],[],[],[],[]

config0=0
j=0
Tc=0.4352
for i in range(config0, x_train.shape[0]):
        config = x_train[i].reshape(1,1024)
        beta =  y_train[i].reshape(1)
        #print(i,beta)
        config_2x = RG_decimation_2(RG_decimation(x_train[i].reshape(32,32)))
        #HRG_img = Half_RG_encoder.predict(config)
        FRG_img = RG_decimation_2(RG_decimation(x_train[i].reshape(32,32)))
        AE_img = Auto_encoder.predict(config)

        lattice=x_train[i].reshape(lattice_size, lattice_size)
        #hrg_lattice=HRG_img.reshape(int(lattice_size), lattice_size)
        frg_lattice = FRG_img.reshape(int(lattice_size/2), int(lattice_size/2))
        ae_lattice=AE_img.reshape(int(lattice_size), lattice_size)

        ae_lattice = np.sign(ae_lattice)
        frg_lattice = np.sign(frg_lattice)
        #hrg_lattice = np.sign(hrg_lattice)

        entropy_lattice.append(np.average(entropy(lattice, square(kernal))))
        entropy_ae.append(np.average(entropy(ae_lattice, square(kernal))))
        entropy_frg.append(np.average(entropy(frg_lattice, square(kernal))))
        #entropy_hrg.append(shannon_entropy(hrg_lattice))

points=20
avg_lat= np.zeros(points)
avg_ae= np.zeros(points)
avg_ae2= np.zeros(points)
avg_frg= np.zeros(points)
#avg_hrg= np.average(entropy_hrg)
fig, ax = plt.subplots()
#twin1 = ax.twinx()

for j in range(10):
    k=j*100
    print(y_train[k],y_train[k+99])
    print(entropy_lattice[k], entropy_lattice[k + 99])
    avg_lat[j]= np.average(entropy_lattice[k : k + 99])
    print(avg_lat[j])
    avg_ae[j] = np.average(entropy_ae[k : k + 99])
    avg_frg[j] = np.average(entropy_frg[k : k + 99])
    ax.scatter(y_train[k], avg_lat[j], marker='.', color='black', label="Lattice'{0}'".format('.'))
    ax.scatter(y_train[k], avg_ae[j], marker='.', color='blue', label="AE predicted'{0}'".format('.'))
    ax.scatter(y_train[k], avg_frg[j], marker='.', color='red', label="Full-RG'{0}'".format('.'))
    #ax.scatter(j, entropy_hrg[j], marker='.', color='green', label="Half-RaG'{0}'".format('d'))

print(j)

xspace= np.arange(0.025,.5,0.025)


#p1, = ax.plot([0, 1, 2], [0, 1, 2], "b-", label="Shannon Entropy(H) per Spin")
#p2, = twin1.plot([0, 1, 2], [0, 3, 2], "r-", label="Shannon Entropy(H) per Spin")
'''print("Average Beta:",avg_lat)
print("Average AE1 Beta:",avg_ae)
print("Average AE2 Beta:",avg_ae2)
#print("Average HRG Beta:", avg_hrg)
print("Average FRG Beta:", avg_frg)

m0, b0 = np.polyfit(xspace, avg_lat, 1)
m1, b1 = np.polyfit(xspace, avg_ae, 1)
m2, b2 = np.polyfit(xspace, avg_frg, 1)'''

print(avg_lat)
#transform = ax.transAxes
#line.set_transform(transform)
ax.plot(xspace,avg_lat, color='black')
ax.plot(xspace,avg_ae, color='blue')
ax.plot(xspace,avg_frg, color='red')
ax.legend(['Initial Entropy', 'AE Entropy','Decimated Entropy'], loc='upper right')

ax.set_ylabel('Shannon Entropy(H) per Spin')
ax.set_xlabel('Inverse Temperature')

np.savetxt("xspace_4.csv", xspace, delimiter = ",")
np.savetxt("ent_lat_4.csv", avg_lat, delimiter = ",")
np.savetxt("ent_ae_4.csv", avg_ae, delimiter = ",")
np.savetxt("ent_frg_4.csv", avg_frg, delimiter = ",")
plt.show()

fig2=plt.plot()
#print(entropy_ae)
plt.plot(entropy_lattice)
plt.plot(entropy_ae)
#plt.plot(entropy_hrg)
plt.plot(entropy_frg)
plt.title('Inverse Temprature')
plt.ylabel('Beta')
plt.xlabel('Image')
plt.legend(['Lattice', 'AE', 'Half-RG TL','Full-RG TL'], loc='upper right')
plt.show()
