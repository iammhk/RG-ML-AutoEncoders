import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from keras.models import load_model
import matplotlib.lines as mlines


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

cmap = colors.ListedColormap(['black','white'])
bounds=[-1.5,0, 1.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

# This is the size of our encoded representations
encoding_dim = out_dim # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

model_name1="Full_Ising_AE_Deci_32x32.h5"
model_name2="Ising_temp32_512_relu.h5"


Auto_encoder = load_model(model_name1)
beta_reg32 = load_model(model_name2)
#RG_encoder.summary()

x_train = np.load("Ising_32_T_1000_60000_1.npz")["Lattice"]
x_train = x_train.astype('float32')
x_train = x_train.reshape(x_train.shape[0],in_dim)


y_train = np.load("Ising_32_T_1000_60000_1.npz")["beta"]
y_train = y_train.astype('float32')
y_train = y_train.reshape(y_train.shape[0],1)
#print("Input shape:",x_train.shape)
#encoded_imgs = RG_encoder.predict(x_train[2000:2999])

beta_lattice, beta_ae, beta_ae1, beta_ae2, beta_frg, beta_hrg = [],[],[],[],[],[]
fig, ax = plt.subplots()
config0=0
j=0
Tc=0.4352
for i in range(config0, x_train.shape[0],5):
    #rnd = np.random.randint(x_train.shape[0])
    # print("Random Number:",i)
    # print(x_train[i].shape)
    if (y_train[i] >= 0.1):
        config = x_train[i].reshape(1,1024)
        beta =  y_train[i].reshape(1)
        #config_2x = RG_decimation_2(RG_decimation(x_train[i].reshape(32,32)))
        AE_img = Auto_encoder.predict(config)

        lattice=x_train[i].reshape(lattice_size, lattice_size)
        #ae_lattice=AE_img.reshape(int(lattice_size), lattice_size)

        #ae_lattice= np.sign(ae_lattice)
        #beta_effective = beta_reg32.predict(ae_lattice.reshape(1, 1024)).reshape(1) - beta_reg32.predict(config)
        beta_ae.append(beta_reg32.predict(np.sign(AE_img)).reshape(1))
        beta_frg.append(3*np.log(np.cosh(4*beta))/8)
        #print(beta)

        if (j%20==0):
            ax.scatter(beta, beta_ae[j], marker='.', color='blue', label="AE predicted'{0}'".format('.'))
            #print(beta)

        j+=1
print(j)

xspace= np.arange(0.1,0.6, 0.00025)
avg_lat= np.average(beta_lattice)
avg_frg= np.average(beta_frg)


print("Average Beta:",avg_lat)
print("Average FRG Beta:", avg_frg)

print(xspace.shape)
m, b = np.polyfit(xspace, beta_ae, 1)
print("m,b=", m,b)
line3 = mlines.Line2D([0.1, 0.6], [0.1,0.6], color='black')

ax.add_line(line3)
ax.plot(xspace, beta_frg, color='red')
ax.plot(xspace, m*xspace + b, color='blue')
ax.legend(['Intitial β', 'β`~ln(cosh(β))', 'AE predicted β'], loc='lower right')
ax.set_xlim(0.1, 0.6)
ax.set_ylim(0.1, 0.6)
#ax.title.set_text('Change ')
ax.set_xlabel('Initial (β)')
ax.set_ylabel('Predicted (β`)')


#plt.xticks(range(0.5, 1))
#plt.yticks(range(0.5, 1))
#plt.xlim(0.7, 1.1)
#plt.ylim(0.7, 1.1)

    #plt.scatter(x, y, s=area2, marker='o', c=c)
# Show the boundary between the regions:

#theta = np.arange(0, np.pi / 2, 0.01)
#plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))
plt.show()

