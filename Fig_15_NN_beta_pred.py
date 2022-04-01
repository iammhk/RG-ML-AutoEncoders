import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from keras.models import load_model
#(x_train, _), (x_test, _) = mnist.load_data()
from keras import Model, regularizers, Input
from keras.layers.core import Dense, Dropout

lattice_size=32
in_dim=int(lattice_size*lattice_size)
out_dim=1
cmap = colors.ListedColormap(['black','white'])
bounds=[-1.5,0, 1.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

model_name="Ising_temp32_512_relu"

Temp_reg = load_model(model_name+".h5")
Temp_reg.summary()

#Preparing the Dataset
x_train = np.load("Ising_32_T_1000_60000_1.npz")["Lattice"]
x_train = np.append(x_train,np.load("Ising_32_T_1000_60000_2.npz")["Lattice"], axis=0)
x_train = np.append(x_train,np.load("Ising_32_T_1000_60000_3.npz")["Lattice"], axis=0)
x_train = np.append(x_train,np.load("Ising_32_T_1000_60000_4.npz")["Lattice"], axis=0)


x_test = np.load("Ising_32_T_1000_60000_Test.npz")["Lattice"]


y_train = np.load("Ising_32_T_1000_60000_1.npz")["beta"]
y_train = np.append(y_train,np.load("Ising_32_T_1000_60000_2.npz")["beta"])
y_train = np.append(y_train,np.load("Ising_32_T_1000_60000_3.npz")["beta"])
y_train = np.append(y_train,np.load("Ising_32_T_1000_60000_4.npz")["beta"])

y_test = np.load("Ising_32_T_1000_60000_Test.npz")["beta"]


x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_test = x_test.astype('float32')
y_test = y_test.astype('float32')

x_test = x_test.reshape(x_test.shape[0],in_dim)
y_test = y_test.reshape(y_test.shape[0],out_dim)
x_train = x_train.reshape(x_train.shape[0],in_dim)
y_train = y_train.reshape(y_train.shape[0],out_dim)
print(x_train.shape)
print(y_train.shape)


results = Temp_reg.evaluate(x_test, y_test, batch_size=10)
print("test loss", results)
#xspace= np.arange(0,j)
N=500
fig, ax = plt.subplots()
for i in range(N):
    rnd = np.random.randint(x_test.shape[0])
    y = Temp_reg.predict(x_test[rnd].reshape(1,in_dim))

    plt.scatter(y_test[rnd], y, marker='.', color = 'red')

scale_factor = 1


# now plot both limits against eachother
#ax.plot(, m*xspace + b)
ax.plot((0.01, 0.6), (0.01, 0.6), 'k-', alpha=0.75, zorder=0)
ax.plot((0.01, 0.6), (0.035, 0.625), 'k-', alpha=0.5, zorder=0)
ax.plot((0.01, 0.6), (-0.015, 0.575), 'k-', alpha=0.5, zorder=0)
#ax.plot((0.1, 0.6), (0.15, 0.65), 'k-', alpha=0.25, zorder=0)
#ax.plot((0.1, 0.6), (0.05, 0.55), 'k-', alpha=0.25, zorder=0)

ax.set_aspect('equal')

ax.set_xlim(0.1, 0.6)
ax.set_ylim(0.1, 0.6)
ax.legend(['y=x', '±0.025 err','MSE='+str(round(results,4))], loc='lower right')
plt.ylabel('Predicted β')
plt.xlabel('Actual β')
#fig.savefig('/Users/paul/Desktop/so.png', dpi=300)

plt.show()


