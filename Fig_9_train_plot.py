import numpy as np
import matplotlib.pyplot as plt

AEloss = np.load("training data\Final_Ising_AE_Deci_32x32_1.npz")["loss"]
AEloss = np.append(AEloss,np.load("training data\Final_Ising_AE_Deci_32x32_2.npz")["loss"], axis=0)
AEloss = np.append(AEloss,np.load("training data\Final_Ising_AE_Deci_32x32_3.npz")["loss"], axis=0)
AEloss = np.append(AEloss,np.load("training data\Final_Ising_AE_Deci_32x32_4.npz")["loss"], axis=0)
AEloss = np.append(AEloss,np.load("training data\Final_Ising_AE_Deci_32x32_5.npz")["loss"], axis=0)
AEloss = np.append(AEloss,np.load("training data\Final_Ising_AE_Deci_32x32_6.npz")["loss"], axis=0)
AEloss = np.append(AEloss,np.load("training data\Final_Ising_AE_Deci_32x32_7.npz")["loss"], axis=0)
AEloss = np.append(AEloss,np.load("training data\Final_Ising_AE_Deci_32x32_8.npz")["loss"], axis=0)

AEvloss = np.load("training data\Final_Ising_AE_Deci_32x32_1.npz")["val_loss"]
AEvloss = np.append(AEvloss,np.load("training data\Final_Ising_AE_Deci_32x32_2.npz")["val_loss"], axis=0)
AEvloss = np.append(AEvloss,np.load("training data\Final_Ising_AE_Deci_32x32_3.npz")["val_loss"], axis=0)
AEvloss = np.append(AEvloss,np.load("training data\Final_Ising_AE_Deci_32x32_4.npz")["val_loss"], axis=0)
AEvloss = np.append(AEvloss,np.load("training data\Final_Ising_AE_Deci_32x32_5.npz")["val_loss"], axis=0)
AEvloss = np.append(AEvloss,np.load("training data\Final_Ising_AE_Deci_32x32_6.npz")["val_loss"], axis=0)
AEvloss = np.append(AEvloss,np.load("training data\Final_Ising_AE_Deci_32x32_7.npz")["val_loss"], axis=0)
AEvloss = np.append(AEvloss,np.load("training data\Final_Ising_AE_Deci_32x32_8.npz")["val_loss"], axis=0)

#loss = np.append(loss,np.load("training data\Final_Ising_AE_Deci_32x32_generic_3.npz")["loss"], axis=0)
#loss = np.append(loss,np.load("training data\Final_Ising_AE_Deci_32x32_generic_4.npz")["loss"], axis=0)
#loss = np.append(loss,np.load("training data\Final_Ising_AE_Deci_32x32_generic_5.npz")["loss"], axis=0)


HRGloss = np.load("training data\Final_Ising_Half_Hard_Deci_32x32_1.npz")["loss"]
HRGloss = np.append(HRGloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_2.npz")["loss"], axis=0)
HRGloss = np.append(HRGloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_3.npz")["loss"], axis=0)
HRGloss = np.append(HRGloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_4.npz")["loss"], axis=0)
HRGloss = np.append(HRGloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_5.npz")["loss"], axis=0)
HRGloss = np.append(HRGloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_6.npz")["loss"], axis=0)
HRGloss = np.append(HRGloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_7.npz")["loss"], axis=0)
HRGloss = np.append(HRGloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_8.npz")["loss"], axis=0)

HRGvloss = np.load("training data\Final_Ising_Half_Hard_Deci_32x32_1.npz")["val_loss"]
HRGvloss = np.append(HRGvloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_2.npz")["val_loss"], axis=0)
HRGvloss = np.append(HRGvloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_3.npz")["val_loss"], axis=0)
HRGvloss = np.append(HRGvloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_4.npz")["val_loss"], axis=0)
HRGvloss = np.append(HRGvloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_5.npz")["val_loss"], axis=0)
HRGvloss = np.append(HRGvloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_6.npz")["val_loss"], axis=0)
HRGvloss = np.append(HRGvloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_7.npz")["val_loss"], axis=0)
HRGvloss = np.append(HRGvloss,np.load("training data\Final_Ising_Half_Hard_Deci_32x32_8.npz")["val_loss"], axis=0)

#vloss = np.load("training data\Final_Ising_AE_Deci_32x32_1.npz")["loss"]
#vloss = np.append(vloss,np.load("training data\Final_Ising_AE_Deci_32x32_2.npz")["loss"], axis=0)

print(AEloss[:500])
FRGloss = np.full((HRGloss.shape[0]),0.6001)
FRGvloss = np.full((HRGloss.shape[0]),0.601)
plt.plot(AEloss)
plt.plot(HRGloss)
plt.plot(FRGloss)
#plt.plot(AEvloss, color= "Red")
#plt.plot(HRGvloss, color= "Yellow")
#plt.plot(FRGvloss, color= "Purple")

plt.grid(True)
#plt.title('model loss')
plt.ylabel('Hinge-Loss')
plt.xlabel('Epochs')
plt.ylim(0.2, 1)
plt.legend(['M1: AE', 'M2: Half-RG HC','M3: Full-RG TL'], loc='upper right')
plt.show()

print("Final training loss:",AEloss[-1],HRGloss[-1],FRGloss[-1])
print("Final validation loss:",AEvloss[-1],HRGvloss[-1],FRGvloss[-1])