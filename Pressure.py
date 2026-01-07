import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


v = 20
#x_pos = np.linspace(0, 1, 113)
alpha_list = np.append(np.arange(-5, 16, 0.5) , np.arange(16, 9.5, -0.5)) 
#print(alpha_list)


data = pd.read_csv("raw_airfoil_20mps.txt", skiprows= [ 1], sep = "\t")
#print(data)
#data = data.drop(columns = ["Run_nr", "Time", "Alpha", "Delta_Pb", "P_bar", "T", "rpm", "rho"])
rho = data[data.columns[[7]]].to_numpy()
#print(rho)
p_inf = data[data.columns[[117]]].to_numpy()
#print(p_inf)
data = data.drop(data.columns[[0,1,2,3,4,5,6,7]], axis = 1)
data_list = data.to_numpy()

#print(data[index = 0])
#print(data.shape)
#print(alpha_list.shape)


def Cp(p, p_inf, rho, v):
    Cp = (p - p_inf) / (0.5 * rho * v**2)

    return Cp



#print(list[0])
Cp_list =  Cp(data_list[0], p_inf[0], rho[0], v)

for i in range(1, 55):
    Cp1 =   Cp(data_list[i], p_inf[i], rho[i], v)
    Cp_list = np.vstack((Cp_list, Cp1))

print(Cp_list.shape)

pos_y_data = []
pos_x_list = []
neg_y_data = []
neg_x_list = []

airfoil_coords = pd.read_excel("SLTpracticalcoordinates2.xlsx", usecols=[1, 2], skiprows= [0])
#print(airfoil_coords.shape)
#print(airfoil_coords.iloc[:, 1])

for i in range(airfoil_coords.shape[0]):
    if airfoil_coords.iloc[i, 1] >=0:
        pos_y_data.append(Cp_list[:, i])
        pos_x_list.append(airfoil_coords.iloc[i, 0])
    if airfoil_coords.iloc[i, 1] <=0:
        neg_y_data.append(Cp_list[:, i])
        neg_x_list.append(airfoil_coords.iloc[i, 0])

pos_x_list = pos_x_list[0:25:1]
neg_x_list = neg_x_list[2:27:1]
pos_y_data = np.transpose(pos_y_data)
pos_y_data = pos_y_data[:, 0:25:1]
neg_y_data = np.transpose(neg_y_data)
neg_y_data = neg_y_data[:, 2:27:1]

#print(np.shape(pos_y_data))
#print(np.shape(neg_y_data))
#print(np.shape(pos_x_list))
#print(np.shape(neg_x_list))

#print(pos_y_data[0])
#for i in range(np.shape(pos_y_data)[0]):
#    plt.plot(pos_x_list, pos_y_data[i])
#    plt.plot(neg_x_list, neg_y_data[i])
    
#fig, ax1 = plt.plot()
#print(pos_y_data[0])

#plt.plot(pos_x_list, pos_y_data[40], 'k-', zorder=1, lw=1,  color="orange", label="Top of the airfoil")
#plt.plot(neg_x_list, neg_y_data[40], 'k-', zorder=1, lw=1,  color="blue", label="Bottom of airfoil")
#plt.gca().invert_yaxis()
#plt.xlabel("X/C %")
#plt.ylabel("Cp")
#plt.legend()
#plt.grid(True, linestyle=':', alpha=0.6)
#plt.show()

#print(pos_x_list)
#print(neg_x_list)

plt.plot(pos_x_list, pos_y_data[7], 'k-', zorder=1, lw=1)
plt.scatter(pos_x_list, pos_y_data[7], edgecolors='k', color="orange", label="Top of the airfoil", zorder=2, s=20)
#plt.scatter(dragForces[43:], liftForces[43:], edgecolors='k',marker="s", color="blue", label=r"Airfoil Drag Polar - Pressure Taps (Decreasing $\alpha$)", zorder=2, s=20)
plt.plot(neg_x_list, neg_y_data[7], 'k-', zorder=1, lw=1)
plt.scatter(neg_x_list, neg_y_data[7], edgecolors='k', marker = "v", color="blue", label="Bottom of airfoil", zorder=2, s=20)
#plt.scatter(dragForcesWR[43:], liftForces[43:], edgecolors='k', marker = "P", color="yellow", label=r"Airfoil Drag Polar - Wake Rake (Decreasing $\alpha$)", zorder=2, s=20)
plt.gca().invert_yaxis()
plt.xlabel("X/C %")
plt.ylabel("Cp")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(pos_x_list, pos_y_data[10], 'k-', zorder=1, lw=1)
plt.scatter(pos_x_list, pos_y_data[10], edgecolors='k', color="orange", label="Top of the airfoil", zorder=2, s=20)
#plt.scatter(dragForces[43:], liftForces[43:], edgecolors='k',marker="s", color="blue", label=r"Airfoil Drag Polar - Pressure Taps (Decreasing $\alpha$)", zorder=2, s=20)
plt.plot(neg_x_list, neg_y_data[10], 'k-', zorder=1, lw=1)
plt.scatter(neg_x_list, neg_y_data[10], edgecolors='k', marker = "v", color="blue", label="Bottom of airfoil", zorder=2, s=20)
#plt.scatter(dragForcesWR[43:], liftForces[43:], edgecolors='k', marker = "P", color="yellow", label=r"Airfoil Drag Polar - Wake Rake (Decreasing $\alpha$)", zorder=2, s=20)
plt.gca().invert_yaxis()
plt.xlabel("X/C %")
plt.ylabel("Cp")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(pos_x_list, pos_y_data[20], 'k-', zorder=1, lw=1)
plt.scatter(pos_x_list, pos_y_data[20], edgecolors='k', color="orange", label="Top of the airfoil", zorder=2, s=20)
#plt.scatter(dragForces[43:], liftForces[43:], edgecolors='k',marker="s", color="blue", label=r"Airfoil Drag Polar - Pressure Taps (Decreasing $\alpha$)", zorder=2, s=20)
plt.plot(neg_x_list, neg_y_data[20], 'k-', zorder=1, lw=1)
plt.scatter(neg_x_list, neg_y_data[20], edgecolors='k', marker = "v", color="blue", label="Bottom of airfoil", zorder=2, s=20)
#plt.scatter(dragForcesWR[43:], liftForces[43:], edgecolors='k', marker = "P", color="yellow", label=r"Airfoil Drag Polar - Wake Rake (Decreasing $\alpha$)", zorder=2, s=20)
plt.gca().invert_yaxis()
plt.xlabel("X/C %")
plt.ylabel("Cp")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(pos_x_list, pos_y_data[33], 'k-', zorder=1, lw=1)
plt.scatter(pos_x_list, pos_y_data[33], edgecolors='k', color="orange", label="Top of the airfoil", zorder=2, s=20)
#plt.scatter(dragForces[43:], liftForces[43:], edgecolors='k',marker="s", color="blue", label=r"Airfoil Drag Polar - Pressure Taps (Decreasing $\alpha$)", zorder=2, s=20)
plt.plot(neg_x_list, neg_y_data[33], 'k-', zorder=1, lw=1)
plt.scatter(neg_x_list, neg_y_data[33], edgecolors='k', marker = "v", color="blue", label="Bottom of airfoil", zorder=2, s=20)
#plt.scatter(dragForcesWR[43:], liftForces[43:], edgecolors='k', marker = "P", color="yellow", label=r"Airfoil Drag Polar - Wake Rake (Decreasing $\alpha$)", zorder=2, s=20)
plt.gca().invert_yaxis()
plt.xlabel("X/C %")
plt.ylabel("Cp")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(pos_x_list, pos_y_data[40], 'k-', zorder=1, lw=1)
plt.scatter(pos_x_list, pos_y_data[40], edgecolors='k', color="orange", label="Top of the airfoil", zorder=2, s=20)
#plt.scatter(dragForces[43:], liftForces[43:], edgecolors='k',marker="s", color="blue", label=r"Airfoil Drag Polar - Pressure Taps (Decreasing $\alpha$)", zorder=2, s=20)
plt.plot(neg_x_list, neg_y_data[40], 'k-', zorder=1, lw=1)
plt.scatter(neg_x_list, neg_y_data[40], edgecolors='k', marker = "v", color="blue", label="Bottom of airfoil", zorder=2, s=20)
#plt.scatter(dragForcesWR[43:], liftForces[43:], edgecolors='k', marker = "P", color="yellow", label=r"Airfoil Drag Polar - Wake Rake (Decreasing $\alpha$)", zorder=2, s=20)
plt.gca().invert_yaxis()
plt.xlabel("X/C %")
plt.ylabel("Cp")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
