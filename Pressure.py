import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


v = 20
x_pos = np.linspace(0, 1, 113)
alpha_list = np.append(np.arange(-5, 16, 0.5) , np.arange(16, 9.5, -0.5)) 
#print(alpha_list)


data = pd.read_csv("raw_airfoil_20mps.txt", skiprows= [ 1], sep = "\t")
#print(data)
#data = data.drop(columns = ["Run_nr", "Time", "Alpha", "Delta_Pb", "P_bar", "T", "rpm", "rho"])
rho = data[data.columns[[7]]].to_numpy()
#print(rho)
p_inf = data[data.columns[[4]]].to_numpy()
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
Cp_list = 1000 * Cp(data_list[0], p_inf[0], rho[0], v)

for i in range(1, 55):
    Cp1 = 1000 * Cp(data_list[i], p_inf[i], rho[i], v)
    Cp_list = np.vstack((Cp_list, Cp1))

print(Cp_list.shape)



plt.plot(x_pos, Cp_list[50])
plt.show()


