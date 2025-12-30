import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json

with open("cl_alpha_2d.json") as f:
    data = json.load(f)

a0 = data["cl_alpha_2d"]

# Read 3D Dataruns
with open("raw_wing_20mps.txt") as data:
    lines = data.readlines()
    lines = lines[2:]
    dataruns = []
    for line in lines:
        datarun = line.split("\t")
        datarun[0] = int(datarun[0])
        for i in range(2, len(datarun)):
            datarun[i] = float(datarun[i])
        
        dataruns.append(datarun)

# Functions for Calculations
def CalcCoef(Force,DyPres,S):
    CL = Force / (DyPres * S)
    return CL

# Constants 
b = 0.4169 #m
c = 0.16 #m
S = 0.066146 #m^2
A = b / c

# Lists needed for plots 
LiftForces = []
DragForces = []

LiftForcesPerSpan = []
DragForcesPerSpan = []

DynamicPressures = []

AoAs = []
AoAdegs = []

CLs = []
CDs = []

for datarun in dataruns:
    LiftForce = datarun[7]
    LiftForces.append(LiftForce)

    DragForce = datarun[6]
    DragForces.append(DragForce)

    LiftForcePerSpan = LiftForce / b
    LiftForcesPerSpan.append(LiftForcePerSpan)

    DragForcePerSpan = DragForce / b
    DragForcesPerSpan.append(DragForcePerSpan)

    DynamicPressure = 0.5 * datarun[10] * 20.23**2
    DynamicPressures.append(DynamicPressure)

    AoA = datarun[2] * np.pi / 180
    AoAs.append(AoA)

    AoAdeg = datarun[2]
    AoAdegs.append(AoAdeg)

    CL = CalcCoef(LiftForce,DynamicPressure,S)
    CLs.append(CL)

    CD = CalcCoef(DragForce,DynamicPressure,S)
    CDs.append(CD)

df = pd.DataFrame({
    "AoA_deg": AoAdegs,
    "AoA_rad": AoAs,
    "CL": CLs,
    "CD": CDs,
    "Lift": LiftForces,
    "Drag": DragForces,
    "Lift_per_span": LiftForcesPerSpan,
    "Drag_per_span": DragForcesPerSpan,
    "q": DynamicPressures
})

df_avg = df.groupby("AoA_deg", as_index=False).mean()



# Plots
plt.plot(df_avg["AoA_deg"], df_avg["Lift"], 'k-', zorder=1, lw=1)
plt.scatter(df_avg["AoA_deg"], df_avg["Lift"], edgecolors='k', color="r", label="Measured Lift Force on Wing", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}^{\circ}\right)$')
plt.ylabel(r'L $\left(\text{N}\right)$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(df_avg["AoA_deg"], df_avg["Lift_per_span"], 'k-', lw=1)
plt.scatter(df_avg["AoA_deg"], df_avg["Lift_per_span"], edgecolors='k', color="r", label="Measured Lift Force per Span on Wing", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}^{\circ}\right)$')
plt.ylabel(r"$L'\;(\mathrm{N/m})$")
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(df_avg["AoA_deg"], df_avg["Drag"], 'k-', zorder=1, lw=1)
plt.scatter(df_avg["AoA_deg"], df_avg["Drag"], edgecolors='k', color="r", label="Measured Drag Force on Wing", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}^{\circ}\right)$')
plt.ylabel(r'D $\left(\text{N}\right)$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(df_avg["AoA_deg"], df_avg["Drag_per_span"], 'k-', lw=1)
plt.scatter(df_avg["AoA_deg"], df_avg["Drag_per_span"], edgecolors='k', color="r", label="Measured Drag Force per Span on Wing", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}^{\circ}\right)$')
plt.ylabel(r"$D'\;(\mathrm{N/m})$")
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(df_avg["AoA_deg"], df_avg["CL"], 'k-', lw=1)
plt.scatter(df_avg["AoA_deg"], df_avg["CL"], edgecolors='k', color="r", label="Calculated CL of Wing", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}^{\circ}\right)$')
plt.ylabel(r'$C_{L}$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(df_avg["AoA_deg"], df_avg["CD"], 'k-', lw=1)
plt.scatter(df_avg["AoA_deg"], df_avg["CD"], edgecolors='k', color="r", label="Calculated CD of Wing", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}^{\circ}\right)$')
plt.ylabel(r'$C_{D}$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(df_avg["CD"], df_avg["CL"], 'k-', lw=1)
plt.scatter(df_avg["CD"], df_avg["CL"], edgecolors='k', color="r", label="Wing Drag Polar", zorder=2, s=20)
plt.xlabel(r'$C_{D}$')
plt.ylabel(r'$C_{L}$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()


aoa_array = np.array(df_avg["AoA_deg"])
cl_array = np.array(df_avg["CL"])

# Select data between -5 and 5 degrees
mask = (aoa_array >= -5) & (aoa_array <= 5)
aoa_rad = np.deg2rad(aoa_array[mask])  # AoA in radians
cl_sel = cl_array[mask]

# Linear fit to get CL_alpha (slope)
coeffs = np.polyfit(aoa_rad, cl_sel, 1)
CL_alpha = coeffs[0]  # slope in per rad
CL0 = coeffs[1]       # CL at 0 AoA

a = CL_alpha
tau = ((a0 / a) - 1) * ((np.pi * A) / a0) - 1


print(f"3D Wing CL_alpha = {CL_alpha:.4f} 1/rad")
print(f"2D Airfoil Cl_alpha= {a0:.4f}")
print(f"Aspect Ratio: {A:.4f}")
print(f"Tau = {tau:.4f}")


# Compute induced drag coefficient for each AoA
CD_induced = cl_array**2 / (np.pi * A * (1 + tau))

# Fraction of induced drag relative to total CD
cd_array = np.array(df_avg["CD"])
induced_fraction = CD_induced / cd_array

# Plot induced drag fraction
plt.plot(aoa_array, induced_fraction, 'k-', lw=1)
plt.scatter(aoa_array, induced_fraction, edgecolors='k', color="orange", zorder=2, s=20)
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'Fraction of Induced Drag $C_{D,i}/C_D$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()


# Find the index where AoA = 0 deg
idx_0 = np.argmin(np.abs(aoa_array - 0))

# CL and CD at AoA = 0
CL_0 = cl_array[idx_0]
CD_0 = cd_array[idx_0]

# Induced drag coefficient at AoA = 0
CDi_0 = CL_0**2 / (np.pi * A * (1 + tau))

# Fraction of total drag that is induced
induced_fraction_0 = CDi_0 / CD_0

print(f"At AoA = 0°:")
print(f"CL = {CL_0:.4f}, CD = {CD_0:.4f}")
print(f"Induced Drag CDi = {CDi_0:.4f}")
print(f"Fraction of total drag due to lift = {induced_fraction_0:.2%}")