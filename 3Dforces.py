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

    DynamicPressure = 0.211804 + 1.928442 * datarun[3] + 1.879374 * 10 ** (-4) * datarun[3] ** 2
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
plt.xlabel(r'$\alpha\;\left(\text{deg}\right)$')
plt.ylabel(r'L $\left(\text{N}\right)$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(df_avg["AoA_deg"], df_avg["Lift_per_span"], 'k-', lw=1)
plt.scatter(df_avg["AoA_deg"], df_avg["Lift_per_span"], edgecolors='k', color="r", label="Measured Lift Force per Span on Wing", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}\right)$')
plt.ylabel(r"$L'\;(\mathrm{N/m})$")
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(df_avg["AoA_deg"], df_avg["Drag"], 'k-', zorder=1, lw=1)
plt.scatter(df_avg["AoA_deg"], df_avg["Drag"], edgecolors='k', color="r", label="Measured Drag Force on Wing", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}\right)$')
plt.ylabel(r'D $\left(\text{N}\right)$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(df_avg["AoA_deg"], df_avg["Drag_per_span"], 'k-', lw=1)
plt.scatter(df_avg["AoA_deg"], df_avg["Drag_per_span"], edgecolors='k', color="r", label="Measured Drag Force per Span on Wing", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}\right)$')
plt.ylabel(r"$D'\;(\mathrm{N/m})$")
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(df_avg["AoA_deg"], df_avg["CL"], 'k-', lw=1)
plt.scatter(df_avg["AoA_deg"], df_avg["CL"], edgecolors='k', color="r", label="Calculated CL of Wing", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}\right)$')
plt.ylabel(r'$C_{L}$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(df_avg["AoA_deg"], df_avg["CD"], 'k-', lw=1)
plt.scatter(df_avg["AoA_deg"], df_avg["CD"], edgecolors='k', color="r", label="Calculated CD of Wing", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}\right)$')
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




with open("aero_coefficients_2d.json") as f:
    data_2d = json.load(f)

aoa_2d_upward = pd.Series(data_2d["upward_sweep"]["AoA_deg"])
cl_2d_upward  = pd.Series(data_2d["upward_sweep"]["CL_2D"])
cd_2d_upward  = pd.Series(data_2d["upward_sweep"]["CD_2D_pressure"])


CDi_from_tau = df_avg["CL"]**2 / (np.pi * A) * (1 + tau)


df_overlap = df_avg[df_avg["AoA_deg"] <= max(aoa_2d_upward)].copy()
CDi_from_graph = df_overlap["CD"] - cd_2d_upward

#print(f'Length of 2d= {len(cl_2d_upward)} \nLength of 3d= {len(CDi_from_graph)}')

plt.plot(aoa_2d_upward, CDi_from_graph, 'k-', lw=1.5)
plt.scatter(aoa_2d_upward, CDi_from_graph, color="blue", edgecolors='k', 
           label=r"$C_{D_{i}}$ (From Finite Wing)", zorder=2, s=40)

plt.plot(df_avg["AoA_deg"], CDi_from_tau, 'k-', lw=1.5)
plt.scatter(df_avg["AoA_deg"], CDi_from_tau, color="red", marker="^", edgecolors='k', 
           label=r"$C_{D_{i}}$ (From Tau Estimation)", zorder=2, s=40)

plt.xlabel(r'$\alpha\;(\text{deg})$')
plt.ylabel(r'$C_{D_{i}}$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

plt.plot(df_avg["AoA_deg"], df_avg["CL"], 'k-', lw=1.5)
plt.scatter(df_avg["AoA_deg"], df_avg["CL"], color="blue", edgecolors='k', 
           label=r"$C_{L}$ (3D)", zorder=2, s=40)

plt.plot(aoa_2d_upward, cl_2d_upward, 'k-', lw=1.5)
plt.scatter(aoa_2d_upward, cl_2d_upward, color="red", marker="^", edgecolors='k', 
           label=r"$C_{l}$ (2D)", zorder=2, s=40)

plt.xlabel(r'$\alpha\;(\text{deg})$')
plt.ylabel(r'$C_L\;\text{or}\;C_l$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# Plot CD comparison (3D vs 2D upward sweep)

plt.plot(df_avg["AoA_deg"], df_avg["CD"], 'k-', lw=1.5)
plt.scatter(df_avg["AoA_deg"], df_avg["CD"], color="blue", edgecolors='k', 
           label=r"$C_{D}$ (3D)", zorder=2, s=40)

plt.plot(aoa_2d_upward, cd_2d_upward, 'k-', lw=1.5)
plt.scatter(aoa_2d_upward, cd_2d_upward, color="red", marker="^", edgecolors='k', 
           label=r"$C_{d}$ (2D)", zorder=2, s=40)

plt.xlabel(r'$\alpha\;(\text{deg})$')
plt.ylabel(r'$C_D\;\text{or}\;C_d$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()


#Data stuff 2D
Cl_max_idx = cl_2d_upward.idxmax()
Cl_max = cl_2d_upward.max()
AoA_stall_2d = aoa_2d_upward.loc[Cl_max_idx]
Cd_min = cd_2d_upward.min()
Cd_min_idx = cd_2d_upward.idxmin()
AoA_minD_2d = aoa_2d_upward.loc[Cd_min_idx]
Cl0_2d = cl_2d_upward.loc[aoa_2d_upward == 0].values[0]

print("\n2D stuff:\n")
print(f"Cl_0 = {Cl0_2d:.4f}")
print(f"Cl_max = {Cl_max:.4f}")
print(f"Stall Angle (2D) = {AoA_stall_2d:.1f}")
print(f"Cd_min = {Cd_min:.4f}")
print(f"Minimal Drag Angle (2D) = {AoA_minD_2d:.4f}")

# Data BS 3D
CL_max_idx = df_avg["CL"].idxmax()
CL_max = df_avg["CL"].max()
AoA_stall = df_avg.loc[CL_max_idx, "AoA_deg"]
CD_min = df_avg["CD"].min()
CD_min_idx = df_avg["CD"].idxmin()
AoA_minD = df_avg.loc[CD_min_idx, "AoA_deg"]

print("\n3D stuff:\n")
print(f"CL_0 = {CL0:.4f}")
print(f"CL_max = {CL_max:.4f}")
print(f"Stall Angle = {AoA_stall:.1f}")
print(f"CD_min = {CD_min:.4f}")
print(f"Minimal Drag Angle = {AoA_minD:.4f}")

print("\nComparison stuff:\n")
print(f"Change in CL_0 (2D-3D) = {Cl0_2d-CL0:.4f}")
print(f"Change Stall angle (2D-3D) = {AoA_stall_2d - AoA_stall:.1f}")
print(f"Change in CL_max (2D - 3D) = {Cl_max-CL_max:.4f}")
