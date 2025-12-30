import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
b = 0.4 #m
c = 0.16 #m
S = b * c #m^2

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

    DynamicPressure = datarun[3]
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