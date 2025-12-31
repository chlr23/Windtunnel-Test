import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load data
# ----------------------------
with open("raw_airfoil_20mps.txt") as data:
    lines = data.readlines()[2:]
    dataruns = []
    for line in lines:
        datarun = line.strip().split("\t")
        datarun[0] = int(datarun[0])
        for i in range(2, len(datarun)):
            datarun[i] = float(datarun[i])
        dataruns.append(datarun)

airfoilTapCoords = pd.read_excel("SLTpracticalcoordinates2.xlsx", usecols=[1, 2])
xCoordsPercent = airfoilTapCoords.iloc[:, 0].tolist()[1:]
yCoordsPercent = airfoilTapCoords.iloc[:, 1].tolist()[1:]

# Convert to meters
c = 0.16  # chord length in meters
xCoords = [item / 100 * c for item in xCoordsPercent]
yCoords = [item / 100 * c for item in yCoordsPercent]

wakeRakeCoords = pd.read_excel("SLTpracticalcoordinates2.xlsx", usecols=[5, 8])
WRTotalCoordsmm = wakeRakeCoords.iloc[:, 0].tolist()[1:48]
WRStaticCoordsmm = wakeRakeCoords.iloc[:, 1].tolist()[1:13]

WRTotalCoords = np.array([item / 1000 for item in WRTotalCoordsmm])
WRStaticCoords = np.array([item / 1000 for item in WRStaticCoordsmm])

# ----------------------------
# Define force and moment functions
# ----------------------------
def calcNormalForce(xCoordsUpper, xCoordsLower, pUpper, pLower):
    forceUp = sum((pUpper[i] + pUpper[i+1])/2 * (xCoordsUpper[i+1]-xCoordsUpper[i])
                  for i in range(len(xCoordsUpper)-1))
    forceDown = sum((pLower[i] + pLower[i+1])/2 * (xCoordsLower[i+1]-xCoordsLower[i])
                    for i in range(len(xCoordsLower)-1))
    return -forceUp + forceDown

def calcTangentForce(yCoordsFront, yCoordsBack, pFront, pBack):
    forceBack = sum((pFront[i] + pFront[i+1])/2 * (yCoordsFront[i+1]-yCoordsFront[i])
                    for i in range(len(yCoordsFront)-1))
    forceForward = sum((pBack[i] + pBack[i+1])/2 * (yCoordsBack[i+1]-yCoordsBack[i])
                       for i in range(len(yCoordsBack)-1))
    return forceBack - forceForward

def calcMoment(xCoordsUpper, xCoordsLower, pUpper, pLower):
    momentUp = sum((pUpper[i] + pUpper[i+1])/2 * (xCoordsUpper[i+1]-xCoordsUpper[i]) * 
                   (xCoordsUpper[i+1]+xCoordsUpper[i])/2
                   for i in range(len(xCoordsUpper)-1))
    momentDown = sum((pLower[i] + pLower[i+1])/2 * (xCoordsLower[i+1]-xCoordsLower[i]) * 
                     (xCoordsLower[i+1]+xCoordsLower[i])/2
                     for i in range(len(xCoordsLower)-1))
    return momentUp - momentDown

def calcLift(normalForce, tangentForce, aoa):
    return normalForce * np.cos(aoa) - tangentForce * np.sin(aoa)

def calcDrag(normalForce, tangentForce, aoa):
    return tangentForce * np.cos(aoa) + normalForce * np.sin(aoa)

def calcCP(liftForce, moment):
    return -moment / liftForce

def force_to_cl(lift_force_per_span, q_inf, c):
    return lift_force_per_span / (q_inf * c)

def force_to_cd(drag_force_per_span, q_inf, c):
    return drag_force_per_span / (q_inf * c)

def moment_to_cm(moment_per_span, q_inf, c):
    return moment_per_span / (q_inf * c**2)

def quarter_moment_to_cm_quarter(quarter_moment_per_span, q_inf, c):
    return quarter_moment_per_span / (q_inf * c**2)

# ----------------------------
# Wake profile and drag from wake rake
# ----------------------------
def wake_profile(rho, static_pos, static_pressure, total_pos, total_pressure, y):
    u_wake = np.sqrt(2 * (np.interp(y, total_pos, total_pressure) - np.interp(y, static_pos, static_pressure)) / rho)
    return u_wake

uFS = 20.233989156212825  # free-stream velocity

# ----------------------------
# Prepare coordinates for force calculations
# ----------------------------
xCoordsUpper = xCoords[0:25]
xCoordsLower = xCoords[26:49]

yCoordsFront1 = yCoords[25:36][::-1]
yCoordsFront2 = yCoords[0:12]
yCoordsFront = yCoordsFront1 + yCoordsFront2

yCoordsBack1 = yCoords[36:49]
yCoordsBack2 = yCoords[12:25][::-1]
yCoordsBack = yCoordsBack1 + yCoordsBack2

# ----------------------------
# Initialize lists
# ----------------------------
liftForces = []
dragForces = []
moments = []
aoas = []
aoasdeg = []
cps = []
quartermoments = []
cl_values = []
cd_values_pressure = []
cd_values_wake = []
cm_le_values = []
cm_quarter_values = []
dragForcesWR = []

# ----------------------------
# Main loop over dataruns
# ----------------------------
for datarun in dataruns:
    # Extract pressures
    pUpper = datarun[8:33]
    pLower = datarun[34:57]

    pFront = datarun[33:44][::-1] + datarun[8:20]
    pBack = datarun[44:57] + datarun[20:33][::-1]

    deltaPb = datarun[3]

    # AoA
    aoa = datarun[2] * np.pi / 180
    aoadeg = datarun[2]
    aoas.append(aoa)
    aoasdeg.append(aoadeg)

    # Forces and moments from pressure taps
    normalForce = calcNormalForce(xCoordsUpper, xCoordsLower, pUpper, pLower)
    tangentForce = calcTangentForce(yCoordsFront, yCoordsBack, pFront, pBack)
    moment = calcMoment(xCoordsUpper, xCoordsLower, pUpper, pLower)
    liftForce = calcLift(normalForce, tangentForce, aoa)
    dragForce = calcDrag(normalForce, tangentForce, aoa)
    quarterMoment = moment + liftForce * 0.25 * c
    cp = calcCP(liftForce, moment)

    # Append results
    liftForces.append(liftForce)
    dragForces.append(dragForce)
    moments.append(moment)
    quartermoments.append(quarterMoment)
    cps.append(cp)

    # Dynamic pressure
    rho = datarun[7]
    q_inf = 0.211804 + 1.928442 * deltaPb + 1.879374 * 10 ** (-4) * deltaPb ** 2

    # Coefficients from pressure taps
    cl_values.append(force_to_cl(liftForce, q_inf, c))
    cd_values_pressure.append(force_to_cd(dragForce, q_inf, c))
    cm_le_values.append(moment_to_cm(moment, q_inf, c))
    cm_quarter_values.append(quarter_moment_to_cm_quarter(quarterMoment, q_inf, c))

    # ----------------------------
    # Drag from wake rake
    # ----------------------------
    pFS = datarun[104]
    pStatic = np.array(datarun[105:117])
    pTotal = np.array(datarun[57:104])
    dragWR = 0
    for i in range(len(WRTotalCoords)-1):
        u1 = wake_profile(rho, WRStaticCoords, pStatic, WRTotalCoords, pTotal, WRTotalCoords[i])
        u2 = wake_profile(rho, WRStaticCoords, pStatic, WRTotalCoords, pTotal, WRTotalCoords[i+1])
        dragWR += 0.5 * ((uFS-u1)*u1 + (uFS-u2)*u2) * (WRTotalCoords[i+1]-WRTotalCoords[i]) * rho
        deltaP1 = pFS - pTotal[i]
        deltaP2 = pFS - pTotal[i+1]
        dragWR += 0.5 * (deltaP1 + deltaP2) * (WRTotalCoords[i+1]-WRTotalCoords[i])
    dragForcesWR.append(dragWR)
    cd_values_wake.append(force_to_cd(dragWR, q_inf, c))

# ----------------------------
# Plotting
# ----------------------------
def plot_force_vs_aoa(ydata, ylabel, label, color='r'):
    plt.plot(aoasdeg, ydata, 'k-', lw=1, zorder=1)
    plt.scatter(aoasdeg, ydata, color=color, edgecolors='k', label=label, zorder=2, s=20)
    plt.xlabel(r'$\alpha\;\left(\mathrm{deg}\right)$')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

plot_force_vs_aoa(liftForces, r'L $\left(\frac{N}{m}\right)$', "Measured Lift Force on Airfoil", "r")
plot_force_vs_aoa(dragForces, r'D $\left(\frac{N}{m}\right)$', "Measured Drag Force on Airfoil", "b")
plot_force_vs_aoa(moments, r'M$_{LE}$ $\left(\frac{Nm}{m}\right)$', "Measured Leading Edge Moment", "orange")
plot_force_vs_aoa(quartermoments, r'M$_{c/4}$ $\left(\frac{Nm}{m}\right)$', "Measured Quarter-chord Moment", "orange")
plot_force_vs_aoa(cps, r'c$_p$ [m]', "Measured Center of Pressure", "green")

# Drag comparison from wake rake
plt.plot(aoasdeg, dragForces, 'k-', lw=1, zorder=1)
plt.scatter(aoasdeg, dragForces, edgecolors='k', color="blue", label="Pressure Tap Drag", zorder=2, s=20)
plt.plot(aoasdeg, dragForcesWR, 'k-', lw=1, zorder=1)
plt.scatter(aoasdeg, dragForcesWR, edgecolors='k', marker="v", color="yellow", label="Wake Rake Drag", zorder=2, s=20)
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'D [N/m]')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# Lift coefficient
plt.plot(aoasdeg, cl_values, 'k-', lw=1, zorder=1)
plt.scatter(aoasdeg, cl_values, color="r", edgecolors='k', label=r"$C_l$", zorder=2, s=20)
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'$C_l$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# Drag coefficient comparison
plt.plot(aoasdeg, cd_values_pressure, 'k-', lw=1, zorder=1)
plt.scatter(aoasdeg, cd_values_pressure, color="blue", edgecolors='k', label=r"$C_d$ from pressure taps", zorder=2, s=20)
plt.plot(aoasdeg, cd_values_wake, 'k-', lw=1, zorder=1)
plt.scatter(aoasdeg, cd_values_wake, color="yellow", marker="v", edgecolors='k', label=r"$C_d$ from wake rake", zorder=2, s=20)
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'$C_d$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# Drag polar
plt.plot(cd_values_pressure, cl_values, 'k-', lw=1, zorder=1)
plt.scatter(cd_values_pressure, cl_values, color="blue", edgecolors='k', label="Pressure taps", zorder=2, s=20)
plt.plot(cd_values_wake, cl_values, 'k-', lw=1, zorder=1)
plt.scatter(cd_values_wake, cl_values, color="yellow", marker="v", edgecolors='k', label="Wake rake", zorder=2, s=20)
plt.xlabel(r'$C_d$')
plt.ylabel(r'$C_l$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# Moment coefficients
plt.plot(aoasdeg, cm_le_values, 'k-', lw=1, zorder=1)
plt.scatter(aoasdeg, cm_le_values, color="orange", edgecolors='k', label=r"$C_{m,LE}$", zorder=2, s=20)
plt.plot(aoasdeg, cm_quarter_values, 'k-', lw=1, zorder=1)
plt.scatter(aoasdeg, cm_quarter_values, color="green", edgecolors='k', label=r"$C_{m,c/4}$", zorder=2, s=20)
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'$C_m$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

aoas_array = np.array(aoasdeg)
cl_array = np.array(cl_values)

mask = (aoas_array >= -5) & (aoas_array <= 5)
aoas_rad = np.deg2rad(aoas_array[mask])  # convert to radians
cl_sel = cl_array[mask]

# Linear fit (Cl = Cl_alpha * alpha + Cl0)
coeffs = np.polyfit(aoas_rad, cl_sel, 1)  # 1 = linear
cl_alpha = coeffs[0]  # slope in 1/rad
cl0 = coeffs[1]       # intercept

print(f"C_l_alpha = {cl_alpha:.4f} per rad")
print(f"C_l at 0 AoA = {cl0:.4f}")

import json

cl_alpha_2d = cl_alpha

with open("cl_alpha_2d.json", "w") as f:
    json.dump({"cl_alpha_2d": cl_alpha_2d}, f)