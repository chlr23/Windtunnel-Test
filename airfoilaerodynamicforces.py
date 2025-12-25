import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open("raw_airfoil_20mps.txt") as data:
    lines = data.readlines()
    lines = lines[2:]
    dataruns = []
    for line in lines:
        datarun = line.split("\t")
        datarun[0] = int(datarun[0])
        for i in range(2, len(datarun)):
            datarun[i] = float(datarun[i])
        
        dataruns.append(datarun)


airfoilTapCoords = pd.read_excel("SLTpracticalcoordinates2.xlsx", usecols=[1, 2])

xCoordsPercent = airfoilTapCoords.iloc[:, 0].tolist()[1:]
yCoordsPercent = airfoilTapCoords.iloc[:, 1].tolist()[1:]

xCoords = []
for item in xCoordsPercent:
    xCoords.append(item / 100 * 0.16)

yCoords = []
for item in yCoordsPercent:
    yCoords.append(item / 100 * 0.16)

wakeRakeCoords = pd.read_excel("SLTpracticalcoordinates2.xlsx", usecols=[5, 8])

WRTotalCoordsmm = wakeRakeCoords.iloc[:, 0].tolist()[1:48]
WRStaticCoordsmm = wakeRakeCoords.iloc[:, 1].tolist()[1:13]

WRTotalCoords = []
for item in WRTotalCoordsmm:
    WRTotalCoords.append(item / 1000)

WRStaticCoords = []
for item in WRStaticCoordsmm:
    WRStaticCoords.append(item / 1000)

WRStaticCoordsNP = np.array(WRStaticCoords)
WRTotalCoordsNP = np.array(WRTotalCoords)


def calcNormalForce(xCoordsUpper, xCoordsLower, pUpper, pLower):
    forceUp = 0
    for i in range(len(xCoordsUpper)-1):
        forceUp = forceUp + (pUpper[i] + pUpper[i+1]) / 2 * (xCoordsUpper[i+1] - xCoordsUpper[i])

    forceDown = 0
    for j in range(len(xCoordsLower)-1):
        forceDown = forceDown + (pLower[j] + pLower[j+1]) / 2 * (xCoordsLower[j+1] - xCoordsLower[j])

    normalForce = -forceUp + forceDown
    return normalForce

def calcTangentForce(yCoordsFront, yCoordsBack, pFront, pBack):
    forceBack = 0
    for i in range(len(yCoordsFront)-1):
        forceBack = forceBack + (pFront[i] + pFront[i+1]) / 2 * (yCoordsFront[i+1] - yCoordsFront[i])

    forceForward = 0
    for j in range(len(yCoordsBack)-1):
        forceForward = forceForward + (pBack[j] + pBack[j+1]) / 2 * (yCoordsBack[j+1] - yCoordsBack[j])

    tangentForce = forceBack - forceForward
    return tangentForce

def calcMoment(xCoordsUpper, xCoordsLower, pUpper, pLower):
    momentUp = 0
    for i in range(len(xCoordsUpper)-1):
        momentUp = momentUp + (pUpper[i] + pUpper[i+1]) / 2 * (xCoordsUpper[i+1] - xCoordsUpper[i]) * (xCoordsUpper[i+1] + xCoordsUpper[i]) / 2

    momentDown = 0
    for j in range(len(xCoordsLower)-1):
        momentDown = momentDown + (pLower[j] + pLower[j+1]) / 2 * (xCoordsLower[j+1] - xCoordsLower[j]) * (xCoordsLower[j+1] + xCoordsLower[j]) / 2

    moment = momentUp - momentDown
    return moment

def calcLift(normalForce, tangentForce, aoa):
    return normalForce * np.cos(aoa) - tangentForce * np.sin(aoa)

def calcDrag(normalForce, tangentForce, aoa):
    return tangentForce * np.cos(aoa) + normalForce * np.sin(aoa)

def calcCP(liftForce, moment):
    return -moment / liftForce

#def calcDragWakeRake(WKTotalCoords, WKStaticCoords, WKpTotal, WKpStatic, vFS, rho):

#calculating normal forces
xCoordsUpper = xCoords[0:25]
xCoordsLower = xCoords[26:49]

yCoordsFront1 = yCoords[25:36]
yCoordsFront2 = yCoords[0:12]
yCoordsFront1 = yCoordsFront1[::-1]
yCoordsFront = yCoordsFront1 + yCoordsFront2

yCoordsBack1 = yCoords[36:49]
yCoordsBack2 = yCoords[12:25]
yCoordsBack2 = yCoordsBack2[::-1]
yCoordsBack = yCoordsBack1 + yCoordsBack2

liftForces = []
dragForces =[]
moments = []
aoas = []
cps = []
aoasdeg = []
quartermoments = []

for datarun in dataruns:
    pUpper = datarun[8:33]
    pLower = datarun[34:57]

    pFront1 = datarun[33:44]
    pFront1 = pFront1[::-1]
    pFront2 = datarun[8:20]
    pFront = pFront1 + pFront2

    pBack1 = datarun[44:57]
    pBack2 = datarun[20:33]
    pBack2 = pBack2[::-1]
    pBack = pBack1 + pBack2

    aoa = datarun[2] * np.pi / 180
    aoadeg = datarun[2]
    aoas.append(aoa)
    aoasdeg.append(aoadeg)

    normalForce = calcNormalForce(xCoordsUpper, xCoordsLower, pUpper, pLower)
    tangentForce = calcTangentForce(yCoordsFront, yCoordsBack, pFront, pBack)

    moment = calcMoment(xCoordsUpper, xCoordsLower, pUpper, pLower)
    moments.append(moment)

    liftForce = calcLift(normalForce, tangentForce, aoa)
    dragForce = calcDrag(normalForce, tangentForce, aoa)
    liftForces.append(liftForce)
    dragForces.append(dragForce)
    quartermoments.append(moment + liftForce * 0.25 * 0.16)

    cp = calcCP(liftForce, moment)
    cps.append(cp)

plt.plot(aoasdeg, liftForces, 'k-', zorder=1, lw=1)
plt.scatter(aoasdeg, liftForces, edgecolors='k', color="r", label="Measured Lift Force on Airfoil", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}^{\circ}\right)$')
plt.ylabel(r'L $\left(\frac{N}{m}\right)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(aoasdeg, dragForces, 'k-', zorder=1, lw=1)
plt.scatter(aoasdeg, dragForces, edgecolors='k', color="blue", label="Measured Drag Force on Airfoil", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}^{\circ}\right)$')
plt.ylabel(r'D $\left(\frac{N}{m}\right)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(aoasdeg, moments, 'k-', zorder=1, lw=1)
plt.scatter(aoasdeg, moments, edgecolors='k', color="orange", label="Measured Leading Edge Moment on Airfoil", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}^{\circ}\right)$')
plt.ylabel(r'$\text{M}_{LE}\;\left(\frac{Nm}{m}\right)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(aoasdeg, quartermoments, 'k-', zorder=1, lw=1)
plt.scatter(aoasdeg, quartermoments, edgecolors='k', color="orange", label="Measured Quarter-chord Moment on Airfoil", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}^{\circ}\right)$')
plt.ylabel(r'$\text{M}_{c/4}\;\left(\frac{Nm}{m}\right)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(dragForces, liftForces, 'k-', zorder=1, lw=1)
plt.scatter(dragForces, liftForces, edgecolors='k', color="blue", label="Measured Airfoil Drag Polar", zorder=2, s=20)
plt.xlabel(r'D $\left(\frac{N}{m}\right)$')
plt.ylabel(r'L $\left(\frac{N}{m}\right)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(aoasdeg, cps, 'k-', zorder=1, lw=1)
plt.scatter(aoasdeg, cps, edgecolors='k', color="green", label="Measured Location of Center of Pressure", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}^{\circ}\right)$')
plt.ylabel(r'$\text{c}_{p}\;\left(m\right)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

#drag from wake rake measurements

def wake_profile(rho,static_pos,static_pressure,total_pos,total_pressure,y):
    u_wake = np.sqrt((np.interp(y, total_pos, total_pressure) - np.interp(y,static_pos,static_pressure))*2 /rho)
    return u_wake

uFS = 20.233989156212825
dragForcesWR = []

for datarun in dataruns:
    aoa = datarun[2]
    rho = datarun[7]
    pFS = datarun[104]
    pStatic = np.array(datarun[105:117])
    pTotal = np.array(datarun[57:104])

    dragForceWR = 0
    for i in range(len(WRTotalCoords)-1):
        u_wake1 = wake_profile(rho, WRStaticCoordsNP, pStatic, WRTotalCoordsNP, pTotal, WRTotalCoords[i])
        u_wake2 = wake_profile(rho, WRStaticCoordsNP, pStatic, WRTotalCoordsNP, pTotal, WRTotalCoords[i+1])
        dragForceWR = dragForceWR + ((uFS - u_wake2) * u_wake2 + (uFS - u_wake1) * u_wake1) / 2 * (WRTotalCoords[i+1] - WRTotalCoords[i]) * rho
        deltap1 = pFS - pTotal[i]
        deltap2 = pFS - pTotal[i+1]
        dragForceWR = dragForceWR + (deltap2 + deltap1) / 2 * (WRTotalCoords[i+1] - WRTotalCoords[i])

    dragForcesWR.append(dragForceWR)

plt.plot(aoasdeg, dragForces, 'k-', zorder=1, lw=1)
plt.scatter(aoasdeg, dragForces, edgecolors='k', color="blue", label="Drag Force on Airfoil from Pressure Taps", zorder=2, s=20)
plt.plot(aoasdeg, dragForcesWR, 'k-', zorder=1, lw=1)
plt.scatter(aoasdeg, dragForcesWR, edgecolors='k', marker = "v", color="yellow", label="Drag Force on Airfoil from Wake Rake", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}^{\circ}\right)$')
plt.ylabel(r'D $\frac{N}{m}$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

plt.plot(dragForces, liftForces, 'k-', zorder=1, lw=1)
plt.scatter(dragForces, liftForces, edgecolors='k', color="blue", label="Airfoil Drag Polar from Pressure Taps", zorder=2, s=20)
plt.plot(dragForcesWR, liftForces, 'k-', zorder=1, lw=1)
plt.scatter(dragForcesWR, liftForces, edgecolors='k', marker = "v", color="yellow", label="Airfoil Drag Polar from Wake Rake", zorder=2, s=20)
plt.xlabel(r'D $\frac{N}{m}$')
plt.ylabel(r'L $\frac{N}{m}$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()