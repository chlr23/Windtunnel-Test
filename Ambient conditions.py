#Ambient conditions and Reference speed

# Imports
import measuredData as mData
import math

# Constants
cMassAir = 29 * 10 ** (-3)  # kg / mol
cGas = 8.314 # J/(mol * K)
cVisc = 1.716 * 10 ** (-5) # kg / (m * s)
cTemp = 273.15 # K
cSutherland = 110.4 # K
cChord = 0.2 # m

# Measurements ! From airfoil average
#mTemp = (mData.airfoilData["T"].mean() + mData.wingData["T"].mean())/2 + cTemp # K
mTemp = (mData.airfoilData["T"].mean() ) + cTemp # K
mPres = (mData.airfoilData["P_bar"].mean() ) * 100 # Pa 
#mPresDiff = (mData.airfoilData["Delta_Pb"].mean() + mData.wingData["Delta_Pb"].mean())/2 # Pa total pres in settling chamber - static pressure in contraction
mPresDiff = (mData.airfoilData["Delta_Pb"].mean()) # Pa total pres in settling chamber - static pressure in contraction
mPresTotal = (mData.airfoilData["P097"]).mean()

print(f'Temp {mTemp}, mPres {mPres}, mPresDiff {mPresDiff}, total pressure {mPresTotal}')

# Calculations

# Density
dens = cMassAir * mPres / (cGas * mTemp)
print(f'density: {dens}')

# Viscosity
visc = cVisc * (mTemp/cTemp) ** (3/2) * (cTemp + cSutherland) / (mTemp +cSutherland)
print(f'viscosity: {visc}')

# Dynamic pressure
presDyn = 0.211804 + 1.928442 * mPresDiff + 1.879374 * 10 ** (-4) * mPresDiff ** 2
print(f'dynamic pressure: {presDyn}')

# Reference static pressure
presStat = mPresTotal - presDyn
print(f'ref static pressure {presStat}')

# Reference free stream
velFreeStream = ((mPresTotal - presStat) * 2 / dens) ** 0.5
print(f'free stream velocity: {velFreeStream}')

#Reynolds Number
reynolds = dens * velFreeStream * cChord / visc
print(f"Reynold's number {reynolds}")

