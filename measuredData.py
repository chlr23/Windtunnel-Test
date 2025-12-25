import pandas as pd

wingData = pd.read_csv('raw_wing_20mps.txt', sep = '\s+',index_col= 0, skiprows = [1] )
#print(wingData["Rho"].mean())

airfoilData = pd.read_csv('raw_airfoil_20mps.txt', sep='\s+', index_col = 0, skiprows=[1])
#print(airfoilData["P097"].mean())
