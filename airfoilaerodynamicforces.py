import pandas as pd

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

xCoords = airfoilTapCoords.iloc[:, 0].tolist()[1:]
yCoords = airfoilTapCoords.iloc[:, 1].tolist()[1:]

xCoords = xCoords / 100 * 0.16 #turn coordinates from percent to location in m
yCoords = yCoords / 100 * 0.16 

def calcNormalForce(xCoordsUpper, xCoordsLower, pUpper, pLower):
    forceUp = 0
    for i in range(len(xCoordsUpper)-1):
        forceUp = forceUp + (pUpper[i] + pUpper[i+1]) / 2 * (xCoordsUpper[i+1] - xCoordsUpper[i])

    forceDown = 0
    for j in range(len(xCoordsLower)-1):
        forceDown = forceDown + (pLower[j] + pLower[j+1]) / 2 * (xCoordsLower[j+1] - xCoordsLower[i])

    normalForce = -forceUp + forceDown
    return normalForce
