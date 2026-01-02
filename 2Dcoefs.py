import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

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
# DataFrane
# ----------------------------

df = pd.DataFrame({
    "AoA_deg": aoasdeg,
    "AoA_rad": aoas,
    "Lift": liftForces,
    "Drag": dragForces,
    "Moment_LE": moments,
    "Moment_c4": quartermoments,
    "CP": cps,
    "CL": cl_values,
    "CD_pressure": cd_values_pressure,
    "CD_wake": cd_values_wake,
    "CM_LE": cm_le_values,
    "CM_c4": cm_quarter_values
})

def split_sweeps(df):
    aoa_values = df["AoA_deg"].values

    max_aoa_idx = np.argmax(aoa_values)

    df_upward = df.iloc[:max_aoa_idx + 1].copy()

    df_downward = df.iloc[max_aoa_idx:].copy()

    return df_upward, df_downward

df_upward, df_downward = split_sweeps(df)

df_upward_avg = df_upward.groupby("AoA_deg", as_index=False).mean()
df_downward_avg = df_downward.groupby("AoA_deg", as_index=False).mean()

df_avg_combined = df.groupby("AoA_deg", as_index=False).mean()


# ----------------------------
# Plotting
# ----------------------------
def plot_force_vs_aoa_split(x_up, y_up, x_down, y_down, ylabel, label_up, label_down, color_up, color_down):
    plt.plot(x_up, y_up, 'k-', lw=1)
    plt.scatter(x_up, y_up, color=color_up, edgecolors='k', label=label_up, s=20)
    plt.plot(x_down, y_down, 'k--', lw=1)
    plt.scatter(x_down, y_down, color=color_down, edgecolors='k', label=label_down, s=20, marker='s')
    plt.xlabel(r'$\alpha$ (deg)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def plot_force_vs_aoa_single(x, y, ylabel, label, color):
    plt.plot(x, y, 'k-', lw=1)
    plt.scatter(x, y, color=color, edgecolors='k', label=label, s=20)
    plt.xlabel(r'$\alpha$ (deg)')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

plot_force_vs_aoa_single(df_upward_avg["AoA_deg"], df_upward_avg["CD_pressure"], r"$C_d$", r"Cd aoa", "r")

plot_force_vs_aoa_split(df_upward_avg["AoA_deg"], df_upward_avg["Lift"], 
                        df_downward_avg["AoA_deg"], df_downward_avg["Lift"],
                        r'L [N/m]', "Upward Sweep", "Downward Sweep", "r", "darkred")

plot_force_vs_aoa_split(df_upward_avg["AoA_deg"], df_upward_avg["Drag"], 
                        df_downward_avg["AoA_deg"], df_downward_avg["Drag"],
                        r'D [N/m]', "Upward Sweep", "Downward Sweep", "b", "darkblue")

plot_force_vs_aoa_split(df_upward_avg["AoA_deg"], df_upward_avg["Moment_LE"], 
                        df_downward_avg["AoA_deg"], df_downward_avg["Moment_LE"],
                        r'M$_{LE}$ [Nm/m]', "Upward Sweep", "Downward Sweep", "orange", "darkorange")

plot_force_vs_aoa_split(df_upward_avg["AoA_deg"], df_upward_avg["Moment_c4"], 
                        df_downward_avg["AoA_deg"], df_downward_avg["Moment_c4"],
                        r'M$_{c/4}$ [Nm/m]', "Upward Sweep", "Downward Sweep", "green", "darkgreen")

plot_force_vs_aoa_split(df_upward_avg["AoA_deg"], df_upward_avg["CP"], 
                        df_downward_avg["AoA_deg"], df_downward_avg["CP"],
                        r'$c_p$ [m]', "Upward Sweep", "Downward Sweep", "purple", "darkviolet")

# Drag coefficient comparison
plt.plot(df_avg_combined["AoA_deg"], df_avg_combined["CD_pressure"], 'k-')
plt.scatter(df_avg_combined["AoA_deg"], df_avg_combined["CD_pressure"], color="blue", edgecolors='k', label="Pressure taps")
plt.plot(df_avg_combined["AoA_deg"], df_avg_combined["CD_wake"], 'k-')
plt.scatter(df_avg_combined["AoA_deg"], df_avg_combined["CD_wake"], color="yellow", marker="v", edgecolors='k', label="Wake rake")
plt.xlabel(r'$\alpha$ [deg]')
plt.ylabel(r'$C_d$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# Lift coefficient
plot_force_vs_aoa_split(df_upward_avg["AoA_deg"], df_upward_avg["CL"], 
                        df_downward_avg["AoA_deg"], df_downward_avg["CL"],
                        r'$C_l$', "Upward Sweep", "Downward Sweep", "red", "darkred")
# Drag polar
plt.plot(df_upward_avg["CD_pressure"], df_upward_avg["CL"], 'k-')
plt.scatter(df_upward_avg["CD_pressure"], df_upward_avg["CL"], color="blue", edgecolors='k', label="Upward - Pressure")
plt.plot(df_downward_avg["CD_pressure"], df_downward_avg["CL"], 'k--')
plt.scatter(df_downward_avg["CD_pressure"], df_downward_avg["CL"], color="cyan", edgecolors='k', label="Downward - Pressure", marker='s')
plt.plot(df_upward_avg["CD_wake"], df_upward_avg["CL"], 'k-')
plt.scatter(df_upward_avg["CD_wake"], df_upward_avg["CL"], color="yellow", marker="v", edgecolors='k', label="Upward - Wake")
plt.plot(df_downward_avg["CD_wake"], df_downward_avg["CL"], 'k--')
plt.scatter(df_downward_avg["CD_wake"], df_downward_avg["CL"], color="gold", marker="^", edgecolors='k', label="Downward - Wake", s=20)
plt.xlabel(r'$C_d$')
plt.ylabel(r'$C_l$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# Moment coefficients
plot_force_vs_aoa_split(df_upward_avg["AoA_deg"], df_upward_avg["CM_LE"], 
                        df_downward_avg["AoA_deg"], df_downward_avg["CM_LE"],
                        r'$C_{m,LE}$', "Upward Sweep", "Downward Sweep", "orange", "darkorange")

plot_force_vs_aoa_split(df_upward_avg["AoA_deg"], df_upward_avg["CM_c4"], 
                        df_downward_avg["AoA_deg"], df_downward_avg["CM_c4"],
                        r'$C_{m,c/4}$', "Upward Sweep", "Downward Sweep", "green", "darkgreen")


aoas_array_up = df_upward_avg["AoA_deg"].to_numpy()
cl_array_up = df_upward_avg["CL"].to_numpy()

mask = (aoas_array_up >= -5) & (aoas_array_up <= 5)
aoas_rad_up = np.deg2rad(aoas_array_up[mask])
cl_sel_up = cl_array_up[mask]

coeffs = np.polyfit(aoas_rad_up, cl_sel_up, 1)
cl_alpha = coeffs[0]
cl0 = coeffs[1]

print(f"C_l_alpha = {cl_alpha:.4f} per rad")
print(f"C_l at 0 AoA = {cl0:.4f}")

with open("cl_alpha_2d.json", "w") as f:
    json.dump({"cl_alpha_2d": cl_alpha}, f)

aero_data_2d = {
    "upward_sweep": {
        "AoA_deg": df_upward_avg["AoA_deg"].to_list(),
        "CL_2D": df_upward_avg["CL"].to_list(),
        "CD_2D_pressure": df_upward_avg["CD_pressure"].to_list(),
        "CD_2D_wake": df_upward_avg["CD_wake"].to_list(),
        "CM_LE": df_upward_avg["CM_LE"].to_list(),
        "CM_c4": df_upward_avg["CM_c4"].to_list()
    },
    "downward_sweep": {
        "AoA_deg": df_downward_avg["AoA_deg"].to_list(),
        "CL_2D": df_downward_avg["CL"].to_list(),
        "CD_2D_pressure": df_downward_avg["CD_pressure"].to_list(),
        "CD_2D_wake": df_downward_avg["CD_wake"].to_list(),
        "CM_LE": df_downward_avg["CM_LE"].to_list(),
        "CM_c4": df_downward_avg["CM_c4"].to_list()
    },
    "combined": {
        "AoA_deg": df_avg_combined["AoA_deg"].to_list(),
        "CL_2D": df_avg_combined["CL"].to_list(),
        "CD_2D_pressure": df_avg_combined["CD_pressure"].to_list(),
        "CD_2D_wake": df_avg_combined["CD_wake"].to_list(),
        "CM_LE": df_avg_combined["CM_LE"].to_list(),
        "CM_c4": df_avg_combined["CM_c4"].to_list()
    }
}

with open("aero_coefficients_2d.json", "w") as f:
    json.dump(aero_data_2d, f, indent=4)

print("\nData summary:")
print(f"Upward sweep: {len(df_upward_avg)} unique AoA points")
print(f"Downward sweep: {len(df_downward_avg)} unique AoA points")
print(f"Combined: {len(df_avg_combined)} unique AoA points")

