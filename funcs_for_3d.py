import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

# Constants 
b = 0.4169 #m
c = 0.16 #m
S = 0.066146 #m^2
A = b / c

def load_3D_sim():
    with open('T1-20_0 m_s-VLM1-Inviscid.txt') as f:
        content = f.readlines()
        header = content[7] # alpha Beta CL CDi CDv CD CY Cl Cm Cn Cni QInf XCP
        clipped_lines = content[8:-2]

        values = []
        for line in clipped_lines:
            values += [line.strip().split()]
        values = np.array(values, dtype=np.float32)

    alpha_sim = values[:, 0]
    CL_sim = values[:, 2]
    CD_i_sim = values[:, 3]
    CD_visc_sim = 0 # Cannot be simulated using VLM

    return alpha_sim, CL_sim, CD_i_sim

def load_3D_experiment():
    '''
    Stolen from 3Dforces.py
    '''

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



    with open("aero_coefficients_2d.json") as f:
        data_2d = json.load(f)

    aoa_2d_upward = pd.Series(data_2d["upward_sweep"]["AoA_deg"])
    cl_2d_upward  = pd.Series(data_2d["upward_sweep"]["CL_2D"])
    cd_2d_upward  = pd.Series(data_2d["upward_sweep"]["CD_2D_pressure"])


    CDi_from_tau = df_avg["CL"]**2 / (np.pi * A) * (1 + tau)


    df_overlap = df_avg[df_avg["AoA_deg"] <= max(aoa_2d_upward)].copy()
    CDi_from_graph = df_overlap["CD"] - cd_2d_upward

    return (df_avg["AoA_deg"], df_avg["CL"]), (aoa_2d_upward, CDi_from_graph), (df_avg["AoA_deg"], CDi_from_tau)
