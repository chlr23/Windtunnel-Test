import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from funcs_for_3d import *

alpha_sim, CL_sim, CD_i_sim = load_3D_sim()
(alpha_CL_exp, CL_exp), (alpha_CDi_difference, CDi_difference), (alpha_CDi_tau, CDi_tau), dynamic_press_exp, dynamic_press_overlap_exp = load_3D_experiment()

# CL
plt.figure(figsize=(8, 6))
plt.plot(alpha_sim, CL_sim, 'k-', zorder=1, lw=1)
plt.scatter(alpha_sim, CL_sim, edgecolors='k', color="r", label="Simulated $C_{L}$ of 3D Wing", zorder=2, s=20, marker='o')
plt.plot(alpha_CL_exp, CL_exp, 'k-', zorder=1, lw=1)
plt.scatter(alpha_CL_exp, CL_exp, edgecolors='k', color="blue", label="Experimental $C_{L}$ of 3D Wing", zorder=2, s=20, marker='^')
plt.xlabel(r'$\alpha\;\left(\text{deg}\right)$')
plt.ylabel(r'$C_{L}$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show(block=False)

# CDi
plt.figure(figsize=(8, 6))
plt.plot(alpha_sim, CD_i_sim, 'k-', zorder=1, lw=1)
plt.scatter(alpha_sim, CD_i_sim, edgecolors='k', color="r", label="$C_{D, i}$ from simulation", zorder=2, s=20, marker='o')
plt.plot(alpha_CDi_difference, CDi_difference, 'k-', zorder=1, lw=1)
plt.scatter(alpha_CDi_difference, CDi_difference, edgecolors='k', color="blue", label="$C_{D, i}$ from experimental drag difference", zorder=2, s=20, marker='^')
plt.plot(alpha_CDi_tau, CDi_tau, 'k-', zorder=1, lw=1)
plt.scatter(alpha_CDi_tau, CDi_tau, edgecolors='k', color="purple", label="$C_{D, i}$ from experimental tau", zorder=2, s=20, marker='s')
plt.xlabel(r'$\alpha\;\left(\text{deg}\right)$')
plt.ylabel(r'$C_{D,i}$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show(block=False)

dynamic_press_sim = 0.5*1.225*20**2
# L
L_sim = CL_sim*dynamic_press_sim*S
L_exp = CL_exp*dynamic_press_exp*S
plt.figure(figsize=(8, 6))
plt.plot(alpha_sim, L_sim, 'k-', zorder=1, lw=1)
plt.scatter(alpha_sim, L_sim, edgecolors='k', color="r", label="Simulated lift of 3D Wing", zorder=2, s=20, marker='o')
plt.plot(alpha_CL_exp, L_exp, 'k-', zorder=1, lw=1)
plt.scatter(alpha_CL_exp, L_exp, edgecolors='k', color="blue", label="Experimental lift of 3D Wing", zorder=2, s=20, marker='^')
plt.xlabel(r'$\alpha\;\left(\text{deg}\right)$')
plt.ylabel(r'$L$ ($N$)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show(block=False)

# Di
Di_sim = CD_i_sim*dynamic_press_sim*S
Di_difference = CDi_difference*dynamic_press_overlap_exp*S
Di_tau = CDi_tau*dynamic_press_exp*S
plt.figure(figsize=(8, 6))
plt.plot(alpha_sim, Di_sim, 'k-', zorder=1, lw=1)
plt.scatter(alpha_sim, Di_sim, edgecolors='k', color="r", label="$D_{i}$ from simulation", zorder=2, s=20, marker='o')
plt.plot(alpha_CDi_difference, Di_difference, 'k-', zorder=1, lw=1)
plt.scatter(alpha_CDi_difference, Di_difference, edgecolors='k', color="blue", label="$D_{i}$ from experimental drag difference", zorder=2, s=20, marker='^')
plt.plot(alpha_CDi_tau, Di_tau, 'k-', zorder=1, lw=1)
plt.scatter(alpha_CDi_tau, Di_tau, edgecolors='k', color="purple", label="$D_{i}$ from experimental tau", zorder=2, s=20, marker='s')
plt.xlabel(r'$\alpha\;\left(\text{deg}\right)$')
plt.ylabel(r'$D_{i}$ ($N$)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show(block=True)