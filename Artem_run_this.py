import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from funcs_for_3d import *

alpha_sim, CL_sim, CD_i_sim = load_3D_sim()
(alpha_CL_exp, CL_exp), (alpha_CDi_difference, CDi_difference), (alpha_CDi_tau, CDi_tau) = load_3D_experiment()

plt.plot(alpha_sim, CL_sim, 'k-', zorder=1, lw=1)
plt.scatter(alpha_sim, CL_sim, edgecolors='k', color="r", label="Simulated $C_{L}$ of 3D Wing", zorder=2, s=20)
plt.plot(alpha_CL_exp, CL_exp, 'k-', zorder=1, lw=1)
plt.scatter(alpha_CL_exp, CL_exp, edgecolors='k', color="blue", label="Experimental $C_{L}$ of 3D Wing", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}\right)$')
plt.ylabel(r'$C_{L}$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()

plt.plot(alpha_sim, CD_i_sim, 'k-', zorder=1, lw=1)
plt.scatter(alpha_sim, CD_i_sim, edgecolors='k', color="r", label="$C_{D, i}$ from simulation", zorder=2, s=20)
plt.plot(alpha_CDi_difference, CDi_difference, 'k-', zorder=1, lw=1)
plt.scatter(alpha_CDi_difference, CDi_difference, edgecolors='k', color="blue", label="$C_{D, i}$ from experimental drag difference", zorder=2, s=20)
plt.plot(alpha_CDi_tau, CDi_tau, 'k-', zorder=1, lw=1)
plt.scatter(alpha_CDi_tau, CDi_tau, edgecolors='k', color="purple", label="$C_{D, i}$ from experimental tau", zorder=2, s=20)
plt.xlabel(r'$\alpha\;\left(\text{deg}\right)$')
plt.ylabel(r'$C_{L}$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.show()
