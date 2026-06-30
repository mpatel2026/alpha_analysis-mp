import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from desc.grid import LinearGrid
import desc.io as dscio
import numpy as np


eq_path = "/pscratch/sd/m/mpatel26/equil/equil_Helios_G1600-12-89_QA2e-1_Bxdl25_free_L12_M12_N20.h5"
fam = dscio.load(eq_path, file_format="hdf5")
try:  # if file is an EquilibriaFamily, use final Equilibrium
    eq = fam[-1]
except:  # file is already an Equilibrium
    eq = fam
    
phi = 0.0  # toroidal angle
r_L = 2e-2  # Larmor radius (meters)

grid = LinearGrid(rho=101, theta=101, zeta=phi, NFP=eq.NFP)
data = eq.compute(["rho", "theta", "|B|", "|grad(B)|"], grid=grid)
field_scale = data["|B|"] / data["|grad(B)|"]  # (meters)
rho_star = r_L / field_scale * 1e2  # (%)

rho = data["rho"].reshape((grid.num_rho, grid.num_theta), order="F")
theta = data["theta"].reshape((grid.num_rho, grid.num_theta), order="F")
rho_star = rho_star.reshape((grid.num_rho, grid.num_theta), order="F")

fig, ax = plt.subplots(figsize=(5, 5))
divider = make_axes_locatable(ax)
contourf_kwargs = {}
contourf_kwargs["norm"] = matplotlib.colors.Normalize()
contourf_kwargs["levels"] = np.linspace(0, np.max(rho_star), 101)
contourf_kwargs["cmap"] = "jet"
contourf_kwargs["extend"] = "both"
cax_kwargs = {"size": "5%", "pad": 0.05}
im = ax.contourf(theta, rho, rho_star, **contourf_kwargs)
cax = divider.append_axes("right", **cax_kwargs)
cbar = fig.colorbar(im, cax=cax, format=lambda x, _: f"{x:.2f}%")
cbar.update_ticks()
ax.set_xlabel("$\\theta$")
ax.set_ylabel("$\\rho$")
ax.set_title(f"$\\rho^* ~(\\phi = {phi:.2f})$")
fig.tight_layout()
plt.show()
plt.savefig("figs/rho_star_contour.png", dpi=600)