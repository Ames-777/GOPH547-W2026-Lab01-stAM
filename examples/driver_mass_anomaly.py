import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
from pathlib import Path

from GOPH547Lab01.gravity import gravity_effect_point, gravity_potential_point

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

#Loading the anomaly data file.
def load_anomaly_data(filename):
    
    data = loadmat(filename)
    
    x = data["x"]
    y = data["y"]
    z = data["z"]
    rho = data["rho"]
    
    return x, y, z, rho


#Calculating mass and barycenter.
def compute_mass_properties(x, y, z, rho):
    
    cell_volume = 2.0**3
    mass_cells = rho * cell_volume

    total_mass = np.sum(mass_cells)

    bary_x = np.sum(mass_cells * x) / total_mass
    bary_y = np.sum(mass_cells * y) / total_mass
    bary_z = np.sum(mass_cells * z) / total_mass

    print("Total mass:", total_mass)
    print("Barycentre:", bary_x, bary_y, bary_z)
    print("Max density:", np.max(rho))
    print("Mean density:", np.mean(rho))

    return total_mass, np.array([bary_x, bary_y, bary_z])


#Creating the cross-sectional plots.
def plot_density_sections(x, y, z, rho, bary):
    script_dir = Path(__file__).resolve().parent
    outputs = ensure_dir(script_dir.parent / "outputs")

    rho_xz = np.mean(rho, axis=0)
    rho_yz = np.mean(rho, axis=1)
    rho_xy = np.mean(rho, axis=2)

    X_xz = np.mean(x, axis=0)
    Z_xz = np.mean(z, axis=0)

    Y_yz = np.mean(y, axis=1)
    Z_yz = np.mean(z, axis=1)

    X_xy = np.mean(x, axis=2)
    Y_xy = np.mean(y, axis=2)

    vmin = min(rho_xz.min(), rho_yz.min(), rho_xy.min())
    vmax = max(rho_xz.max(), rho_yz.max(), rho_xy.max())

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    c1 = axes[0].contourf(X_xz, Z_xz, rho_xz, 30, vmin=vmin, vmax=vmax)
    axes[0].plot(bary[0], bary[2], "xk", markersize=4)
    axes[0].set_title("XZ Mean Density")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Z (m)")
    plt.colorbar(c1, ax=axes[0])

    c2 = axes[1].contourf(Y_yz, Z_yz, rho_yz, 30, vmin=vmin, vmax=vmax)
    axes[1].plot(bary[1], bary[2], "xk", markersize=4)
    axes[1].set_title("YZ Mean Density")
    axes[1].set_xlabel("Y (m)")
    axes[1].set_ylabel("Z (m)")
    plt.colorbar(c2, ax=axes[1])

    c3 = axes[2].contourf(X_xy, Y_xy, rho_xy, 30, vmin=vmin, vmax=vmax)
    axes[2].plot(bary[0], bary[1], "xk", markersize=4)
    axes[2].set_title("XY Mean Density")
    axes[2].set_xlabel("X (m)")
    axes[2].set_ylabel("Y (m)")
    plt.colorbar(c3, ax=axes[2])

    output1 = outputs / "Mean_Density_Plots.png"
    fig.savefig(output1, dpi=300)
    plt.close(fig)
    print(f"Saved: {output1}")

#Calculating the region of non-zero density:
def compute_region_stats(x, y, z, rho, frac=0.10):
    threshold = frac * np.max(rho)
    mask = rho >= threshold

    print("\nNon-negligible region:")
    print("Threshold:", threshold)
    print("X range:", (x[mask].min(), x[mask].max()))
    print("Y range:", (y[mask].min(), y[mask].max()))
    print("Z range:", (z[mask].min(), z[mask].max()))
    print("Mean density:", np.mean(rho[mask]))

    return mask


#Forward modelling the gravity:
def forward_gravity(x, y, z, rho, z_obs, spacing):

    pad = 10.0
    x_grid = np.arange(np.min(x)-pad, np.max(x)+pad, spacing)
    y_grid = np.arange(np.min(y)-pad, np.max(y)+pad, spacing)

    X, Y = np.meshgrid(x_grid, y_grid)

    gz = np.zeros_like(X)
    U = np.zeros_like(X)

    cell_volume = 2.0**3

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    rho_flat = rho.flatten()

    # density cutoff
    rho_cut = 0.10 * np.max(rho_flat)
    keep = rho_flat >= rho_cut

    x_flat = x_flat[keep]
    y_flat = y_flat[keep]
    z_flat = z_flat[keep]
    rho_flat = rho_flat[keep]

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            obs = np.array([X[i, j], Y[i, j], z_obs])

            g_total = 0.0
            U_total = 0.0

            for xc, yc, zc, rc in zip(x_flat, y_flat, z_flat, rho_flat):
                m_cell = rc * cell_volume
                xm = np.array([xc, yc, zc])

                g_total += gravity_effect_point(obs, xm, m_cell)
                U_total += gravity_potential_point(obs, xm, m_cell)

            gz[i, j] = g_total
            U[i, j] = U_total

    return X, Y, U, gz
    


    print("\Creating forward model.")
    for z_int in z_dist:
        print(f"  Calculating gz at z={z_dist} m.")
        X, Y, U, gz = forward_model_density(x, y, z, rho, x_s, y_s, z_int, rho_frac_cut=0.10)
        gz_maps[z_int] = gz


#Calculating the second order derivatives:
def d2x(F, dx):
    out = np.full_like(F, np.nan, dtype=float)
    out[:, 1:-1] = (F[:, 2:] - 2 * F[:, 1:-1] + F[:, :-2]) / (dx**2)
    return out


def d2y(F, dy):
    out = np.full_like(F, np.nan, dtype=float)
    out[1:-1, :] = (F[2:, :] - 2 * F[1:-1, :] + F[:-2, :]) / (dy**2)
    return out


#Creating the gravity maps.
def plot_gravity_maps(results):

   #Ensuring the colourbars are the same for all plots.
    all_gz = np.array([gz for _, _, _, gz in results])
    vmin = all_gz.min()
    vmax = all_gz.max()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for ax, (z_obs, X, Y, gz) in zip(axes, results):
        c = ax.contourf(X, Y, gz, 30, vmin=vmin, vmax=vmax)
        ax.set_title(f"gz at z = {z_obs} m")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        plt.colorbar(c, ax=ax)


#Main Function.
def main():
    script_dir = Path(__file__).resolve().parent
    outputs = ensure_dir(script_dir.parent / "outputs")    

    x, y, z, rho = load_anomaly_data("anomaly_data.mat")

    total_mass, bary = compute_mass_properties(x, y, z, rho)

    plot_density_sections(x, y, z, rho, bary)

    compute_region_stats(x, y, z, rho)

    elevations = [0.0, 1.0, 100.0, 110.0]
    results = []

    for z_obs in elevations:
        X, Y, U, gz = forward_gravity(x, y, z, rho, z_obs, spacing=5.0)
        results.append((z_obs, X, Y, gz))

    #Calculating the first order derivatives.
    gz0 = results[0][3]
    gz1 = results[1][3]
    gz100 = results[2][3]
    gz110 = results[3][3]

    dgz_dz_0 = (gz1 - gz0) / 1.0
    dgz_dz_100 = (gz110 - gz100) / 10.0

    print("\ndgz/dz computed.")

    gz_dict = {z: gz for (z, X, Y, gz) in results}
    
    # --- Get grid (same for all) ---
    _, X, Y, _ = results[0]

    #Elevations
    z_levels = [0.0, 1.0, 100.0, 110.0]

    #Properly scaling colour.
    gz_stack = np.stack([gz_dict[z] for z in z_levels])
    vmin = gz_stack.min()
    vmax = gz_stack.max()

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

#Contour plots on each subplot.
    last_contour = None
    for ax, z in zip(axes, z_levels):
        gz = gz_dict[z]
        last_contour = ax.contourf(X, Y, gz, 30, vmin=vmin, vmax=vmax)
        ax.set_title(f"gz at z = {z} m")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    #Same colourbar for all plots.
    fig.colorbar(last_contour, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

    output2 = outputs / "gz_4_Elevation_Plots.png"
    fig.savefig(output2, dpi=300)
    plt.close(fig)
    print(f"Saved: {output2}")
    
    #Calculating the second order derivatives from defined functions.
    dx = 5.0
    d2z0 = -(d2x(gz0, dx) + d2y(gz0, dx))
    d2z100 = -(d2x(gz100, dx) + d2y(gz100, dx))

    vmin2 = float(np.nanmin([d2z0, d2z100]))
    vmax2 = float(np.nanmax([d2z0, d2z100]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    c1 = axes[0].contourf(X, Y, d2z0, 30, vmin=vmin2, vmax=vmax2, cmap="plasma")
    axes[0].set_title(r"$\partial^2 g_z/\partial z^2$ at z=0 m.")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    fig.colorbar(c1, ax=axes[0])

    c2 = axes[1].contourf(X, Y, d2z100, 30, vmin=vmin2, vmax=vmax2, cmap="plasma")
    axes[1].set_title(r"$\partial^2 g_z/\partial z^2$ at z=100 m.")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    fig.colorbar(c2, ax=axes[1])


    output3 = outputs / "Second_Order_Derivative_Plots.png"
    fig.savefig(output3, dpi=300)
    plt.close(fig)
    print(f"Saved: {output3}")

    print("\nDone.")


if __name__ == "__main__":
    main()
