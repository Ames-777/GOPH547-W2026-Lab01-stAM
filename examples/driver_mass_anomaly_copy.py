import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from GOPH547Lab01.gravity import gravity_effect_point


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

    cell_volume = 2.0*2.0*2.0  # 2m cube

    mass_cells = rho * cell_volume
    total_mass = np.sum(mass_cells)

    bary_x = np.sum(mass_cells * x) / total_mass
    bary_y = np.sum(mass_cells * y) / total_mass
    bary_z = np.sum(mass_cells * z) / total_mass

    max_density = np.max(rho)
    mean_density = np.mean(rho)

    print("Total mass:", total_mass)
    print("Barycentre:", bary_x, bary_y, bary_z)
    print("Max density:", max_density)
    print("Mean density:", mean_density)

    return total_mass, np.array([bary_x, bary_y, bary_z])


#Creating the cross-sectional plots.
def plot_density_sections(x, y, z, rho, bary):

    rho_xz = np.mean(rho, axis=0)
    rho_yz = np.mean(rho, axis=1)
    rho_xy = np.mean(rho, axis=2)

    vmin = min(rho_xz.min(), rho_yz.min(), rho_xy.min())
    vmax = max(rho_xz.max(), rho_yz.max(), rho_xy.max())

    fig, axes = plt.subplots(3, 1, figsize=(6, 6))

    X_xz = x.mean(axis=0)
    Z_xz = z.mean(axis=0)
    im1 = axes[0].contourf(X_xz, Z_xz, rho_xz, 30, vmin=vmin, vmax=vmax)
    axes[0].plot(bary[0], bary[2], "xk", markersize=6)
    axes[0].set_title("XZ Mean Density")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Z (m)")
    plt.colorbar(im1, ax=axes[0])

    Y_yz = y.mean(axis=1)
    Z_yz = z.mean(axis=1)
    im2 = axes[0].contourf(Y_yz, Z_yz, rho_yz, 30, vmin=vmin, vmax=vmax)
    axes[1].plot(bary[1], bary[2], "xk", markersize=6)
    axes[1].set_title("YZ Mean Density")
    axes[1].set_xlabel("Y (m)")
    axes[1].set_ylabel("Z (m)")
    plt.colorbar(im2, ax=axes[1])

    X_xy = x.mean(axis=2)
    Y_xy = z.mean(axis=2)
    im3 = axes[0].contourf(X_xy, Y_xy, rho_xy, 30, vmin=vmin, vmax=vmax)
    axes[2].plot(bary[0], bary[1], "xk", markersize=6)
    axes[2].set_title("XY Mean Density")
    axes[2].set_xlabel("X (m)")
    axes[2].set_ylabel("Y (m)")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()


#Forward modelling the gravity:
def forward_gravity(x, y, z, rho, z_obs, spacing):

    x_grid = np.arange(np.min(x), np.max(x), spacing)
    y_grid = np.arange(np.min(y), np.max(y), spacing)

    X, Y = np.meshgrid(x_grid, y_grid)
    gz = np.zeros_like(X)

    cell_volume = 2.0**3

    #Flattening the anomaly arrays for looping.
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    rho_flat = rho.flatten()

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            obs = np.array([X[i, j], Y[i, j], z_obs])
            g_total = 0.0

            for xc, yc, zc, rc in zip(x_flat, y_flat, z_flat, rho_flat):

                m_cell = rc * cell_volume
                xm = np.array([xc, yc, zc])

                g_total += gravity_effect_point(obs, xm, m_cell)

            gz[i, j] = g_total

    return X, Y, gz


#Creating contour plots.
def plot_gravity_maps(results):

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    for ax, (z_obs, X, Y, gz) in zip(axes.flat, results):

        c = ax.contourf(X, Y, gz, 30, cmap="plasma")
        ax.set_title(f"gz at z = {z_obs} m")
        plt.colorbar(c, ax=ax)

    plt.tight_layout()
    plt.show()



#Main function:
def main():

    x, y, z, rho = load_anomaly_data("anomaly_data.mat")

    total_mass, bary = compute_mass_properties(x, y, z, rho)

    plot_density_sections(x, y, z, rho, bary)

    elevations = [0.0, 1.0, 100.0, 110.0]
    results = []

    for z_obs in elevations:

        X, Y, gz = forward_gravity(x, y, z, rho, z_obs, spacing=5.0)
        results.append((z_obs, X, Y, gz))

    plot_gravity_maps(results)


if __name__ == "__main__":
    main()
