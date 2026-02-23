import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat

from GOPH547Lab01.gravity import gravity_potential_point, gravity_effect_point


#Generating the masses.
def generate_mass_set(total_mass, centroid):
    n = 5

    #Mass distributions:
    mu_m = total_mass / n
    sigma_m = total_mass / 100

    mu_xyz = np.array([0.0, 0.0, -10.0])
    sigma_xyz = np.array([20.0, 20.0, 2.0])

    #Generating the first 4 masses randomly.
    m = np.random.normal(mu_m, sigma_m, 4)
    xm = np.random.normal(mu_xyz, sigma_xyz, (4, 3))

    #Computing the 5th mass from the total mass:
    m5 = total_mass - np.sum(m)

    #Centroid constraint:
    weighted_sum = np.sum(m[:, None] * xm, axis=0)
    x5 = (total_mass * centroid - weighted_sum) / m5

    m = np.append(m, m5)
    xm = np.vstack([xm, x5])

    #Testing the depth constraint:
    if np.any(xm[:, 2] > -1.0):
        return generate_mass_set(total_mass, centroid)

    return m, xm


#Computing the grid:

def compute_fields_on_grid(x_bounds, y_bounds, z_obs, xm_list, m_list):

    X, Y = np.meshgrid(x_bounds, y_bounds)

    U = np.zeros_like(X)
    gz = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            x = np.array([X[i, j], Y[i, j], z_obs])

            U_total = 0.0
            gz_total = 0.0

            #Calculating gravity effect for multiple masses:
            for xm, m in zip(xm_list, m_list):
                U_total += gravity_potential_point(x, xm, m)
                gz_total += gravity_effect_point(x, xm, m)

            U[i, j] = U_total
            gz[i, j] = gz_total

    return X, Y, U, gz


#Plotting the figures:
def run_plots(m, xm, dx, set_number):

    z_levels = [0.0, 10.0, 100.0]

    x_bounds = np.arange(-100.0, 100.0 + dx, dx)
    y_bounds = np.arange(-100.0, 100.0 + dx, dx)

    results = []
    U_all = []
    gz_all = []

    for z in z_levels:
        X, Y, U, gz = compute_fields_on_grid(x_bounds, y_bounds, z, xm, m)
        results.append((z, X, Y, U, gz))
        U_all.append(U)
        gz_all.append(gz)

    U_all = np.array(U_all)
    gz_all = np.array(gz_all)

    U_min, U_max = U_all.min(), U_all.max()
    gz_min, gz_max = gz_all.min(), gz_all.max()

    fig, axes = plt.subplots(3, 2, figsize=(6, 6), constrained_layout=True)

    for row, (z, X, Y, U, gz) in enumerate(results):

        axU = axes[row, 0]
        cU = axU.contourf(X, Y, U, 30, vmin=U_min, vmax=U_max, cmap="plasma")
        axU.plot(X, Y, "xk", markersize=2)
        axU.set_title(f"U at z={z} m")
        fig.colorbar(cU, ax=axU)

        axG = axes[row, 1]
        cG = axG.contourf(X, Y, gz, 30, vmin=gz_min, vmax=gz_max, cmap="plasma")
        axG.plot(X, Y, "xk", markersize=2)
        axG.set_title(f"gz at z={z} m")
        fig.colorbar(cG, ax=axG)

    fig.suptitle(f"Mass Set {set_number} (dx={dx} m)")

    plt.savefig(f"mass_set_{set_number}_dx{int(dx)}.png")
    plt.show()


#Main function:
def main():

    total_mass = 1.0e7
    centroid = np.array([0.0, 0.0, -10.0])

    #Generating and saving the three mass sets.
    for i in range(1, 4):

        m, xm = generate_mass_set(total_mass, centroid)

        savemat(f"mass_set_{i}.mat", {"m": m, "xm": xm})

        #Reloading the generated mass sets.
        data = loadmat(f"mass_set_{i}.mat")
        m_loaded = data["m"].flatten()
        xm_loaded = data["xm"]

        for dx in [5.0, 25.0]:
            run_plots(m_loaded, xm_loaded, dx, i)


if __name__ == "__main__":
    main()
