import numpy as np
import matplotlib.pyplot as plt

from GOPH547Lab01.gravity import gravity_potential_point, gravity_effect_point


def compute_fields_on_grid(x_Bounds, y_Bounds, z_Vals, xm, m):
    #Creating a 2D grid with x and y coordinates.
    X, Y = np.meshgrid(x_Bounds, y_Bounds)

    #Creating empty grids to store the computed values within.
    U = np.zeros_like(X, dtype=float)
    gz = np.zeros_like(X, dtype=float)

    #Creating loop to store new grid locations.
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j], z_Vals], dtype=float)

            #Calculating U and gz for each point within the grid.
            U[i, j] = gravity_potential_point(x, xm, m)
            gz[i, j] = gravity_effect_point(x, xm, m)

    return X, Y, U, gz


def main():
    #Defining parameters:
    m = 1.0e7
    xm = np.array([0.0, 0.0, -10.0], dtype=float)
    z_Bounds = [0.0, 10.0, 100.0]

    #Creating two different grid spacings, 5 m and 25 m.
    dx_Grid = [5.0, 25.0]

    #Grid boundaries.
    for dx in dx_Grid:
        x_Bounds = np.arange(-100.0, 100.0 + dx, dx)
        y_Bounds = np.arange(-100.0, 100.0 + dx, dx)

        #Collecting calculations to help normallize the colourbars.
        Calculated = []
        U_All = []
        gz_All = []

        #Computing the gravity fields at each different height of observation.
        for z_Vals in z_Bounds:
            X, Y, U, gz = compute_fields_on_grid(x_Bounds, y_Bounds, z_Vals, xm, m)

            #Storing results.
            Calculated.append((z_Vals, X, Y, U, gz))
            U_All.append(U)
            gz_All.append(gz)

        #Converting the lists back into arrays.
        U_All = np.array(U_All)
        gz_All = np.array(gz_All)

        U_Min, U_Max = float(U_All.min()), float(U_All.max())
        gz_Min, gz_Max = float(gz_All.min()), float(gz_All.max())

        #Setting up the plots.
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), constrained_layout=True)

        for row, (z_Vals, X, Y, U, gz) in enumerate(Calculated):
            
            #Plotting the gravitational potential, U, on the left side:
            axU = axes[row, 0]
            cU = axU.contourf(X, Y, U, levels=30, vmin=U_Min, vmax=U_Max, cmap="plasma")
            axU.plot(X, Y, "xk", markersize=2)
            axU.set_title(f"Gravity Potential, U, at z = {z_Vals:.0f} m (dx = {dx:g} m)")
            axU.set_xlabel("x (m)")
            axU.set_ylabel("y (m)")
            fig.colorbar(cU, ax=axU)

            #Plotting the vertical gravitational effect, gx, on the right side:
            axG = axes[row, 1]
            cG = axG.contourf(X, Y, gz, levels=30, vmin=gz_Min, vmax=gz_Max, cmap="plasma")
            axG.plot(X, Y, "xk", markersize=2)
            axG.set_title(f"Vertical Gravity Effect, gz, at z = {z_Vals:.0f} m (dx = {dx:g} m)")
            axG.set_xlabel("x (m)")
            axG.set_ylabel("y (m)")
            fig.colorbar(cG, ax=axG)

        fig.suptitle(f"Single Point Mass Anomaly, where mass, m, is {m:.1e} kg,and point mass anomaly coordinates, xm, are {xm}.)", fontsize=14)
        plt.show()


if __name__ == "__main__":
    main()
