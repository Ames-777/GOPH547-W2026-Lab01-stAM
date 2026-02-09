import numpy as np
import matplotlib.pyplot as plt
import math

def gravity_potential_point(x, xm, m, G=6.674e-11) :
    """Compute the gravity potential due to a point mass.
    
    Parameters
    ----------
    x : array_like, shape=(3,)
        Coordinates of survey point.
    xm : array_like, shape=(3,)
        Coordinates of point mass anomaly.
    m : float
        Mass of the anomaly.
    G : float, optional, default=6.674e-11
        Constant of gravitation.
        Default in SI units.
        Allows user to modify if using different unit.

    Returns
    -------
    float
        Gravity potential at x due to anomaly at xm.
    """

    #Variable Definitions:
    G = float(6.674e-11)
    x = np.asarray(x, dtype = float).reshape(3)
    xm = np.asarray(xm, dtype = float).reshape(3)
    r_vec = x - xm
    r = np.linalg.norm(r_vec)
    #Prevents equation from being divided by zero.
    if r == 0.0:
        raise ValueError("Survey point and point mass locations cannot be the same value.")

    #Calculating gravity potential.
    U = (G * m) / r

    return float(U)

def gravity_effect_point(x, xm, m, G=6.674e-11) :
    """Compute the vertical gravity effect due to a point mass (positive downward).

    Parameters
    ----------
    x : array_like, shape=(3,)
        Coordinates of survey point.
    xm : array_like, shape=(3,)
        Coordinates of point mass anomaly.
    m : float
        Mass of the anomaly.
    G : float, optional, default=6.67e-11
        Constant of gravitation.
        Default in SI units.
        Allows user to modify if using different unit.

    Returns
    -------
    float
        Gravity effect at x due to anomaly at xm.
    """

    #Variable Definitions:
    G = float(6.674e-11)
    x = np.asarray(x, dtype = float).reshape(3)
    xm = np.asarray(xm, dtype = float).reshape(3)
    r_vec = x - xm
    r = np.linalg.norm(r_vec)
    #Prevents equation from being divided by zero.
    if r == 0.0:
        raise ValueError("Survey point and point mass location cannot be the same.")
        
    z_diff = x[2] - xm[2]

    #Calculating the vertical gravity effect due to point mass.
    gz = (G * m * z_diff) / (r * r * r)

    return float(gz)

