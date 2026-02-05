import numpy as np
import matplotlib.pyplot as plt
import math

from GOPH547Lab01.gravity import (
    gravity_potential_point, gravity_effect_point,
    )

def gravity_potential_test():
    #Point mass below survey point, testing gravity_potential_point function.
    x = np.array([0.0, 0.0, 0.0])
    xm = np.array([0.0, 0.0, -10.0])
    m = 1.0e7
    G = 6.674e-11

    #With a known r, can calculate true value.
    r = 10.0
    Measured = G * m / r

    #Comparing value calculated with function to the true value to determine if they are similar.
    Calculated = gravity_potential_point(x, xm, m, G)

    assert np.isclose(Calculated, Measured)


def gravity_effect_test():
    #Since point mass directly below survey point, there should be a positive gz, testing gravity_effect_point.
    x = np.array([0.0, 0.0, 0.0])
    xm = np.array([0.0, 0.0, -10.0])
    m = 1.0e7
    G = 6.674e-11

    #With a known r, can calculate true value.
    r = 10.0
    dz = 10.0
    Measured = G * m * dz / (r * r * r)

    #Comparing value calculated with function to the true value to determine if they are similar.
    Calculated = gravity_effect_point(x, xm, m, G)

    assert np.isclose(Calculated, Measured)


def test_gravity_decreases_with_distance():
    #Gravity effects should decrease in strength with increased distance from object, testing gravity_potential_point function.
    xm = np.array([0.0, 0.0, -10.0])
    m = 1.0e7

    #Points near and far from point mass.
    x_near = np.array([0.0, 0.0, 0.0])
    x_far = np.array([0.0, 0.0, 100.0])

    #Computing the gravitational potentials for both distances.
    U_near = gravity_potential_point(x_near, xm, m)
    U_far = gravity_potential_point(x_far, xm, m)

    #Calculating the vertical gravity effects for both distances.
    gz_near = gravity_effect_point(x_near, xm, m)
    gz_far = gravity_effect_point(x_far, xm, m)

    #Test to ensure near values are larger than further values.
    assert abs(U_near) > abs(U_far)
    assert abs(gz_near) > abs(gz_far)
