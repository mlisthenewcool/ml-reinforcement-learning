import numpy as np


def obs_to_state_1(observation):
    """
    Transforms a raw observation into a discrete state.
    Currently evaluate only the pole angle and velocity.

    Number of possible states : 200(angle) * 12(velocity) = 2400
    """
    cart_position, cart_velocity, pole_angle, pole_velocity = observation

    # ----- CART POSITION -----
    # ----- (-4.8 to 4.8) -----
    # cart_positions = np.arange(start=-2, stop=2, step=0.5)
    # cart_position = np.digitize(cart_position, cart_positions)

    # ----- CART VELOCITY -----
    # ----- (-inf to inf) -----
    # cart_velocities = np.arange(start=-0.5, stop=0.5, step=0.1)
    # cart_velocity = np.digitize(cart_velocity, cart_velocities)

    # ----- POLE ANGLE  -----
    # ----- (-24 to 24) -----
    pole_angles = np.arange(start=-0.2, stop=0.2, step=0.01)
    pole_angle = np.digitize(pole_angle, pole_angles)

    # ----- POLE VELOCITY -----
    # ----- (-inf to inf) -----
    pole_velocities = np.arange(start=-3, stop=3, step=0.5)
    pole_velocity = np.digitize(pole_velocity, pole_velocities)

    """
    n_cart_positions = len(cart_positions)
    n_cart_velocities = len(cart_velocities)
    n_pole_angles = len(pole_angles)
    n_pole_velocities = len(pole_velocities)
    
    print('n cart_positions', n_cart_positions)
    print('n cart_velocities', n_cart_velocities)
    print('n pole_angles', n_pole_angles)
    print('n pole_velocities', n_pole_velocities)
    print('n total states', n_pole_angles * n_pole_velocities)
    """

    # return cart_position, cart_velocity, pole_angle, pole_velocity
    return pole_angle, pole_velocity
