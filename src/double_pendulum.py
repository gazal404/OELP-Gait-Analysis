import numpy as np

def inverse_dynamics(q1, q2, q1_dot, q2_dot,
                     q1_ddot, q2_ddot, params):
    """
    Calculate joint torques from known motion.
    """

    m1 = params["m1"]
    m2 = params["m2"]
    l1 = params["l1"]
    c1 = params["c1"]
    c2 = params["c2"]
    I1 = params["I1"]
    I2 = params["I2"]
    g = params["g"]

    A = m1 * c1**2 + I1 + m2 * l1**2
    B = m2 * c2**2 + I2
    C = m2 * l1 * c2

    dq = q1 - q2

    tau1 = (
        A * q1_ddot
        + C * np.cos(dq) * q2_ddot
        + C * np.sin(dq) * q2_dot**2
        + (m1 * c1 + m2 * l1) * g * np.sin(q1)
    )

    tau2 = (
        C * np.cos(dq) * q1_ddot
        + B * q2_ddot
        - C * np.sin(dq) * q1_dot**2
        + m2 * c2 * g * np.sin(q2)
    )

    return tau1, tau2

def dynamics(state, params):
    """
    Two-link double pendulum dynamics.

    state = [q1, q2, q1_dot, q2_dot]

    q1, q2 are absolute angles measured from the vertical.
    """

    q1, q2, q1_dot, q2_dot = state

    m1 = params["m1"]
    m2 = params["m2"]
    l1 = params["l1"]
    c1 = params["c1"]
    c2 = params["c2"]
    I1 = params["I1"]
    I2 = params["I2"]
    g = params["g"]

    A = m1 * c1**2 + I1 + m2 * l1**2
    B = m2 * c2**2 + I2
    C = m2 * l1 * c2

    dq = q1 - q2

    # Mass matrix
    M = np.array([
        [A, C * np.cos(dq)],
        [C * np.cos(dq), B]
    ])

    # Non-inertial terms
    h = np.array([
        C * np.sin(dq) * q2_dot**2
        + (m1 * c1 + m2 * l1) * g * np.sin(q1),

        -C * np.sin(dq) * q1_dot**2
        + m2 * c2 * g * np.sin(q2)
    ])

    # No external torque for now
    tau = np.array([0.0, 0.0])

    # M q_ddot + h = tau
    q_ddot = np.linalg.solve(M, tau - h)

    return np.array([
        q1_dot,
        q2_dot,
        q_ddot[0],
        q_ddot[1]
    ])