import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from double_pendulum import dynamics, inverse_dynamics

def total_energy(q1, q2, q1_dot, q2_dot, params):
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

    # Kinetic energy
    T = (
        0.5 * A * q1_dot**2
        + 0.5 * B * q2_dot**2
        + C * np.cos(q1 - q2) * q1_dot * q2_dot
    )

    # Potential energy
    V = (
        -(m1 * c1 + m2 * l1) * g * np.cos(q1)
        - m2 * c2 * g * np.cos(q2)
    )

    return T + V

params = {
    "m1": 5.0,
    "m2": 3.0,
    "l1": 0.45,
    "c1": 0.20,
    "c2": 0.20,
    "I1": 0.10,
    "I2": 0.05,
    "g": 9.81
}


initial_state = [
    np.deg2rad(30),   # q1
    np.deg2rad(-20),  # q2
    0.0,              # q1_dot
    0.0               # q2_dot
]


t_span = (0, 10)
t_eval = np.linspace(0, 10, 2000)

solution = solve_ivp(
    lambda t, state: dynamics(state, params),
    t_span,
    initial_state,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-11
)

q1 = solution.y[0]
q2 = solution.y[1]

q1_dot = solution.y[2]
q2_dot = solution.y[3]

q1_ddot = np.gradient(q1_dot, solution.t)
q2_ddot = np.gradient(q2_dot, solution.t)

energy = total_energy(
    q1, q2, q1_dot, q2_dot, params
)
tau1, tau2 = inverse_dynamics(
    q1, q2,
    q1_dot, q2_dot,
    q1_ddot, q2_ddot,
    params
)

print("Maximum |tau1|:", np.max(np.abs(tau1)))
print("Maximum |tau2|:", np.max(np.abs(tau2)))

print("Initial energy:", energy[0])
print("Final energy:", energy[-1])
print("Energy change:", energy[-1] - energy[0])
print("Maximum energy deviation:",
      np.max(np.abs(energy - energy[0])))

plt.plot(solution.t, np.rad2deg(q1), label="q1")
plt.plot(solution.t, np.rad2deg(q2), label="q2")

plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.legend()
plt.savefig("results/double_pendulum_motion.png", dpi=150)
print("Plot saved to results/double_pendulum_motion.png")