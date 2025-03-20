
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

m_phi = 1.0
lambda_phi = 0.1
eta = 0.05
gamma = 4.2e4

def scalar_field_system(x, y):
    phi, dphi_dx = y
    nonlinear_damping = (1 - np.exp(-gamma * abs(dphi_dx)))
    d2phi_dx2 = m_phi**2 * phi - lambda_phi * phi**3 - eta * (phi * nonlinear_damping)
    return [dphi_dx, d2phi_dx2]

phi0 = 10.0
dphi_dx0 = 0.0
y0 = [phi0, dphi_dx0]
x_span = (0, 10)
x_eval = np.linspace(*x_span, 500)

sol = solve_ivp(scalar_field_system, x_span, y0, t_eval=x_eval, method='RK45')

plt.figure(figsize=(8, 5))
plt.plot(sol.t, sol.y[0], label=r'$\\phi(x)$', linewidth=2)
plt.xlabel('x')
plt.ylabel(r'$\\phi$ Field Amplitude')
plt.title(r'Scalar Field Stability Simulation ($\\gamma=4.2 \\times 10^4$)')
plt.grid(True)
plt.legend()
plt.savefig('../results/scalar_field_stability_plot.png')
