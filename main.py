
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Uppdaterade parametrar och värden
my = 0.00089
rho = 1000
nu = my / rho
R = 0.5  # Antaget värde för R, behöver justeras enligt specifika förhållanden
r_max = 100 * R
r_steps = 10**4
t_steps = 10**6
t_max = 10

# Skapa rums- och tidsnät
r = np.linspace(R, r_max, r_steps)
t = np.linspace(0, t_max, t_steps)

# Initialvillkor (startar med noll överallt)
u_theta_initial = np.zeros(r_steps)

# Funktion som beskriver tidsutvecklingen av u_theta
def du_theta_dt(t, u_theta, r, nu, R):
    # Diskretisering av rumsderivator
    dr = r[1] - r[0]
    du_dr = np.gradient(u_theta, dr)
    d2u_dr2 = np.gradient(du_dr, dr)

    # Tidsberoende för Omega(t)
    omega_t = 1 + np.sin(t) * 0.2

    # Randvillkor
    u_theta[0] = omega_t * R  # vid r = R
    u_theta[-1] = 0  # när r -> oändligheten

    # PDE diskretiserad
    du_theta_dt = nu * (d2u_dr2 + (1/r) * du_dr - (u_theta/r**2)) - (u_theta/r**2) - (R / r) * du_dr

    return du_theta_dt

# Löser PDE:n över tid
sol = solve_ivp(du_theta_dt, [0, t_max], u_theta_initial, t_eval=t, args=(r, nu, R))

# Visualisering av resultatet
plt.figure(figsize=(10, 6))
for i in range(0, len(sol.t), len(sol.t)//10):
    plt.plot(r, sol.y[:, i], label=f'Tid = {sol.t[i]:.1f}')
plt.title('Utveckling av u_theta över tid')
plt.xlabel('r')
plt.ylabel('u_theta')
plt.legend()
plt.show()

