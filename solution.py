import numpy as np
import matplotlib.pyplot as plt

my = 0.00089
rho = 1000
U = 0.1
nu = my / rho
R = 0.5  # Antaget värde för R, behöver justeras enligt specifika förhållanden
omega = lambda t: 0.5 * t # Linjärt beroende, omega = 0.5 * t

# Rums- och tidsparametrar
r_max = 10 * R  # Anta att detta approximerar oändligheten
dr = 0.1        # Rumssteg
dt = 0.01       # Tidssteg
r = np.arange(R, r_max, dr)
t_max = 2.0     # Total tid för simulering
time = np.arange(0, t_max, dt)

# Initialisera lösningen u_theta
u_theta = np.zeros((len(time), len(r)))

# Tillämpa randvillkoret vid r = R
u_theta[:, 0] = omega(time) * R

# Lösning med finita differensmetoden
for n in range(0, len(time) - 1):
    for j in range(1, len(r) - 1):
        # Central differens i r
        dudr = (u_theta[n, j+1] - u_theta[n, j-1]) / (2 * dr)
        d2udr2 = (u_theta[n, j+1] - 2*u_theta[n, j] + u_theta[n, j-1]) / dr**2

        # Tidsderivatet av u_theta
        du_dt = (U * R / r[j]) * dudr + (U * R / r[j]**2) * u_theta[n, j] + nu * (d2udr2 + (1/r[j]) * dudr - u_theta[n, j] / r[j]**2)

        # Explicit Euler för att uppdatera u_theta
        u_theta[n + 1, j] = u_theta[n, j] + dt * du_dt

# Visa resultatet
plt.figure(figsize=(10, 6))
plt.imshow(u_theta, extent=[R, r_max, t_max, 0], aspect='auto')
plt.colorbar(label='$u_\\theta$')
plt.xlabel('Radie (r)')
plt.ylabel('Tid (t)')
plt.title('Utveckling av $u_\\theta$ över tid och radie')
plt.show()
