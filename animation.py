import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

R = 1.0  # Radien på cylindern


def simulate(nu):
    U = 1.0  # Exempelvärde
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

    return u_theta, r, t_max

# Skapa en animation över olika värden på nu
nus = np.linspace(0.1, 0.5, num=5)
fig, ax = plt.subplots(figsize=(10, 6))

def update(frame):
    nu = nus[frame]
    u_theta, r, t_max = simulate(nu)
    ax.clear()
    im = ax.imshow(u_theta, extent=[R, 10*R, t_max, 0], aspect='auto')
    ax.set_xlabel('Radie (r)')
    ax.set_ylabel('Tid (t)')
    ax.set_title('Utveckling av $u_\\theta$ över tid och radie, för $\\nu = {:.2f}$'.format(nu))
    return im,

ani = animation.FuncAnimation(fig, update, frames=len(nus), blit=True)
plt.colorbar(ax.imshow(np.zeros((100, 100))), ax=ax, label='$u_\\theta$')

# Visa animationen
HTML(ani.to_jshtml())
