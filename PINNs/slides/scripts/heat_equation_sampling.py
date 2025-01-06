import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Alegreya",
})


def solution(x, t, kappa, L):
    u = np.sin(np.pi*x/L) * np.exp(-kappa*np.pi**2/L**2 * t)
    return u


def plot():

    # --> Parameters.
    kappa, L = 1, 1
    tau = L**2 / kappa

    # --> Spatio-temporal domain.
    x = np.linspace(0, L, 256)
    t = np.linspace(0, tau/2, 512)
    x, t = np.meshgrid(x, t)

    # --> Solution.
    u = solution(x, t, kappa, L)

    # --------------------------------------------
    # -----     Spatio-temporal sampling     -----
    # --------------------------------------------

    samples = np.random.rand(64, 2)
    samples[:, 1] *= tau/2

    # --> Figure.
    fig, ax = plt.subplots(1, 1, figsize=(2, 4))

    pcm = ax.pcolormesh(x, t, u, shading="gouraud", cmap="hot", vmin=0, vmax=1)

    ax.scatter(samples[:, 0], samples[:, 1],
               marker="x", fc="white")

    ax.set(xlim=(0, L), xlabel=r"Position $x$",
           xticks=[0, L], xticklabels=["0", "L"])

    ax.set(ylim=(0, tau/2), ylabel=r"Time $t$",
           yticks=[0, tau/2], yticklabels=["0", r"$\tau$"])

    fig.colorbar(pcm, ax=ax, location="top")

    plt.savefig("../imgs/spatio_temporal_sampling.png", bbox_inches="tight",
                dpi=1200, transparent=True)

    # ------------------------------------------------
    # -----     Boundary conditions sampling     -----
    # ------------------------------------------------

    samples = np.random.rand(16, 2)
    samples *= tau/2

    # --> Figure.
    fig, ax = plt.subplots(1, 1, figsize=(2, 4))

    pcm = ax.pcolormesh(x, t, u, shading="gouraud", cmap="hot", vmin=0, vmax=1)

    ax.scatter(np.zeros(len(samples[:, 0])), samples[:, 0],
               marker="x", fc="dodgerblue", clip_on=False)

    ax.scatter(np.ones(len(samples[:, 0])), samples[:, 1],
               marker="x", fc="dodgerblue", clip_on=False)

    ax.set(xlim=(0, L), xlabel=r"Position $x$",
           xticks=[0, L], xticklabels=["0", "L"])

    ax.set(ylim=(0, tau/2), ylabel=r"Time $t$",
           yticks=[0, tau/2], yticklabels=["0", r"$\tau$"])

    fig.colorbar(pcm, ax=ax, location="top")

    plt.savefig("../imgs/boundary_conditions_sampling.png", bbox_inches="tight",
                dpi=1200, transparent=True)

    # -----------------------------------------------
    # -----     Initial conditions sampling     -----
    # -----------------------------------------------

    samples = np.random.rand(16) * L

    # --> Figure.
    fig, ax = plt.subplots(1, 1, figsize=(2, 4))

    pcm = ax.pcolormesh(x, t, u, shading="gouraud", cmap="hot", vmin=0, vmax=1)

    ax.scatter(samples, np.zeros_like(samples),
               marker="x", fc="dodgerblue", clip_on=False)

    ax.set(xlim=(0, L), xlabel=r"Position $x$",
           xticks=[0, L], xticklabels=["0", "L"])

    ax.set(ylim=(0, tau/2), ylabel=r"Time $t$",
           yticks=[0, tau/2], yticklabels=["0", r"$\tau$"])

    fig.colorbar(pcm, ax=ax, location="top")

    plt.savefig("../imgs/initial_condition_sampling.png", bbox_inches="tight",
                dpi=1200, transparent=True)


if __name__ == "__main__":
    plot()
    plt.show()
