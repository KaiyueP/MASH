import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
plt.rcParams['font.size'] = 15

autoev = 27.2114079527  # 1 Hartree in eV

def plot_mash_population(
    pop_file,
    n_start=0,
    time_conversion_factor=0.0241888,  # a.u. -> fs; set to None to skip
    save_pop_fig=None,
    save_avgE_fig="avgE_compare.pdf",
):
    """
    Plot (1) state populations vs time from a MASH population file,
    and (2) the corresponding average energy vs time.

    Parameters
    ----------
    pop_file : str
        Path to the MASH population file. First column = time,
        remaining columns = populations of each state.
    n_start : int, optional
        Starting index (0-based) in the exciton.dat file for the states.
    time_conversion_factor : float or None, optional
        Factor to multiply the time column by (e.g. a.u. -> fs).
        If None, no conversion is done.
    save_pop_fig : str or None, optional
        Filename for saving the population figure. If None, not saved.
    save_avgE_fig : str or None, optional
        Filename for saving the average energy figure. If None, not saved.

    Returns
    -------
    fig_pop : matplotlib.figure.Figure
        Figure for populations vs time.
    fig_avgE : matplotlib.figure.Figure
        Figure for average energy vs time.
    """

    # ---- Load population data ----
    mash = np.loadtxt(pop_file)

    # Convert time if requested
    #if time_conversion_factor is not None:
        #mash[:, 0] *= time_conversion_factor

    # Determine the number of population columns actually present
    num_lines = mash.shape[1] - 1  # first column is time

    # We'll only use up to min(num_lines, n_state) states consistently
    n_state = num_lines

    # ---- Plot 1: populations vs time ----
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=n_state - 1)

    fig_pop, ax_pop = plt.subplots(figsize=(10, 6))

    for n in range(n_state):
        color = cmap(norm(n))
        ax_pop.plot(
            mash[:, 0],
            mash[:, n + 1],
            label=f"mash{n + 1}",
            color=color,
            linestyle='-'
        )

    ax_pop.set_xlabel("Time (fs)" if time_conversion_factor is not None else "Time (a.u.)")
    ax_pop.set_ylabel("Population")
    ax_pop.set_title("Population Over Time")
    ax_pop.legend(loc='best', fontsize='small')
    fig_pop.tight_layout()

    if save_pop_fig is not None:
        fig_pop.savefig(save_pop_fig, dpi=500)

    plt.close()

    return fig_pop


fig_pop= plot_mash_population(
    pop_file="./pop.out",
    n_start=0,
    save_pop_fig="pop_mash.png",
)


