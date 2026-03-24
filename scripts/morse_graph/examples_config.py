"""Configuration for all 9 autonomous switched network examples.

Extracted from autonomous_switched_network_oscillatory_examples_executed.ipynb.
"""

import numpy as np
from .dynamics import (stuart_landau, radial_poly, subcritical_hopf,
                       van_der_pol, fitzhugh_nagumo, selkov, toggle_osc_surrogate)


# Default CMGDB parameters keyed by total state dimension (2*N).
# Subdivision is total (not per-dimension). subdiv_min is the level at which
# CMGDB starts SCC analysis; everything below is uniformly refined.
# subdiv_max is the deepest adaptive refinement allowed.
# Keep subdiv_min moderate to avoid blind uniform refinement.
CMGDB_PARAMS = {
    6: dict(subdiv_min=12, subdiv_max=24, subdiv_init=6, subdiv_limit=10000, tau=0.5),
    8: dict(subdiv_min=12, subdiv_max=24, subdiv_init=6, subdiv_limit=10000, tau=0.5),
}


def get_examples():
    """Return list of 9 example dicts. Must be called (not module-level) because
    oscillator factories create closures that capture mutable state."""
    examples = []

    examples.append(dict(
        name="EX01_stuart_landau_3nodes_detuned",
        N=3, eps=1.55, gamma=0.10, T=60.0, dt=0.02,
        f_list=[
            stuart_landau(mu=1.0, omega=0.95),
            stuart_landau(mu=1.0, omega=1.05),
            stuart_landau(mu=1.0, omega=1.18),
        ],
        X0=np.array([[1.4, 0.0], [0.1, 1.2], [-1.1, -0.4]])
    ))

    examples.append(dict(
        name="EX02_stuart_landau_4nodes_radius_mismatch",
        N=4, eps=1.35, gamma=0.09, T=65.0, dt=0.02,
        f_list=[
            stuart_landau(mu=0.7, omega=0.95),
            stuart_landau(mu=1.0, omega=1.00),
            stuart_landau(mu=1.3, omega=1.08),
            stuart_landau(mu=0.9, omega=1.16),
        ],
        X0=np.array([[1.2, 0.0], [0.1, 1.0], [-1.0, 0.2], [0.2, -1.1]])
    ))

    examples.append(dict(
        name="EX03_radial_two_cycles_3nodes",
        N=3, eps=1.25, gamma=0.08, T=70.0, dt=0.02,
        f_list=[
            radial_poly(omega=0.92, roots=(0.75, 1.45), stable_pattern="two_cycles"),
            radial_poly(omega=1.04, roots=(0.75, 1.45), stable_pattern="two_cycles"),
            radial_poly(omega=1.18, roots=(0.75, 1.45), stable_pattern="two_cycles"),
        ],
        X0=np.array([[0.9, 0.0], [1.55, 0.1], [0.1, -0.85]])
    ))

    examples.append(dict(
        name="EX04_radial_two_cycles_4nodes_mixed_basins",
        N=4, eps=1.15, gamma=0.07, T=70.0, dt=0.02,
        f_list=[
            radial_poly(omega=0.90, roots=(0.65, 1.55), stable_pattern="two_cycles"),
            radial_poly(omega=1.00, roots=(0.65, 1.55), stable_pattern="two_cycles"),
            radial_poly(omega=1.10, roots=(0.65, 1.55), stable_pattern="two_cycles"),
            radial_poly(omega=1.22, roots=(0.65, 1.55), stable_pattern="two_cycles"),
        ],
        X0=np.array([[0.75, 0.0], [1.65, 0.0], [0.0, 0.7], [-1.55, 0.0]])
    ))

    examples.append(dict(
        name="EX05_subcritical_hopf_rest_cycle",
        N=4, eps=1.00, gamma=0.08, T=80.0, dt=0.02,
        f_list=[
            subcritical_hopf(mu=-0.05, omega=0.92, beta=1.35),
            subcritical_hopf(mu=-0.05, omega=1.00, beta=1.35),
            subcritical_hopf(mu=-0.05, omega=1.08, beta=1.35),
            subcritical_hopf(mu=-0.05, omega=1.18, beta=1.35),
        ],
        X0=np.array([[0.15, 0.0], [1.05, 0.0], [0.0, -1.0], [-0.2, 0.1]])
    ))

    examples.append(dict(
        name="EX06_vanderpol_3nodes",
        N=3, eps=2.15, gamma=0.05, T=80.0, dt=0.02,
        f_list=[
            van_der_pol(mu=1.1),
            van_der_pol(mu=1.6),
            van_der_pol(mu=2.0),
        ],
        X0=np.array([[1.8, 0.0], [0.3, 1.6], [-1.5, 0.4]])
    ))

    examples.append(dict(
        name="EX07_fitzhugh_nagumo_4nodes",
        N=4, eps=1.65, gamma=0.06, T=120.0, dt=0.03,
        f_list=[
            fitzhugh_nagumo(a=0.7, b=0.8, tau=12.5, I=0.52),
            fitzhugh_nagumo(a=0.7, b=0.8, tau=12.5, I=0.56),
            fitzhugh_nagumo(a=0.7, b=0.8, tau=12.5, I=0.60),
            fitzhugh_nagumo(a=0.7, b=0.8, tau=12.5, I=0.64),
        ],
        X0=np.array([[-1.1, -0.2], [0.7, 0.5], [1.2, -0.1], [0.0, 1.0]])
    ))

    examples.append(dict(
        name="EX08_selkov_3nodes",
        N=3, eps=0.80, gamma=0.05, T=120.0, dt=0.02,
        f_list=[
            selkov(alpha=0.08, beta=0.55),
            selkov(alpha=0.08, beta=0.60),
            selkov(alpha=0.08, beta=0.65),
        ],
        X0=np.array([[0.7, 1.0], [1.2, 0.7], [0.9, 1.4]])
    ))

    examples.append(dict(
        name="EX09_toggle_rotational_3nodes",
        N=3, eps=1.75, gamma=0.08, T=70.0, dt=0.02,
        f_list=[
            toggle_osc_surrogate(a=2.4, b=2.0, k=0.9),
            toggle_osc_surrogate(a=2.4, b=2.0, k=1.0),
            toggle_osc_surrogate(a=2.4, b=2.0, k=1.1),
        ],
        X0=np.array([[1.3, 0.4], [-1.1, 0.9], [0.7, -1.0]])
    ))

    return examples
