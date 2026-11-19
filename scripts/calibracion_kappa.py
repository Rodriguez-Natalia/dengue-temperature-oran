"""Reference larval carrying capacity (kappa_ref) per season with observed climate.

Writes ``parametros/kappa_ref.txt``. Each season T (July of T-1 to June of T)
is simulated from 1 July to 2 July of the following year with observed temperature,
rainfall and humidity, and the state and breeding-site humidity on 1 July are passed
to the next season. Season 2009 starts on 1 January 2007 (burn-in) with a single
kappa, and its target is measured from July 2008 to June 2009. For each season,
kappa is the first value at which simulated infections (July-June) equal the
reported cases. Seasons 2011 and 2015 (almost no reported cases) use a fixed
placeholder kappa = 150.

Usage (from the repository root)::

    python scripts/calibracion_kappa.py [output_file]
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from epidemi.core.utils_changed_model import change_k, load_data_file, ALPHA, poblacion

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

beta_day = 3
K_PLACEHOLDER = 150.0
TEMPORADAS = list(range(2009, 2023))
PLACEHOLDER = (2011, 2015)
K_GRID = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 180, 270, 400, 600, 900,
                   1350, 2000, 3000, 4000])

# Fixed initial condition (1 January 2007), 25% of the population immune (H_R)
ic_fija = np.array([22876., 102406., 24962., 2003., 28836., 0., 0., 0., 0.,
                    ALPHA*poblacion, 0., 0., (1. - ALPHA)*poblacion])

# Reported cases: rows = months July..June, columns = seasons 2009..
nc_casos = load_data_file('new_clinic_cases.txt')
casos_clinicos = {2009 + i: float(nc_casos[:, i].sum()) for i in range(nc_casos.shape[1])}


def ventana_temporada(T):
    """(start_sim, end_sim, start_target, end_target) of season T. The run ends on
    2 July so that the last row of ``tabla`` is the state on 1 July."""
    ini_sim = '2007-01-01' if T == 2009 else f'{T-1}-07-01'
    return ini_sim, f'{T}-07-02', f'{T-1}-07-01', f'{T}-06-30'


def correr_temporada(k, T, estado, H):
    """Runs season T with observed climate. Returns (July-June infections, state and H on 1 July)."""
    ini_sim, fin_sim, ini_obj, fin_obj = ventana_temporada(T)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G_T, _, tabla, H_fin = change_k([k]*20, beta_day, [ini_sim, fin_sim], '2001-01-01',
                                        initial_c=estado, ci=None, initial_H=H, devolver_H=True)
    fechas = np.arange(np.datetime64(ini_sim), np.datetime64(ini_sim) + len(G_T), dtype='datetime64[D]')
    assert fechas[-1] == np.datetime64(f'{T}-07-01')
    total = G_T[(fechas >= np.datetime64(ini_obj)) & (fechas <= np.datetime64(fin_obj))].sum()
    return total, tabla.iloc[-1].values, H_fin


def ajustar_k(T, estado, H, objetivo):
    """First kappa on K_GRID whose total crosses the target, refined with brentq."""
    def total_k(k):
        try:
            total = correr_temporada(k, T, estado, H)[0]
        except Exception:
            return np.nan
        return total if np.isfinite(total) else np.nan
    f = lambda k: (lambda v: (v if np.isfinite(v) else 1e12) - objetivo)(total_k(k))
    vals = np.array([total_k(k) for k in K_GRID]) - objetivo
    for k_a, k_b, v_a, v_b in zip(K_GRID[:-1], K_GRID[1:], vals[:-1], vals[1:]):
        if np.isfinite(v_a) and np.isfinite(v_b) and v_a < 0 <= v_b:
            return brentq(f, float(k_a), float(k_b), xtol=0.5)
    raise RuntimeError(f'{T}: no bracket found (max on grid: {np.nanmax(vals) + objetivo:.1f})')


def main(salida):
    filas, estado, H = [], ic_fija, None
    for T in TEMPORADAS:
        placeholder = T in PLACEHOLDER
        k = K_PLACEHOLDER if placeholder else ajustar_k(T, estado, H, casos_clinicos[T])
        total, estado, H = correr_temporada(k, T, estado, H)
        filas.append({'temporada': T, 'tipo': 'placeholder' if placeholder else 'ajustar',
                      'k_recalibrado': round(k, 2), 'total_sim': round(total, 2),
                      'total_real': casos_clinicos[T]})
        print(f'{T}: kappa={k:.2f} total={total:.1f} (reported {casos_clinicos[T]:.0f})', flush=True)
    pd.DataFrame(filas).set_index('temporada').to_csv(salida, sep=' ')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, 'parametros', 'kappa_ref.txt'))
