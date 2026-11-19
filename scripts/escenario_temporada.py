"""Thermal scenario for one season: synthetic temperature, perturbation and epidemic.

For season T (July of T-1 to June of T):

1. Context: the model state and breeding-site humidity on 1 July come from the
   chain run with observed climate and kappa_ref (``parametros/kappa_ref.txt``).
2. Temperature: the ten-day-block generator draws the season with the block
   parameters, the quarterly delta and lambda of each calendar year (conditioned on
   the observed quarterly means) and, optionally, a perturbation: ``d_warm`` added
   to October-March and ``d_cold`` added to one cold block (July-September or
   April-June).
3. Epidemic: the model runs with observed rainfall, humidity and imported cases,
   kappa = factor * kappa_ref, and returns the infections from July to June.

Realization r uses the seed [7000, T, r], so a baseline and a perturbed scenario
with the same r share the random innovations (paired contrasts). The ratio
R = mean(perturbed cases) / mean(baseline cases).

Usage (from the repository root)::

    python scripts/escenario_temporada.py --season 2016 --d_warm 1 --n 300
"""
import argparse
import calendar
import json
import os
import warnings

import numpy as np
import pandas as pd

from epidemi.core.utils_changed_model import change_k, ALPHA, poblacion
from epidemi.core.temperature_generator import TemperatureGenerator

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAR = os.path.join(RAIZ, 'parametros')

beta_day = 3
TEMPORADAS = list(range(2009, 2023))
SEMILLA = 7000
BLOQUES_FRIOS = {'jul_sep': slice(18, 27), 'abr_jun': slice(9, 18)}

# Fixed initial condition (1 January 2007), 25% of the population immune (H_R)
ic_fija = np.array([22876., 102406., 24962., 2003., 28836., 0., 0., 0., 0.,
                    ALPHA*poblacion, 0., 0., (1. - ALPHA)*poblacion])
kappa_ref = pd.read_csv(os.path.join(PAR, 'kappa_ref.txt'), sep=' ', index_col='temporada')['k_recalibrado']

# --- Temperature generator: block parameters (2000-2022), spacing and noise factor ---
CFG = json.load(open(os.path.join(PAR, 'generador_config.json')))
PB = pd.read_csv(os.path.join(PAR, 'parametros_bloques_mv_2000_2022.txt'), sep=' ')
PG = pd.read_csv(os.path.join(PAR, 'gamma_bloques_mv_2000_2022.txt'), sep=' ')
DELTAS = pd.read_csv(os.path.join(PAR, 'deltas_trimestrales_2000.txt'), sep=' ', index_col='anio')
LAMBDAS = pd.read_csv(os.path.join(PAR, 'lambdas_trimestrales_2000.txt'), sep=' ', index_col='anio')
GEN = {bis: TemperatureGenerator(leap_year=bis, spacing=CFG['espaciado'], noise_factor=CFG['factor_correccion'])
       for bis in (True, False)}


def ventana_temporada(T):
    """(start_sim, end_sim, start_target, end_target). The run ends on 2 July so that
    the last row of the state table is 1 July."""
    ini_sim = '2007-01-01' if T == 2009 else f'{T-1}-07-01'
    return ini_sim, f'{T}-07-02', f'{T-1}-07-01', f'{T}-06-30'


def estados_observados():
    """State and breeding-site humidity on 1 July of each season (observed-climate chain)."""
    estados, estado, H = {}, ic_fija, None
    for T in TEMPORADAS:
        estados[T] = (estado, H)
        ini_sim, fin_sim, _, _ = ventana_temporada(T)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, _, tabla, H = change_k([kappa_ref[T]]*20, beta_day, [ini_sim, fin_sim], '2001-01-01',
                                      initial_c=estado, ci=None, initial_H=H, devolver_H=True)
        estado = tabla.iloc[-1].values
    return estados


def temperatura_temporada(T, r, d_warm=0.0, d_cold=0.0, bloque_frio='jul_sep'):
    """(tmean, tmin) of realization r of season T, from 1 July of T-1 to 2 July of T."""
    np.random.seed([SEMILLA, T, r])
    pert = np.zeros(36)
    pert[:9] = d_warm                      # January-March
    pert[27:] = d_warm                     # October-December
    pert[BLOQUES_FRIOS[bloque_frio]] = d_cold
    tmean, tmin = [], []
    for anio in (T - 1, T):
        medias = PB.tmean.values + np.repeat(DELTAS.loc[anio].values, 9) + pert
        escala = PG.scale.values*np.repeat(LAMBDAS.loc[anio].values, 9)
        tn, _, tm = GEN[calendar.isleap(anio)].generate_year(
            medias, PB.std_tmean.values, PB.std_tmax.values, PB.std_tmin.values, PG.Shape.values, escala)
        tmean.append(tm); tmin.append(tn)
    desde = (np.datetime64(f'{T-1}-07-01') - np.datetime64(f'{T-1}-01-01')).astype(int)
    largo = (np.datetime64(f'{T}-07-02') - np.datetime64(f'{T-1}-07-01')).astype(int) + 1
    return np.concatenate(tmean)[desde:desde + largo], np.concatenate(tmin)[desde:desde + largo]


def casos_temporada(kappa, T, tmean, tmin, estado, H):
    """Daily infections from July to June of season T (T > 2009) with synthetic temperature."""
    ini_sim, fin_sim, ini_obj, fin_obj = ventana_temporada(T)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G_T, _, _ = change_k([kappa]*20, beta_day, [ini_sim, fin_sim], ini_sim, initial_c=estado,
                             ci=None, tmin=tmin, tmean=tmean, initial_H=H)
    fechas = np.arange(np.datetime64(ini_sim), np.datetime64(ini_sim) + len(G_T), dtype='datetime64[D]')
    return G_T[(fechas >= np.datetime64(ini_obj)) & (fechas <= np.datetime64(fin_obj))]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--season', type=int, default=2016, help='season (year in which it ends), 2010-2022')
    p.add_argument('--d_warm', type=float, default=1.0, help='shift of October-March (°C)')
    p.add_argument('--d_cold', type=float, default=0.0, help='shift of the cold block (°C)')
    p.add_argument('--cold_block', default='jul_sep', choices=list(BLOQUES_FRIOS))
    p.add_argument('--kappa_factor', type=float, default=1.0, help='kappa = factor * kappa_ref')
    p.add_argument('--n', type=int, default=300, help='number of realizations')
    a = p.parse_args()

    T = a.season
    estado, H = estados_observados()[T]
    kappa = a.kappa_factor*kappa_ref[T]
    base, pert = np.zeros(a.n), np.zeros(a.n)
    for r in range(a.n):
        base[r] = casos_temporada(kappa, T, *temperatura_temporada(T, r), estado, H).sum()
        pert[r] = casos_temporada(kappa, T, *temperatura_temporada(T, r, a.d_warm, a.d_cold, a.cold_block),
                                  estado, H).sum()
    print(f'season {T}, kappa = {kappa:.1f} ({a.kappa_factor} x kappa_ref), {a.n} realizations')
    print(f'baseline: mean {base.mean():.1f} cases (p2.5-p97.5: {np.percentile(base, 2.5):.0f}-{np.percentile(base, 97.5):.0f})')
    print(f'perturbed (d_warm {a.d_warm:+g}, d_cold {a.d_cold:+g} in {a.cold_block}): mean {pert.mean():.1f}')
    print(f'R = {pert.mean()/base.mean():.3f}')
