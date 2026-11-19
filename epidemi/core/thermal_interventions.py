"""Thermal interventions on an already generated temperature realization.

They are applied to the complete baseline realization (the two calendar years of a
season) and draw no random numbers, so every day of the modified series is paired
with the same day of the baseline. They are used for the experiments on day-to-day
variability and minimum temperature at fixed ten-day block means.
"""
import numpy as np


def transform(tm, tn, dates, noise=1., amp=1.):
    """Changes the day-to-day variability within ten-day blocks and the daily gap Tmean - Tmin.

    Within-block deviations (`noise`): in each ten-day block (days 1-10, 11-20 and
    21 to the end of the month) the deviation of each day's Tmean from the block mean
    is multiplied by `noise`, and the same daily change is added to Tmin. Block means
    and the daily gap Tmean - Tmin are unchanged.

    Daily gap (`amp`): Tmin' = Tmean - amp*(Tmean - Tmin), with Tmean unchanged.
    amp < 1 raises Tmin (warmer nights) and amp > 1 lowers it.

    When both are given, `noise` is applied first. Results are not clipped to the
    physical limits of the generator; Tmin <= Tmean is preserved by construction.

    Args:
        tm: daily Tmean of the baseline realization.
        tn: daily Tmin of the baseline realization.
        dates: date of each day (`pd.DatetimeIndex`, same length as `tm`).
        noise: factor of the within-block deviations (1 = no change).
        amp: factor of the daily gap Tmean - Tmin (1 = no change).

    Returns:
        (tmean, tmin): modified series (copies; the inputs are not changed).
    """
    mt = np.array(tm, copy=True)
    nt = np.array(tn, copy=True)
    if noise != 1:
        grupos = dates.year*1000 + dates.month*10 + np.minimum((dates.day - 1)//10, 2)
        for grupo in np.unique(grupos):
            ii = grupos == grupo
            delta = (noise - 1)*(mt[ii] - mt[ii].mean())
            mt[ii] += delta
            nt[ii] += delta
    if amp != 1:
        nt = mt - amp*(mt - nt)
    return mt, nt
