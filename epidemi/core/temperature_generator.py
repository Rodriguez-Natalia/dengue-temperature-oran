"""Ten-day-block stochastic generator of daily temperature.

The year is divided into 36 ten-day blocks (three per month: 10/10/11 days in 31-day
months, 10/10/10 in 30-day months, 10/10/8 in February and 10/10/9 in a leap-year
February). Each block has its own parameters: mean and standard deviation of Tmean,
standard deviations of Tmax and Tmin, and a gamma distribution for the amplitude.

One year is generated in three steps:

1. Anchor days, every ``spacing`` days (plus the last day of the year):
   Tmean ~ Normal(block mean, block std of Tmean) and amplitude
   Delta ~ Gamma(block shape, block scale); Tmax = Tmean + Delta and
   Tmin = Tmean - Delta.
2. Intermediate days: linear interpolation between consecutive anchors plus normal
   noise. The noise of a day at fractional position f between its two anchors has
   standard deviation

       block std * sqrt(2 f (1 - f)) * noise_factor

   It is zero at an anchor and largest halfway between anchors, so the anchors and
   the noise share the variance of the block instead of counting it twice.
   Tmean is generated first; Tmax and Tmin are redrawn until Tmax > Tmean and
   Tmin < Tmean on every day.
3. Physical limits: all values are kept between ``min_temp_limit`` and
   ``max_temp_limit``.

To perturb the temperature, modify the block parameters before generating (e.g. add
a shift to the block means, or multiply the gamma scale to change the amplitude).

Example::

    gen = TemperatureGenerator(leap_year=False)
    np.random.seed(1)
    tmin, tmax, tmean = gen.generate_year(means, std_tmean, std_tmax, std_tmin, shape, scale)

Random numbers are drawn from NumPy's global generator (``np.random.seed``).
"""

import numpy as np
from scipy.stats import gamma

SPACING = 3          # default days between anchors
NOISE_FACTOR = 0.70  # default noise factor of intermediate days


class TemperatureGenerator:
    """Daily Tmin, Tmax and Tmean for one calendar year.

    Args:
        leap_year (bool): True for a 366-day year.
        spacing (int): days between anchor days.
        noise_factor (float): factor c of the noise of intermediate days.
        max_temp_limit, min_temp_limit (float): physical limits (degC).
        max_attempts (int): maximum redraws to satisfy Tmin < Tmean < Tmax.
    """

    def __init__(self, leap_year=False, spacing=SPACING, noise_factor=NOISE_FACTOR,
                 max_temp_limit=46, min_temp_limit=-2, max_attempts=1000):
        self.leap_year = leap_year
        self.spacing = spacing
        self.noise_factor = noise_factor
        self.max_temp_limit = max_temp_limit
        self.min_temp_limit = min_temp_limit
        self.max_attempts = max_attempts
        feb = 29 if leap_year else 28
        self.month_days = [31, feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.total_days = sum(self.month_days)
        self.block_of_day = self._blocks()

    def _blocks(self):
        """Block index (0-35) of each day of the year."""
        block = []
        for m, days in enumerate(self.month_days):
            for j, size in enumerate((10, 10, days - 20)):
                block += [3*m + j]*size
        return np.array(block)

    def _anchor_days(self):
        """Anchor days: every `spacing` days from day 0, plus the last day of the year."""
        days = np.arange(0, self.total_days, self.spacing)
        if days[-1] != self.total_days - 1:
            days = np.append(days, self.total_days - 1)
        return days

    def _limit(self, t, kind=None):
        """Applies the upper limit (kind 'max'), the lower limit ('min') or both (None)."""
        if kind == 'max':
            return min(t, self.max_temp_limit)
        if kind == 'min':
            return max(t, self.min_temp_limit)
        return np.clip(t, self.min_temp_limit, self.max_temp_limit)

    def _anchor(self, b, mean_params, std_tmean, shape_params, scale_params):
        """(Tmean, Tmax, Tmin) of an anchor day in block b."""
        for _ in range(self.max_attempts):
            tmean = np.random.normal(mean_params[b], std_tmean[b])
            delta = gamma.rvs(a=shape_params[b], loc=0, scale=scale_params[b])
            tmax = self._limit(tmean + delta, 'max')
            tmin = self._limit(tmean - delta, 'min')
            if tmax > tmin:
                return (tmax + tmin)/2, tmax, tmin
        return (self.max_temp_limit + self.min_temp_limit)/2, self.max_temp_limit - 1, self.min_temp_limit + 1

    def _interpolate(self, days, values, std, kind, tmean=None):
        """Daily series from anchor values: linear interpolation plus scaled noise.

        kind is 'mean', 'max' or 'min'. For 'max' and 'min', `tmean` is the daily Tmean
        series and the noise is redrawn until Tmax > Tmean or Tmin < Tmean.
        """
        series = []
        for x1, x2, y1, y2 in zip(days[:-1], days[1:], values[:-1], values[1:]):
            m = (y2 - y1)/(x2 - x1)
            b0 = y1 - m*x1
            xs = np.linspace(x1, x2, int(x2 - x1), endpoint=False)
            ys = m*xs + b0
            for j, (x, y) in enumerate(zip(xs, ys)):
                if j == 0:                       # anchor day: no noise
                    series.append(self._limit(y, None if kind == 'mean' else kind))
                    continue
                day = int(round(x))
                f = (x - x1)/(x2 - x1)
                # keep this order of operations: reordering changes the last bit of the results
                sd = std[self.block_of_day[day]]*(np.sqrt(2.0*f*(1.0 - f))*self.noise_factor)
                if kind == 'mean':
                    series.append(self._limit(y + np.random.normal(0, sd)))
                    continue
                for attempt in range(self.max_attempts):
                    t = self._limit(y + np.random.normal(0, sd), kind)
                    if (t > tmean[day]) if kind == 'max' else (t < tmean[day]):
                        break
                    if attempt == self.max_attempts - 1:   # fallback, practically never used
                        t = max(y, tmean[day] + 0.5) if kind == 'max' else min(y, tmean[day] - 0.5)
                        t = self._limit(t, kind)
                series.append(t)
        series.append(self._limit(values[-1], None if kind == 'mean' else kind))
        return np.array(series[:self.total_days])

    def generate_year(self, mean_params, std_tmean, std_tmax, std_tmin, shape_params, scale_params):
        """Generates one year.

        Args:
            mean_params: mean of Tmean of each block (36), including any shift delta.
            std_tmean, std_tmax, std_tmin: standard deviations of each block (36).
            shape_params, scale_params: gamma amplitude parameters of each block (36);
                the scale includes any factor lambda.

        Returns:
            (tmin, tmax, tmean): arrays of 365 or 366 days.
        """
        days = self._anchor_days()
        anchors = np.array([self._anchor(self.block_of_day[d], mean_params, std_tmean, shape_params,
                                         scale_params) for d in days])
        tmean = self._interpolate(days, anchors[:, 0], std_tmean, 'mean')
        tmax = self._interpolate(days, anchors[:, 1], std_tmax, 'max', tmean)
        tmin = self._interpolate(days, anchors[:, 2], std_tmin, 'min', tmean)
        return tmin, tmax, tmean
