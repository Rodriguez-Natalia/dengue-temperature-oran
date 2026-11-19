This folder contains the main code and data needed to reproduce the results of our
manuscript on temperature effects in a vector-host dengue model for San Ramón de la
Nueva Orán (Argentina).

INSTALLATION (Python >= 3.10)

    pip install -r requirements.txt
    pip install .

CONTENTS

epidemi/core/utils_changed_model.py
    Vector-host model (13 compartments, fourth-order Runge-Kutta with a daily step).
    change_k(k, beta_day, temporada, i_date, ...) runs the model between two dates with
    daily temperature, rainfall, humidity and imported cases, and one larval carrying
    capacity kappa per July-June season. It returns the daily new infections, the adult
    mosquitoes per person (V/H) and the daily state.

epidemi/core/temperature_generator.py
    Ten-day-block stochastic temperature generator (TemperatureGenerator). Draws anchor
    days every 3 days (Tmean normal, amplitude gamma), interpolates between them and
    adds noise scaled so that the daily variance matches the observed one.

epidemi/core/thermal_interventions.py
    Changes of day-to-day variability and of minimum temperature applied to a synthetic
    realization.

epidemi/core/utils.py
    Model constants, temperature- and rainfall-dependent rates, breeding-site water
    level, Runge-Kutta step and data loading.

epidemi/data/
    Daily weather of Orán (Servicio Meteorológico Nacional): ORAN_2001_2022.txt (model
    input) and oran_diaria_98_22.txt (generator parameters, 2000-2022).
    Assumed imported cases: serie_ci_2001_2022.txt.
    Monthly reported cases per season, 2009-2022: new_clinic_cases.txt. Cases for
    2009-2017 were provided by the National Ministry of Health of Argentina
    (Gutierrez et al. 2022, Infectious Disease Modelling 7(4), 823-834,
    https://doi.org/10.1016/j.idm.2022.10.004); from 2018 onwards they come from the
    open dengue surveillance data of the Ministry of Health
    (http://datos.salud.gob.ar/dataset/vigilancia-de-dengue-y-zika).

parametros/
    parametros_bloques_mv_2000_2022.txt  normal parameters of each ten-day block
    gamma_bloques_mv_2000_2022.txt       gamma parameters of the amplitude (Tmean - Tmin)
    generador_config.json                spacing and noise factor
    deltas_trimestrales_2000.txt         quarterly delta of each year (observed mean temperature)
    lambdas_trimestrales_2000.txt        quarterly lambda of each year (observed minimum temperature)
    kappa_ref.txt                        reference kappa of each season (written by calibracion_kappa.py)

scripts/calibracion_kappa.py
    Fits kappa_ref of each season with observed climate, so that simulated infections
    (July-June) equal the reported cases.

        python scripts/calibracion_kappa.py

scripts/escenario_temporada.py
    Building block of every experiment in the paper. It generates the synthetic
    temperature of a season, applies a perturbation (for example +1 °C from October to
    March) and runs the epidemic from the state on 1 July of the observed-climate chain,
    with paired realizations (seed [7000, season, r]). It prints the baseline and
    perturbed mean cases and their ratio R.

        python scripts/escenario_temporada.py --season 2016 --d_warm 1 --n 300

    For example, season 2016 with +1 °C from October to March and kappa = kappa_ref gives
    R = 8.677 with 300 realizations (about 20 s).

Seasons run from July to June and are named by the year in which they end.
