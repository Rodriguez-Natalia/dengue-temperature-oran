"""Vector-host dengue model: equations (`modelo`) and simulation (`change_k`).

State (13 compartments, absolute numbers):

    0 E_D  dry eggs           5 V    adult female mosquitoes    9 H_S   susceptible humans
    1 E_W  wet eggs           6 V_S  susceptible vectors       10 H_E   exposed humans
    2 L    larvae             7 V_E  exposed vectors           11 H_I   infected humans
    3 P    pupae              8 V_I  infectious vectors        12 H_R   recovered humans
    4 M    emerging mosquitoes

Processes (constants and rate functions are in `utils.py`):

- Oviposition, biting and development depend on daily mean temperature; egg wetting
  depends on daily rainfall. Wetting and cold losses enter the equations as rates.
- Larval carrying capacity KL = households*(Kmax*H_t/Hmax + 1), where H_t is the
  water level of breeding sites, updated daily from rainfall and evaporation, and
  Kmax (kappa) is the larval carrying capacity per household.
- Thresholds on daily minimum temperature: cold loss of aquatic stages, cold
  mortality of adults, no larval maturation and no transmission below their values.
- Humans: SEIR with constant population; imported infections enter H_I and leave H_R.

`change_k` integrates the model with a fourth-order Runge-Kutta scheme and a daily
step, and counts the new human infections of each day (`nuevos_casos`).
"""
import numpy as np
import pandas as pd

from epidemi.core.utils import (
    load_data_file,
    MIObh, MIObv, bite_rate, mu_Dry, mu_Wet, ALPHA,
    Remove_infect, Remove_expose, MATAR_VECTORES, NO_INFECCION,
    Temp_ACUATICA, RATE_CASOS_IMP, MU_MOSQUITO_JOVEN,
    MADURACION_MOSQUITO, MU_MOSQUITA_ADULTA, Hmax, hogares,
    poblacion, oran,
    egg_wet, C_Gillet, hume, theta_T, rate_mx, muerte_V, runge, cases,
)

# Cold loss of wet eggs, larvae and pupae when Tmin < Temp_ACUATICA:
# rate 1 - SUPERVIVENCIA_ACUATICA = 0.4/day.
SUPERVIVENCIA_ACUATICA = 0.6

# Adult mortality rate when Tmin < MATAR_VECTORES: 0.5/day.
MUERTE_FRIO_ADULTAS = 0.5


# ################################ RIGHT-HAND SIDE ################################
def modelo(v,t,EV,H_t,Tmean,Tmin,Rain,CasosImp,beta_day, Kmax):
    """Right-hand side of the ODE (see the module docstring).

    Args:
        v: state (13 compartments).
        t: time (days since the start of the run); day int(t) of the forcing is used.
        EV: transition rate of exposed to infectious vectors on this day (sigma_V*V_E).
        H_t: water level of breeding sites (mm).
        Tmean, Tmin, Rain, CasosImp: daily series of the run.
        beta_day: oviposition rate.
        Kmax: effective larval carrying capacity kappa.
    """

    dv = np.zeros(13)

    tt = int(t)

    E_D 	=	v[0]
    E_W		=	v[1]
    L		=	v[2]
    P		=	v[3]
    M		=	v[4]
    V		=	v[5]
    V_S		=	v[6]
    V_E		=	v[7]
    V_I		=	v[8]
    H_S		=	v[9]
    H_E		=	v[10]
    H_I		=	v[11]
    H_R		=	v[12]

    Tm      = Tmean[tt]

    rain    = Rain[tt]

    Tmin    = Tmin[tt]

    beta_day_theta_0 = beta_day*theta_T(Tm)

    # wetting rate of dry eggs [1/day]
    fR      = egg_wet(rain)

    KL      = hogares*( Kmax*H_t/Hmax + 1.0 )

    m_E_C_G = 0.24*rate_mx(Tm, 10798.,100000.,14184.)*C_Gillet(L,KL)

    m_L     = 0.2088*rate_mx(Tm, 26018.,55990.,304.6)

    if (Tmin < 13.4):
        m_L = 0.

    mu_L    = 0.01 + 0.9725*np.exp(- (Tm - 4.85)/2.7035)

    C_L     = 1.5*(L/KL)

    m_P		=	0.384*rate_mx(Tm, 14931.,-472379.,148.)

    mu_P	=	0.01 + 0.9725*np.exp(- (Tm - 4.85)/2.7035)

    # transmission


    b_theta_pV	=	bite_rate*theta_T(Tm)*MIObv

    # human -> vector infection: zero when Tmin < 15 degC
    if ( Tmin < NO_INFECCION):
        b_theta_pV = 0.

    mu_V    = muerte_V(Tm)*MU_MOSQUITA_ADULTA

    # adult mortality with cold
    if ( Tmin < MATAR_VECTORES):
        mu_V = MUERTE_FRIO_ADULTAS

    m_M     = MADURACION_MOSQUITO

    mu_M    = MU_MOSQUITO_JOVEN

    b_theta_pH		=	bite_rate*theta_T(Tm)*MIObh
    if ( Tmin < NO_INFECCION ):
        b_theta_pH = 0.

    sigma_H			=	1./Remove_expose
    gama			=	1./Remove_infect

    deltaI = RATE_CASOS_IMP*CasosImp[tt]

    # cold loss of wet eggs, larvae and pupae, 0.4/day (0 if Tmin >= Temp_ACUATICA)
    if (Tmin < Temp_ACUATICA):
        mu_frio = 1. - SUPERVIVENCIA_ACUATICA
    else:
        mu_frio = 0.

    # equations

    dv[0]	=	beta_day_theta_0*V - fR*E_D - mu_Dry*E_D

    dv[1]	=	fR*E_D - m_E_C_G*E_W - mu_Wet*E_W - mu_frio*E_W

    dv[2]	=	m_E_C_G*E_W - m_L*L - ( mu_L + C_L )*L - mu_frio*L

    dv[3]	=	m_L*L - m_P*P - mu_P*P - mu_frio*P

    dv[4]	=	m_P*P - m_M*M - mu_M*M

    dv[5]	=	0.5*m_M*M - mu_V*V

    dv[6]	=	0.5*m_M*M - b_theta_pV*(H_I/poblacion)*V_S - mu_V*V_S

    dv[7]	=	b_theta_pV*(H_I/poblacion)*V_S - EV - mu_V*V_E

    dv[8]	=	EV - mu_V*V_I

    # susceptible humans are lost only by infection
    dv[9]	=	- b_theta_pH*(H_S/poblacion)*V_I

    dv[10]	=	b_theta_pH*(H_S/poblacion)*V_I - sigma_H*H_E

    dv[11]	=	sigma_H*H_E - gama*H_I + deltaI

    # imported infections leave the recovered (constant population)
    dv[12]	=	gama*H_I - deltaI

    return dv


def nuevos_casos(Tmean_t, Tmin_t, v):
    """New human infections of the day (vector -> human).

    Same force of infection as in `modelo`: zero when Tmin < NO_INFECCION (15 degC).
    """
    if Tmin_t < NO_INFECCION:
        return 0.
    return bite_rate*theta_T(Tmean_t) * MIObh * v[8] * v[9]/poblacion


def change_k(k,beta_day,temporada,i_date,initial_c=None,ci=None,tmin=None,
             rain=None, hr=None,
             tmean=oran[:,2], initial_H=None, devolver_H=False):
    """Runs the model between two dates, with kappa changing every 1 July.

    Args:
        k (list): kappa of each July-June season; the first value is used from
            temporada[0], and the next one from each 1 July.
        beta_day (float): oviposition rate.
        temporada (tuple): (start date, end date) of the run, e.g.
            ('2015-07-01', '2016-07-02').
        i_date (str): date of the first element of `tmean`, and of `tmin`, `rain` and
            `hr` when they are given.
        initial_c (array, optional): initial state (13 compartments). If None, a
            fixed initial condition is used, with 25% of the population immune.
        ci (tuple, optional): (date, number) of a single imported infection. If None,
            the daily series data/serie_ci_2001_2022.txt is used.
        tmin, rain, hr (array, optional): daily minimum temperature, rainfall and
            relative humidity, indexed from `i_date`. If None, the observed series of
            data/ORAN_2001_2022.txt are used, indexed from 2001-01-01.
        tmean (array): daily mean temperature indexed from `i_date` (default: the
            observed series from 2001-01-01, so `i_date` must then be '2001-01-01').
        initial_H (float, optional): initial water level of breeding sites (None = 24).
        devolver_H (bool): if True, also returns the final water level.

    Returns:
        (G_T, aedes, tabla), or (G_T, aedes, tabla, H_t) with devolver_H=True:
        daily new human infections, adult female mosquitoes per person (V/H), and a
        DataFrame with the daily state. Outputs have one day less than the window:
        row t is the state t days after temporada[0] (row 0 is the initial state).
        To chain seasons, run from 1 July to 2 July of the following year and pass
        the last row (state on 1 July) and the final H_t to the next season.
    """

    i_temporada = (np.datetime64(temporada[0]) - np.datetime64(i_date)).astype(int)
    f_temporada = (np.datetime64(temporada[1]) - np.datetime64(i_date)).astype(int) + 1

    i_ci = (np.datetime64(temporada[0]) - np.datetime64('2001-01-01')).astype(int)
    f_ci = (np.datetime64(temporada[1]) - np.datetime64('2001-01-01')).astype(int) + 1

    Tmean = tmean[i_temporada:f_temporada]

    if hr is None:
        HR = oran[i_ci:f_ci,4]
    else:
        HR = hr[i_temporada:f_temporada]

    if rain is None:
        Rain = oran[i_ci:f_ci,3]
    else:
        Rain = rain[i_temporada:f_temporada]

    if tmin is None:
        TMIN = oran[i_ci:f_ci,0]
    else:
        TMIN = tmin[i_temporada:f_temporada]

    DAYS = np.size(Tmean)

    if ci is None:
        casosIMP = load_data_file('serie_ci_2001_2022.txt')[i_ci:f_ci]
    else:
        casosIMP = cases(ci[0], ci[1], temporada)

    if initial_c is None:
        ED0  = 22876.
        EW0  = 102406
        L0   = 24962.
        P0   = 2003.
        M0   = 28836.
        V0   = 0.
        V_S0 = 0.
        V_E0 = 0.
        V_I0 = 0.
        H_S0 = ALPHA*poblacion
        H_E0 = 0.
        H_I0 = 0.
        H_R0 = poblacion - H_S0   # 25% of the population initially immune
        v = np.array([ED0, EW0, L0, P0, M0, V0, V_S0, V_E0, V_I0,
                      H_S0, H_E0, H_I0, H_R0], dtype=float)
    else:
        v = np.array(initial_c, dtype=float)[:13].copy()

    H_t  = 24. if initial_H is None else float(initial_H)

    dias = DAYS-1
    V_time = np.zeros((dias,13))
    V_time[0,:] = v

    aedes = np.empty(dias)
    aedes[0] = v[5]/poblacion

    G_T = np.empty(dias)
    G_T[0] = nuevos_casos(Tmean[0], TMIN[0], v)

    inicio = np.datetime64(temporada[0])
    fin = np.datetime64(temporada[1])
    fechas = np.arange(inicio, fin + 1, dtype='datetime64[D]')

    años = fechas.astype('datetime64[Y]')
    meses = fechas.astype('datetime64[M]')

    # month and day of each date
    mes_num = (meses - años).astype(int) + 1
    dia_num = (fechas - meses).astype(int) + 1

    julio = np.where((mes_num == 6) & (dia_num == 30))[0]
    julio = julio[0:-1]

    valor = 0
    Kmax = k[valor]

    for t in range(1,dias):
        h = 1.
        G_T[t]	=  nuevos_casos(Tmean[t], TMIN[t], v)

        sigma_V     =	1./(1. + (0.1216*Tmean[int(t)]*Tmean[int(t)] - 8.66*Tmean[int(t)] + 154.79) )

        EV = sigma_V*v[7]
        H_t     = hume(H_t,Rain[int(t)], Tmean[int(t)], HR[int(t)])

        v = runge(modelo,t,h,v,args=(EV,H_t,Tmean,TMIN,Rain,casosIMP,beta_day,Kmax))

        v[v < 0.] = 0.

        V_time[t,:] = v
        aedes[t] = v[5]/poblacion

        if t in julio:
            valor = valor + 1
        Kmax = k[valor]

    labels = [
        "ED","EW","L","P","M","V",
        "V_S","V_E","V_I",
        "H_S","H_E","H_I","H_R"
    ]
    tabla = pd.DataFrame(V_time, columns=labels)

    if devolver_H:
        return G_T,aedes,tabla,H_t
    return G_T,aedes,tabla
