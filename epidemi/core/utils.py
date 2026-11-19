"""Constants, helper functions and data loading for the dengue model.

Used by `utils_changed_model.py`, which contains the model equations and `change_k`.
Rates are per day and temperatures in degC.
"""
import numpy as np
import importlib.resources as pkg_resources


def load_data_file(filename):
    """Loads a whitespace-separated file of the package data (one header row)."""
    with pkg_resources.files('epidemi.data').joinpath(filename).open('r') as f:
        return np.loadtxt(f, skiprows=1)


# ---------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------
# transmission
MIObh         = 0.75   # probability of human infection per infectious bite
MIObv         = 0.75   # probability of vector infection per bite on an infected human
bite_rate = 0.27       # bites per mosquito per day
# egg life [days]
EGG_LIFE            = 120.
EGG_LIFE_wet        = 90.
mu_Dry              = 1./EGG_LIFE
mu_Wet              = 1./EGG_LIFE_wet
ALPHA               = 0.75   # initially susceptible fraction of the population
Remove_infect       = 7.     # human infectious period [days]
Remove_expose       = 5.     # human incubation period [days]
MATAR_VECTORES      = 12.5   # Tmin threshold for adult cold mortality
NO_INFECCION        = 15.    # Tmin threshold below which there is no transmission
Temp_ACUATICA       = 10.    # Tmin threshold for cold loss of aquatic stages
RATE_CASOS_IMP      = 1.     # factor of the imported-infection series
# mosquitoes [1/day]
MU_MOSQUITO_JOVEN   = 1./2.  # mortality of emerging mosquitoes
MADURACION_MOSQUITO = 1./5.  # maturation of emerging mosquitoes
MU_MOSQUITA_ADULTA  = 1./10. # adult mortality at 22 degC
# breeding sites
Rthres              = 12.5   # rainfall [mm] at which egg wetting is half its maximum
Hmax                = 24.    # maximum water level of breeding sites [mm]
kmmC                = 3.3E-6 # evaporation coefficient
# San Ramón de la Nueva Orán
hogares = 17633              # households
poblacion = 75697            # population

# Observed daily weather 2001-2022: columns tmin, tmax, tmean, rain, hr
oran = load_data_file('ORAN_2001_2022.txt')


# ---------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------
def egg_wet(rain):
    """Wetting rate of dry eggs [1/day]: Hill function of daily rainfall, maximum 0.8."""
    salida = 0.
    lluvia   = rain / Rthres
    salida = 0.8*lluvia**5/(1. + lluvia**5)

    return salida


def C_Gillet(Larva,KL):
    """Reduction of egg hatching by larval density: 1 - L/KL (0 if L >= KL)."""
    salida = 0.
    if (Larva < KL):
        salida = 1. - Larva/KL
    else:
        salida = 0.

    return salida


def hume(H_t,rain,Tm,Hum):
    """Water level of breeding sites on the next day [mm].

    Adds the rainfall and subtracts evaporation kmmC*(100 - Hum)*(25 + Tm)^2, where
    Hum is the relative humidity; the result is kept between 0 and Hmax.
    """
    salida = H_t + rain - kmmC*(100. - Hum)*(25. + Tm)*(25. + Tm)

    if ( salida < 0.):
        salida = 0.

    if (Hmax < salida):
        salida = Hmax

    return salida


def theta_T(Tm):
    """Temperature dependence of oviposition and biting activity (0 outside 11.7-37.2 degC)."""
    salida = 0.

    if ( (11.7 < Tm) and (Tm < 37.2)) :
        salida = 0.1137*(-5.4 + 1.8*Tm - 0.2124*Tm*Tm + 0.01015*Tm*Tm*Tm - 0.0001515*Tm*Tm*Tm*Tm);

    return salida


def rate_mx(Tm,DHA,DHH,T12):
    """Development rate (Sharpe-Schoolfield) of eggs, larvae and pupae at temperature Tm."""
    salida = 0.
    aux0 = (DHA/1.987207) * (1./298. - 1./(Tm + 273.15) )
    aux1 = (DHH/1.987207) * (1./T12 - 1./(Tm + 273.15) )
    salida = ( (Tm + 273.15)/298. ) * ( np.exp( aux0 )  / (1.+ np.exp(aux1)) )

    return salida


def muerte_V(Tm):
    """Adult mortality at temperature Tm relative to 22 degC (polynomial fit)."""
    salida = 0.
    factor = 0.0360827 # value at 22 degC
    salida = 8.692E-1 -(1.59E-1)*Tm + (1.116E-2)*Tm*Tm -(3.408E-4)*Tm*Tm*Tm + (3.809E-6)*Tm*Tm*Tm*Tm;
    salida = salida/factor

    return salida


def runge(f,t,h,X,args=()):
    """One fourth-order Runge-Kutta step of size h for dX/dt = f(X, t, *args)."""
    n = len(X)
    salida = np.zeros(n)

    k1 = np.zeros(n)
    k2 = np.zeros(n)
    k3 = np.zeros(n)
    k4 = np.zeros(n)

    k1 = h*f(X, t,*args)
    k2 = h*f(X + 0.5*k1, t + 0.5*h,*args)
    k3 = h*f(X + 0.5*k2, t + 0.5*h,*args)
    k4 = h*f(X + k3, t + h,*args)

    salida = X + (1./6.)*( k1 + 2.*k2 + 2.*k3 + k4 )

    return salida


def cases(date, amount, season):
    """Daily series of imported infections with a single entry `amount` on `date`.

    `season` is (start date, end date) of the run.
    """
    u = np.datetime64(season[0])
    v = np.datetime64(season[1])
    z = (v-u).astype(int) + 1

    enter = (np.datetime64(date)-u).astype(int)

    aux = np.zeros(z)
    aux[enter] = amount
    return aux
