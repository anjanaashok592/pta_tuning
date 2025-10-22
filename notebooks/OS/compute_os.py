import numpy as np
from matplotlib import pyplot as plt
import discovery as ds

from la_forge.core import Core
import os
import defiant
from defiant import OptimalStatistic
from defiant import utils, orf_functions
from defiant import plotting as defplot

from defiant.extra import mdc1_utils
import glob
from enterprise.pulsar import Pulsar as ePulsar

from enterprise.signals import parameter, gp_signals, white_signals, signal_base
import enterprise.signals
from enterprise.signals import selections
from enterprise.signals.selections import Selection
#make noise dictionary

#sarah's noise file
sarahs_noise_file = '/Users/ashokan/osu/sarahspta/20yr_noisedict_rn+curn-bpl.json'
import json
with open(sarahs_noise_file,'r') as f:
    noise = json.load(f)
rn_psrs = noise
#print(rn_psrs)

#Nihan's noise dict for 15yr 
nihans_dict = '/Users/ashokan/osu/sarahspta/15yr_v1_fl_fwn_dict.json'
with open(nihans_dict,'r') as f:
    nihans_noise = json.load(f)

#NG20 real noise dict
ng20_noise_dict = '/Users/ashokan/osu/sarahspta/ng20_v1p1_dmx_noise_dict.json'
#from https://drive.google.com/drive/folders/1GQXnpyRglqPHwgQs6vE8VK-6VoL9WbSj
with open(ng20_noise_dict,'r') as f:
    ng20_noise_real = json.load(f)

#make a new noise dict with only chime and vegas pulsars
chime_vegas_noise_dict = {}
for k, v in ng20_noise_real.items():
    new_key = k.replace("t2equad", "equad")
    if "CHIME" in k or "YUPPI" in k or "VEGAS" in k or "J0125-2327" in k or "J0154+1833" in k or "J0614-3329" in k or "J0621+2514" in k or "J0732+2314" in k or "J0751+1807" in k or "J1022+1001" in k or "J1803+1358" in k or "J2022+2534" in k or "J2039-3616" in k or "J2150-0326" in k:
        print(k)
        chime_vegas_noise_dict[new_key] = v

#update nihans noise dict with chime and vegas pulsars from ng20
nihans_noise.update(chime_vegas_noise_dict)


#add sarah's red noise to nihans dict
for key, val in rn_psrs.items():
    if key.endswith("_red_noise_log10_A") or key.endswith("_red_noise_gamma"):
        nihans_noise[key] = val


#add gwb
nihans_noise["gw_log10_A"] = np.log10(6.4e-15)
nihans_noise["gw_gamma"] = 3.2

def setup_pta(ePsrs, noise_dict_fixed):
        # Timing model
    tm = gp_signals.TimingModel()
    efac = parameter.Constant()
    equad = parameter.Constant()
    # White noise (EFAC/EQUAD per backend)
    selection = selections.Selection(selections.by_backend)
    ef = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad, selection=selection)

    # Red noise 
    log10_A_IRN = parameter.Uniform(-18., -11.)
    gamma_IRN = parameter.Uniform(0., 7.)
    pl = enterprise.signals.utils.powerlaw(log10_A=log10_A_IRN, gamma=gamma_IRN)
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

    # GWB 
    log10_Agw = parameter.Uniform(-18., -11.)('gw_log10_A')
    gamma_gw = parameter.Uniform(0., 7.)('gw_gamma')
    orf = enterprise.signals.utils.hd_orf()
    crn = gp_signals.FourierBasisCommonGP(pl, orf, components=30, name='gw', Tspan=Tspan)

    # Combine signals
    s = ef + rn + tm + crn
    pta = signal_base.PTA([s(p) for p in ePsrs])

    pta.set_default_params(noise_dict_fixed)
    print(pta.summary())
    return pta

combined_file = './parameters_dict.json'
with open(combined_file, 'w') as f:
    json.dump(nihans_noise, f, indent=2)

combined_noise_file = './parameters_dict.json'
with open(combined_noise_file, 'r') as f:
    noise = json.load(f)
noise_dict_fixed = {
    k.replace('_log10_equad', '_log10_t2equad'): v
    for k, v in noise.items()
}

with open(combined_file, 'r') as f:
    params_dict = json.load(f)

# File to save SNRs
output_file = './snr_values.txt'
for i in range(100):
    data_main = '/Users/ashokan/osu/sarahspta/ng20bridgesims/NG20v1p1SimData/data/NG20v1p1wGWB/'
    rlnName = 'realization_'+str(i)+'/'
    data = os.path.join(data_main, rlnName)
    pardir = data
    timdir = data
    pars = sorted(glob.glob(pardir+'*.par'))
    tims = sorted(glob.glob(timdir+'*.tim'))


    ePsrs = []
    for par,tim in zip(pars,tims):
        ePsr = ePulsar(par, tim,  ephem='DE440')
        ePsrs.append(ePsr)
        print('\rPSR {0} complete'.format(ePsr.name),end='',flush=True)


    # find the maximum time span to set GW frequency sampling
    tmin = [p.toas.min() for p in ePsrs]
    tmax = [p.toas.max() for p in ePsrs]
    Tspan = np.max(tmax) - np.min(tmin)

    pta = setup_pta(ePsrs, noise_dict_fixed)
    OS_obj = OptimalStatistic(ePsrs,pta=pta,gwb_name='gw')
    OS_obj.set_orf(orfs=['hd'])
    a = OS_obj.compute_OS(params=params_dict,return_pair_vals=False)

    snr  = a['A2']/a['A2s']

    with open(output_file, 'a') as f:
        f.write(f'{snr}\n')