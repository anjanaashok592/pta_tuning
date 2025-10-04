#!/usr/bin/env python

from __future__ import division
import matplotlib
import matplotlib.pyplot as plt

import numpy as np, pickle
import math, sys, os, glob, h5py, json
from astropy import units as u
from sklearn.neighbors import KernelDensity

import pint
from pint import toa
from pint import models
from pint.residuals import Residuals
from pint.simulation import make_fake_toas_fromMJDs
pint.logging.setup(sink=sys.stderr, level="ERROR", usecolors=True)

import pta_replicator
from pta_replicator import simulate
from pta_replicator import white_noise
from pta_replicator import red_noise

from plot_utils import make_residual_plot


seed_efac_equad = 10660
seed_red = 19870
seed_gwb = 16666


datadir = '/Users/vigeland/Documents/Research/NANOGrav/nanograv_data/NG20/Data/ng20_v1p0_dmx/'

parfiles = list(np.genfromtxt(datadir + 'parfile_names.txt', dtype='str'))
timfiles = list(np.genfromtxt(datadir + 'timfile_names.txt', dtype='str'))
print(len(parfiles), len(timfiles))

rn_dict_file = '/Users/vigeland/Documents/Research/NANOGrav/nanograv_20yr_gwb/20yr_noisedict_rn+curn-bpl.json'
with open(rn_dict_file, 'r') as f:
    rn_dict = json.load(f)


print('Making simulation with noise only...')

psrs = []

for ii in range(len(parfiles)):

    psr = simulate.load_pulsar(parfiles[ii], timfiles[ii], ephem='DE440')

    # remove red noise from the model (we will add it back later)
    if 'PLRedNoise' in psr.model.components.keys():
        psr.model.remove_component('PLRedNoise')

    # remove Shapiro delay parameters if they are in the binary model
    if 'BinaryDD' in psr.model.components.keys():
        if 'SINI' in psr.model.components['BinaryDD'].params:
            psr.model.components['BinaryDD'].remove_param('SINI')
    
        if 'M2' in psr.model.components['BinaryDD'].params:
            psr.model.components['BinaryDD'].remove_param('M2')

    elif 'BinaryELL1' in psr.model.components.keys():

        if 'SINI' in psr.model.components['BinaryELL1'].params:
            psr.model.components['BinaryELL1'].remove_param('SINI')
    
        if 'M2' in psr.model.components['BinaryELL1'].params:
            psr.model.components['BinaryELL1'].remove_param('M2')
            
    psr.generate_daily_avg_toas(ideal=True)
    
    white_noise.add_measurement_noise(psr, efac=1, seed = seed_efac_equad + ii)
    
    for _ in range(3):
        psr.fit(fitter='downhill')    
    
    red_noise.add_red_noise(psr, log10_amplitude = rn_dict[psr.name + '_red_noise_log10_A'], 
                            spectral_index = rn_dict[psr.name + '_red_noise_gamma'], 
                            components = 30, seed = seed_red + ii)
    
    for _ in range(3):
        psr.fit(fitter='downhill')
        
    simdir = '../data/NG20_irn/'
    if not os.path.isdir(simdir):
        os.mkdir(simdir)

    make_residual_plot(psr, save=True, simdir=simdir)
    
    psr.write_partim(simdir + psr.name + '.par', simdir + psr.name + '.tim', tempo2=False)
    
    psrs.append(psr)

print('Making simulation with GWB...')
red_noise.add_gwb(psrs, log10_amplitude = rn_dict['gw_log10_A'], spectral_index = rn_dict['gw_gamma'], 
                  seed = seed_gwb)

simdir = '../data/NG20_gwb/'
if not os.path.isdir(simdir):
    os.mkdir(simdir)

for psr in psrs:
    
    print('Refitting residuals for {0}...'.format(psr.name))
    
    for _ in range(3):
        try:
            psr.fit(fitter='downhill')
        except:
            print('Downhill fitter didn\'t work for {0}'.format(psr.name))
            print('Trying to fit with gls fitter...'.format(psr.name))
            
            try:
                psr.fit(fitter='gls')
            except:
                print('gls fitter didn\'t work either!')

    make_residual_plot(psr, save=True, simdir=simdir)
    
    psr.write_partim(simdir + psr.name + '.par', simdir + psr.name + '.tim', tempo2=False)

