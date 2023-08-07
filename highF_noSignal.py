import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import sys
import setigen
from astropy import units as u
import random
import setigen as stg
import blimpy as bl
import matplotlib.pyplot as plt
import subprocess
from PIL import Image

###### Required params #############

h5_filepaths = 'highF_noSignal.txt' #.txt file with .h5 filepaths separated by ",\n"
savedir = 'highF_noSignalBlips'
samples_per_h5 = 50 #number of images to be generated per .h5 file
n_percent_blips = 0 #percentage of dataset to contain blips (0 to 1)

diff = 6.523e-3 #Difference between fstop and fstart that yields 1954 channels; 6.837e-3 for 2048
n_channels = 1954

###################################

no_signal = open(h5_filepaths).readlines()
no_signal = [h5.split(",\n")[0] for h5 in no_signal]
sample_num = 0

def short_signal_t_profile(frame, snr):
    tchans = frame.tchans
    ntime_samples = np.random.randint(1,int(tchans*0.6)) #blip will span maximum 60% of the time channel
    intensity=frame.get_intensity(snr=snr)
    temp_arr = np.arange(tchans)
    start_idx = np.random.randint(tchans - ntime_samples + 1)
    return np.where((start_idx <= temp_arr) & (temp_arr < start_idx + ntime_samples), intensity, 0),ntime_samples

for noise_h5 in no_signal:
    try:
        header = subprocess.check_output(f"watutil -i /{noise_h5}",shell=True).decode('utf8').split('\n')
        fmin,fmax = float(header[-3].split(' ')[-1]),float(header[-2].split(' ')[-1])
        nreal_tsamps = int(header[-7].split(' ')[-1])
        if nreal_tsamps==20 or nreal_tsamps==19:
            if fmin>1359.5-5 and fmin<1359.5+5 and fmax>1513.5-5 and fmax<1513.5+5:  #set to upper half of the freqeuncy band

                for i in range(samples_per_h5):
                    f_start = np.random.uniform(fmin,fmax-diff)
                    f_stop = f_start+diff
                    drift = np.random.uniform(-4,4) #min and max drift rate
                    snr = np.random.uniform(15,40) #min and max signal to noise ratio
                    ind = np.random.uniform(0.2*n_channels,0.8*n_channels) #centered to middle 60% of n_channels 

                    print(f"\nsample : {sample_num}\nf_start: {f_start}\nf_stop : {f_stop}")

                    obs = bl.Waterfall(noise_h5,f_start=f_start,f_stop=f_stop)
                    frame = stg.Frame(waterfall = obs)

                    if np.random.rand()>n_percent_blips: #percentage of non-blips i.e. just background
                        drift = np.random.uniform(-1,1)
                        t_profile, ntime_samps = short_signal_t_profile(frame,snr)
                        signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(ind),drift_rate=drift*u.Hz/u.s),
                                                  t_profile,
                                                  stg.gaussian_f_profile(width=40*u.Hz),)

                        print(f"t_samp : {ntime_samps}")

                    data = frame.get_data()
                    sample_num+=1

                    fig = plt.figure(figsize=(1.65, 1.245)) #determined figsize to output image with dimensions (96,128)
                    img = plt.imshow(frame.data,aspect='auto',interpolation='none').make_image(renderer=None, unsampled=False)[0][:,:,:3]
                    im = Image.fromarray(img)
        #                 print(img.shape)
                    im.save(os.path.join(savedir,'img_{0:06}.png'.format(sample_num)))
    except:
        pass
