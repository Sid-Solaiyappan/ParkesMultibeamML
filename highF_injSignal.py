# import os
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

h5_filepaths = 'highF_injSignal.txt' #.txt file with .h5 filepaths separated by ",\n"
savedir = 'highF_injSignalBlips' #directory to save images
samples_per_h5 = 50 #number of images to be generated per .h5 file

diff = 6.523e-3 #Difference between fstop and fstart that yields 1954 channels; 6.837e-3 for 2048
n_channels = 1954 

###################################

inj_signal = open(h5_filepaths).readlines()
inj_signal = [h5.split(",\n")[0] for h5 in inj_signal]
sample_num = 0

if not os.path.exists(savedir):
    os.mkdir(savedir)
else:
    pass

for inj_h5 in inj_signal: 
    
#     header = subprocess.check_output(f"watutil -i {inj_h5}",shell=True).decode('utf8').split('\n')

    try:
        header = subprocess.check_output(f"watutil -i /{inj_h5}",shell=True).decode('utf8').split('\n')
        fmin,fmax = float(header[-3].split(' ')[-1]),float(header[-2].split(' ')[-1])
        nreal_tsamps = int(header[-7].split(' ')[-1])
        if nreal_tsamps==20 or nreal_tsamps==19:
            if fmin>1359.5-5 and fmin<1359.5+5 and fmax>1513.5-5 and fmax<1513.5+5: #set to upper half of the freqeuncy band

                for i in range(samples_per_h5):
                    f_start = np.random.uniform(fmin,fmax-diff)
                    f_stop = f_start+diff
                    drift = np.random.uniform(-4,4) #min and max drift rate
                    snr = np.random.uniform(10,40) #min and max signal to noise ratio
                    ind = np.random.uniform(0.2*n_channels,0.8*n_channels) #centered to middle 60% of n_channels 
                    print(f"sample : {sample_num}\nf_start: {f_start}\nf_stop : {f_stop}\ndrift  : {drift}\nsnr    : {snr}\nind    : {ind}\n{inj_h5}")
                    obs = bl.Waterfall(inj_h5,f_start=f_start,f_stop=f_stop)

                    frame = stg.Frame(waterfall = obs)
                    signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(ind),
                                                                drift_rate=drift*u.Hz/u.s),
                                              stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
                                              stg.gaussian_f_profile(width=40*u.Hz),)
                #     frame.bl_plot()
                    data = frame.get_data()
                #     plt.show()
                    sample_num+=1

                    fig = plt.figure(figsize=(1.65, 1.245)) #determined figsize to output image with dimensions (96,128)
                    img = plt.imshow(frame.data,aspect='auto',interpolation='none').make_image(renderer=None, unsampled=False)[0][:,:,:3]
                    im = Image.fromarray(img)#.convert('L')
    #                 print(img.shape)
                    im.save(os.path.join(savedir,'img_{0:06}.png'.format(sample_num)))
        #             np.save("highF_injSignal/img_{0:06}.npy".format(sample_num),data,allow_pickle=True)
            
    except:
        pass
        
#     fmin,fmax = float(header[-3].split(' ')[-1]),float(header[-2].split(' ')[-1])

#     if fmin>1359.5-5 and fmin<1359.5+5 and fmax>1513.5-5 and fmax<1513.5+5:
    
#         for i in range(140):
#             f_start = np.random.uniform(fmin,fmax-diff)
#             f_stop = f_start+diff
#             drift = np.random.uniform(-4,4)
#             snr = np.random.uniform(10,40)
#             ind = np.random.uniform(0.2*1954,0.8*1954)
#             print(f"sample : {sample_num}\nf_start: {f_start}\nf_stop : {f_stop}\ndrift  : {drift}\nsnr    : {snr}\nind    : {ind}\n{inj_h5}")
#             obs = bl.Waterfall(inj_h5,f_start=f_start,f_stop=f_stop)

#             frame = stg.Frame(waterfall = obs)
#     #         fig = plt.figure(figsize=(7, 4))
#             signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(ind),
#                                                         drift_rate=drift*u.Hz/u.s),
#                                       stg.constant_t_profile(level=frame.get_intensity(snr=snr)),
#                                       stg.gaussian_f_profile(width=40*u.Hz),)
#         #     frame.bl_plot()
#             data = frame.get_data()
#         #     plt.show()
#             sample_num+=1
#             img = plt.imshow(frame.data,aspect='auto',interpolation='none').make_image(renderer=None, unsampled=False)[0][:,:,:3]
#             im = Image.fromarray(img)#.convert('L')
#             im.save('highF_injSignal/img_{0:06}.png'.format(sample_num))
# #             np.save("highF_injSignal/img_{0:06}.npy".format(sample_num),data,allow_pickle=True)
            
