import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
plt.style.use('default')
import pandas as pd
# plt.ioff()
import keras #Deep learning library using Te
from keras.preprocessing.image import ImageDataGenerator

#Graph loss vs accuracy over epochs
# from livelossplot import PlotLossesKeras
from tensorflow.keras.callbacks import ModelCheckpoint


import matplotlib
matplotlib.use('agg')
import logging; logging.disable(logging.CRITICAL);
from astropy.time import Time

import subprocess
#BL imports
import blimpy as bl
from blimpy.utils import rebin
from datetime import datetime
import pickle
import h5py
import sys

#preliminary plot arguments
fontsize=12
font = {'family' : 'DejaVu Sans',
'size' : fontsize}
MAX_IMSHOW_POINTS = (4096, 4096)

directory = sys.argv[1].split('/')[0]
savedir = os.path.join(directory,sys.argv[1].split('.pkl')[0].split('/')[1])
scaler = False
# dirs = [ 'blpd10_11_direcs', 'datag_direcs', 'blpd18_2_4_7_9_direcs', 'blpd15_16_direcs',
#          'blpd12_direcs', 'blpd14_direcs','blpd13_direcs','new_pointings']
# savedir = 'MLonBeams20kDataGenS_datag_blpd18_2_4_7_9_10_11_direcs'


#def don't change
run_name = 'resnet50train20k100e'
# strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
model = keras.models.load_model(f'cnnRuns/{run_name}.keras')

if not os.path.exists(savedir):
    os.mkdir(savedir)
else:
    print('Directory already exists')
    
if scaler:
    tree = pickle.load(open('real2trainDist.pkl', 'rb'))#comment out

beam_configs = ['0000000000001', '0000000000010', '0000000000011', '0000000000100', '0000000000101', 
                '0000000000110', '0000000000111', '0000000001000', '0000000001001', '0000000001100', 
                '0000000001101', '0000000001111', '0000000010000', '0000000010001', '0000000011000', 
                '0000000011001', '0000000011101', '0000000100000', '0000000100001', '0000000110000', 
                '0000000110001', '0000000111001', '0000001000000', '0000001000001', '0000001000010', 
                '0000001000011', '0000001000111', '0000001100000', '0000001100001', '0000001100011', 
                '0000001110001', '0000010000000', '0000010000010', '0000011000000', '0000011000010', 
                '0000011000011', '0000100000000', '0000100000010', '0000100000100', '0000100000110', 
                '0000100000111', '0000110000010', '0001000000000', '0001000000100', '0001000001000', 
                '0001000001100', '0001000001101', '0001100000100', '0010000000000', '0010000001000', 
                '0010000010000', '0010000011000', '0010000011001', '0011000001000', '0100000000000', 
                '0100000010000', '0100000100000', '0100000110000', '0100000110001', '0110000010000']

def overlay_drift(f_event, f_start, f_stop, drift_rate, t_duration, offset=10):
    '''creates a dashed red line at the recorded frequency and drift rate of 
    the plotted event - can overlay the signal exactly or be offset by 
    some amount (offset can be 0 or 'auto')
    '''
    #determines automatic offset and plots offset lines
    if offset == 'auto':
        offset = ((f_start - f_stop) / 10)
        plt.plot((f_event - offset, f_event),
                 (10, 10),
                 "o-",
                 c='#cc0000',
                 lw=2)

    #plots drift overlay line, with offset if desired
    plt.plot((f_event + offset, f_event + drift_rate/1e6 * t_duration + offset),
             (0, t_duration),
             c='#cc0000',
             ls='dashed', lw=2)


metadf = pd.DataFrame(columns=['Top_Hit_No', 'Drift_Rate', 'SNR', 'Uncorrected_frequency',
       'Corrected_Frequency', 'Index', 'freq_start', 'freq_end', 'SEFD',
       'SEFD_freq', 'Coarse_Channel_No', 'Full_No_of_Hits', 'fch1', 'Beam No',
       'Source Name', 'Datfile Location', 'Filterbank Location',
       'turboSETI_bc', 'nodes_file','Candidate Location','ML_bc','Timestamp','Probabilities','Time Samples'])

file = sys.argv[1]
data=pd.read_pickle(file)
freq_list=data.iloc[:]["Corrected_Frequency"].values
freq_start_list=data.iloc[:]["freq_start"].values
freq_end_list=data.iloc[:]["freq_end"].values
#beam_no_list=data.iloc[:]["Beam No"].values

SNR_list=data.iloc[:]["SNR"].values

# fch1_string=file.split('fch1_')[1]
# fch1_string=fch1_string.split('_cands')[0]

drate_list=data.iloc[:]["Drift_Rate"].values

get_file_name=file.split('_cands_')[0]
strings_list=data['turboSETI_bc'].values
# print ('strings_list', strings_list) 
#for index, filterbank_main in enumerate(filterbank_loc):
#filterbank_list=[]
#print ('filterbank_main', filterbank_main)
#filterbank_clean=filterbank_main.split('_.fil')[0]
#filterbank_clean=filterbank_clean.split(source_name)[0]
#print ('filterbank_clean', filterbank_clean)
#beam_no_og=beam_no_list[index]
#print ('beam_no_og', beam_no_og)
#nodes_file=file.split('_cands_')[0]
#print ('nodes_file', nodes_file)
# nodes_file=get_file_name+'_all_nodes_list.npy' 
# print ('nodes_file', nodes_file)
# filterbank_list=np.load(nodes_file)

#for beams in beam_no_all:
#    #if str(beam_no_og) != other_beams:
#    other_fils=filterbank_clean+ source_name + '_' + beams + '.fil'
#    print ('other_fils', other_fils)
#    filterbank_list.append(other_fils)
#for index, row in enumerate(data):
print ('freq_list', freq_list)
print ('freq_list', freq_list)

for index_freq, freq in enumerate(freq_list):
    try:

        nodes_file = data.iloc[index_freq]['nodes_file']
        filterbank_list=np.load(nodes_file)
        
        for i,filterbank in enumerate(filterbank_list):
            if '/mnt_blpd14/datax/' in filterbank:
                filterbank_list[i] = os.path.join('/datag/',filterbank.split('/mnt_blpd14/datax/')[1])
            elif '/mnt_blpd15/datax/' in filterbank:
                filterbank_list[i] = os.path.join('/datag/',filterbank.split('/mnt_blpd15/datax/')[1])
            elif '/mnt_blpd16/datax/' in filterbank:
                filterbank_list[i] = os.path.join('/datag/',filterbank.split('/mnt_blpd16/datax/')[1])
            elif '/mnt_blpd16/datax2/' in filterbank:
                filterbank_list[i] = os.path.join('/datag/',filterbank.split('/mnt_blpd16/datax2/')[1])
        
        print ('filterbank_list', filterbank_list)

        fch1_string = str(data.iloc[index_freq]['fch1'])
        filterbank_main=data.iloc[index_freq]["Filterbank Location"]
        source_name=data.iloc[index_freq]["Source Name"]

        print ('freq', freq)
        print ('freq_index', index_freq)
        freq_start=freq_start_list[index_freq]
        freq_stop=freq_end_list[index_freq]
        print ('freq_start', freq_start)
        print ('freq_stop', freq_stop)
        drate=-1*drate_list[index_freq]
        ###should I turn drate negative??? We will see ...
        string=strings_list[index_freq]
        print ('string', string)
        print("Plotting some events for: ", source_name)

        offset = ((freq_start - freq_stop) / 10)

        #calculate the length of the total cadence from the fil files' headers
        #first_fil = bl.Waterfall(filterbank_list[0], load_data=False)
        #tfirst = first_fil.header['tstart']
        #last_fil = bl.Waterfall(filterbank_list[-1], load_data=False)
        #tlast = last_fil.header['tstart']
        #t_elapsed = Time(tlast, format='mjd').unix - Time(tfirst, format='mjd').unix + (last_fil.n_ints_in_file -1) * last_fil.header['tsamp']

        #calculate the width of the plot based on making sure the full drift is visible
        #bandwidth = 2.4 * abs(drate)/1e6 * t_elapsed
        #bandwidth = np.max((bandwidth, 500./1e6))


        #Print useful values
        print('')
        print('*************************************************')
        print('***     The Parameters for This Plot Are:    ****')
        print('Target = ', source_name)
        #print('Bandwidth = ', round(bandwidth, 5), ' MHz')
        print('Frequency = ', round(freq, 4), " MHz")
        print('Expected Drift = ', round(drate, 4), " Hz/s")
        print('*************************************************')
        print('*************************************************')
        print('')

        #prepare for plotting
        matplotlib.rc('font', **font)

        #set up the sub-plots
        #n_plots = len(fil_file_list)
        #fig = plt.subplots(n_plots, sharex=True, sharey=True,figsize=(10, 2*n_plots))
        fig=plt.figure(frameon=False)
        plt.box(False)
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        #fig.patch.set_visible(False)
        ax1 = plt.gca()
        ax1.get_xaxis().get_major_formatter().set_useOffset(False)

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #ax1=plt.axes()
        #ax1=fig.add_axes([0.1, 0.1, 0.8, 0.4])
        #ax1.set_frame_on(False)
        #ax1.axis('off')
        #ax1.patch.set_visible(False)            
        #ax = plt.gca()
        #plt.text(0.03, 0.8, 'new', transform=ax1.transAxes, bbox=dict(facecolor='white'))

        beam_img = [];plot_order=[]; tsamp_counts=[]

        for index_fil, fil in enumerate(filterbank_list):
            print ('fil', fil)

            f_start=freq_start 
            f_stop=freq_stop
            h5 = h5py.File(filterbank_list[index_fil], 'r')
            header = dict(h5['data'].attrs.items())
            real_beam = int(header['ibeam'])
            t0 = header['tstart']

            if header['foff'] < 0: # descending frequency values
                minfreq = freq_start - header['foff']
                maxfreq = freq_stop
            else: # ascending frequency values
                minfreq = freq_start
                maxfreq = freq_stop - header['foff']

            start_ind = int(np.abs((header['fch1']-minfreq)/header['foff']))
            stop_ind = int(np.abs((header['fch1']-maxfreq)/header['foff']))
            plot_data1 = np.squeeze(h5['data'][:,:,start_ind:stop_ind])
            
            tsamp_counts.append(h5['data'].shape[0])

            h5.close()                    

            print ('real_beam', real_beam)           
            plot_order.append(real_beam)

            #rebin data to plot correctly with fewer points
            dec_fac_x, dec_fac_y = 1, 1
            if plot_data1.shape[0] > MAX_IMSHOW_POINTS[0]:
                dec_fac_x = plot_data1.shape[0] / MAX_IMSHOW_POINTS[0]
            try: 
                if plot_data1.shape[1] > MAX_IMSHOW_POINTS[1]:
                    dec_fac_y =  int(np.ceil(plot_data1.shape[1] /  MAX_IMSHOW_POINTS[1]))
            except IndexError:
                print ('pass')
                continue
            except ValueError:
                print ('pass')
                continue

            plot_data1 = rebin(plot_data1, dec_fac_x, dec_fac_y)

            #define more plot parameters
            delta_f = 0.000250
            #mid_f = np.abs(f_start+f_stop)/2.

            try:
                vmin = plot_data1.min()
            except IndexError:
                print ('pass')
                continue
            except ValueError:
                print ('pass')
                continue

            try:
                vmax = plot_data1.max()
            except IndexError:
                print ('pass')
                continue
            except ValueError:
                print ('pass')
                continue

            try:
                vmedian = np.median(plot_data1)
            except IndexError:
                print ('pass')
                continue
            except ValueError:
                print ('pass')
                continue

            try:
                vstd=np.std(plot_data1)
            except IndexError:
                print ('pass')
                continue
            except ValueError:
                print ('pass')
                continue


            #prepare font


            #print ('extent', extent)
            plot_data = (plot_data1)
            normalized_plot_data = (plot_data1 - vmin) / (vmax - vmin)

            #real_beam=index_fil +1

            ###1st column
            if real_beam==8:
                ax1=fig.add_axes([0.0, 0.33, 0.2, 0.2])
                this_plot = ax1.imshow(normalized_plot_data,
                aspect='auto',
                rasterized=False,
                interpolation='nearest',)# )
                ax1.set_frame_on(False)
                ax1.axis('off')
    #                             pixel dim is (96,128,3)
    #                             print(this_plot.make_image(renderer=None, unsampled=False)[0][:,:,:3].shape)
    #                             break

                beam_fig = plt.figure(figsize=(1.65, 1.245))
                img = plt.imshow(normalized_plot_data,aspect='auto',rasterized=False,interpolation='nearest',).make_image(renderer=None, unsampled=False)[0][:,:,:3]
                beam_img.append((real_beam,img))
    #             plt.imsave(f'G278.48-5.86_fch1_1361.5_cands/beam{real_beam}.png',img)
                plt.close(beam_fig)

            ###2nd column
            if real_beam==9:
                ax1=fig.add_axes([0.2, 0.0, 0.2, 0.2])
                this_plot = ax1.imshow(normalized_plot_data,
                aspect='auto',
                rasterized=False,
                interpolation='nearest', )
                ax1.set_frame_on(False)
                ax1.axis('off')

                beam_fig = plt.figure(figsize=(1.65, 1.245))
                img = plt.imshow(normalized_plot_data,aspect='auto',rasterized=False,interpolation='nearest',).make_image(renderer=None, unsampled=False)[0][:,:,:3]
                beam_img.append((real_beam,img))
    #             plt.imsave(f'G278.48-5.86_fch1_1361.5_cands/beam{real_beam}.png',img)
                plt.close(beam_fig)

            if real_beam==2:
                ax1=fig.add_axes([0.2, 0.22, 0.2, 0.2])
                this_plot = ax1.imshow(normalized_plot_data,
                aspect='auto',
                rasterized=True,
                interpolation='nearest', )
                ax1.set_frame_on(False)
                ax1.axis('off')

                beam_fig = plt.figure(figsize=(1.65, 1.245))
                img = plt.imshow(normalized_plot_data,aspect='auto',rasterized=False,interpolation='nearest',).make_image(renderer=None, unsampled=False)[0][:,:,:3]
                beam_img.append((real_beam,img))
    #             plt.imsave(f'G278.48-5.86_fch1_1361.5_cands/beam{real_beam}.png',img)
                plt.close(beam_fig)

            if real_beam==7:
                ax1=fig.add_axes([0.2, 0.44, 0.2, 0.2])
                this_plot = plt.imshow(normalized_plot_data,
                aspect='auto',
                rasterized=True,
                interpolation='nearest', )
                ax1.set_frame_on(False)
                ax1.axis('off')

                beam_fig = plt.figure(figsize=(1.65, 1.245))
                img = plt.imshow(normalized_plot_data,aspect='auto',rasterized=False,interpolation='nearest',).make_image(renderer=None, unsampled=False)[0][:,:,:3]
                beam_img.append((real_beam,img))
    #             plt.imsave(f'G278.48-5.86_fch1_1361.5_cands/beam{real_beam}.png',img)
                plt.close(beam_fig)

            if real_beam==13:
                ax1=fig.add_axes([0.2, 0.66, 0.2, 0.2])
                this_plot = plt.imshow(normalized_plot_data,
                aspect='auto',
                rasterized=True,
                interpolation='nearest', )
                ax1.set_frame_on(False)
                ax1.axis('off')

                beam_fig = plt.figure(figsize=(1.65, 1.245))
                img = plt.imshow(normalized_plot_data,aspect='auto',rasterized=False,interpolation='nearest',).make_image(renderer=None, unsampled=False)[0][:,:,:3]
                beam_img.append((real_beam,img))
    #             plt.imsave(f'G278.48-5.86_fch1_1361.5_cands/beam{real_beam}.png',img)
                plt.close(beam_fig)

            ###3rd column
            if real_beam==6:
                ax1=fig.add_axes([0.4, 0.55, 0.2, 0.2])
                this_plot = plt.imshow(normalized_plot_data,
                aspect='auto',
                rasterized=True,
                interpolation='nearest', )
                ax1.set_frame_on(False)
                ax1.axis('off')
                beam_fig = plt.figure(figsize=(1.65, 1.245))
                img = plt.imshow(normalized_plot_data,aspect='auto',rasterized=False,interpolation='nearest',).make_image(renderer=None, unsampled=False)[0][:,:,:3]
                beam_img.append((real_beam,img))
    #             plt.imsave(f'G278.48-5.86_fch1_1361.5_cands/beam{real_beam}.png',img)
                plt.close(beam_fig)

            if real_beam==1:
                ax1=fig.add_axes([0.4, 0.33, 0.2, 0.2])
                this_plot = plt.imshow(normalized_plot_data,
                aspect='auto',
                rasterized=True,
                interpolation='nearest', )
                ax1.set_frame_on(False)
                ax1.axis('off')

                beam_fig = plt.figure(figsize=(1.65, 1.245))
                img = plt.imshow(normalized_plot_data,aspect='auto',rasterized=False,interpolation='nearest',).make_image(renderer=None, unsampled=False)[0][:,:,:3]
                beam_img.append((real_beam,img))
    #             plt.imsave(f'G278.48-5.86_fch1_1361.5_cands/beam{real_beam}.png',img)
                plt.close(beam_fig)

            if real_beam==3:
                ax1=fig.add_axes([0.4, 0.11, 0.2, 0.2])
                this_plot = plt.imshow(normalized_plot_data,
                aspect='auto',
                rasterized=True,
                interpolation='nearest', )
                ax1.set_frame_on(False)
                ax1.axis('off')

                beam_fig = plt.figure(figsize=(1.65, 1.245))
                img = plt.imshow(normalized_plot_data,aspect='auto',rasterized=False,interpolation='nearest',).make_image(renderer=None, unsampled=False)[0][:,:,:3]
                beam_img.append((real_beam,img))
    #             plt.imsave(f'G278.48-5.86_fch1_1361.5_cands/beam{real_beam}.png',img)
                plt.close(beam_fig)

            ###4th column
            if real_beam==10:
                ax1=fig.add_axes([0.6, 0.0, 0.2, 0.2])
                this_plot = plt.imshow(normalized_plot_data,
                aspect='auto',
                rasterized=True,
                interpolation='nearest', )
                ax1.set_frame_on(False)
                ax1.axis('off')

                beam_fig = plt.figure(figsize=(1.65, 1.245))
                img = plt.imshow(normalized_plot_data,aspect='auto',rasterized=False,interpolation='nearest',).make_image(renderer=None, unsampled=False)[0][:,:,:3]
                beam_img.append((real_beam,img))
    #             plt.imsave(f'G278.48-5.86_fch1_1361.5_cands/beam{real_beam}.png',img)
                plt.close(beam_fig)

            if real_beam==4:
                ax1=fig.add_axes([0.6, 0.22, 0.2, 0.2])
                this_plot = plt.imshow(normalized_plot_data,
                aspect='auto',
                rasterized=True,
                interpolation='nearest', )
                ax1.set_frame_on(False)
                ax1.axis('off')

                beam_fig = plt.figure(figsize=(1.65, 1.245))
                img = plt.imshow(normalized_plot_data,aspect='auto',rasterized=False,interpolation='nearest',).make_image(renderer=None, unsampled=False)[0][:,:,:3]
                beam_img.append((real_beam,img))
    #             plt.imsave(f'G278.48-5.86_fch1_1361.5_cands/beam{real_beam}.png',img)
                plt.close(beam_fig)

            if real_beam==5:
                ax1=fig.add_axes([0.6, 0.44, 0.2, 0.2])
                this_plot = plt.imshow(normalized_plot_data,
                aspect='auto',
                rasterized=True,
                interpolation='nearest', )
                ax1.set_frame_on(False)
                ax1.axis('off')

                beam_fig = plt.figure(figsize=(1.65, 1.245))
                img = plt.imshow(normalized_plot_data,aspect='auto',rasterized=False,interpolation='nearest',).make_image(renderer=None, unsampled=False)[0][:,:,:3]
                beam_img.append((real_beam,img))
    #             plt.imsave(f'G278.48-5.86_fch1_1361.5_cands/beam{real_beam}.png',img)
                plt.close(beam_fig)

            if real_beam==12:
                ax1=fig.add_axes([0.6, 0.66, 0.2, 0.2])
                this_plot = plt.imshow(normalized_plot_data,
                aspect='auto',
                rasterized=True,
                interpolation='nearest', )
                ax1.set_frame_on(False)
                ax1.axis('off')

                beam_fig = plt.figure(figsize=(1.65, 1.245))
                img = plt.imshow(normalized_plot_data,aspect='auto',rasterized=False,interpolation='nearest',).make_image(renderer=None, unsampled=False)[0][:,:,:3]
                beam_img.append((real_beam,img))
    #             plt.imsave(f'G278.48-5.86_fch1_1361.5_cands/beam{real_beam}.png',img)
                plt.close(beam_fig)

            ###5th column
            if real_beam==11:
                ax1=fig.add_axes([0.8, 0.33, 0.2, 0.2])
                this_plot = plt.imshow(normalized_plot_data,
                aspect='auto',
                rasterized=True,
                interpolation='nearest', )
                ax1.set_frame_on(False)
                ax1.axis('off')

                beam_fig = plt.figure(figsize=(1.65, 1.245))
                img = plt.imshow(normalized_plot_data,aspect='auto',rasterized=False,interpolation='nearest',).make_image(renderer=None, unsampled=False)[0][:,:,:3]
                beam_img.append((real_beam,img))
    #             plt.imsave(f'G278.48-5.86_fch1_1361.5_cands/beam{real_beam}.png',img)
                plt.close(beam_fig)

        inds = np.argsort([beam for beam,img in beam_img])
        beam_img = [beam_img[ind] for ind in inds]
    #                 test = np.stack([img for beam,img in beam_img])

        if scaler:
            beam_img = [(beam,img/max(np.ndarray.flatten(img))) for beam,img in beam_img]
            beam_img = [(beam,(tree.predict(np.ndarray.flatten(img).reshape(-1,1)).reshape((96,128,3)))) for beam,img in beam_img]

        if not os.path.exists(f'{savedir}/temp'):
            os.mkdir(f'{savedir}/temp')
        else:
            print('Directory already exists')

        [plt.imsave(f'{savedir}/temp/beam{str(beam).zfill(2)}.png',img) for beam,img in beam_img]
        test = tf.keras.utils.image_dataset_from_directory(f'{savedir}/temp/', labels=None, label_mode="int",image_size=(96,128,),shuffle=False)


        proba = model.predict(test)
        pred = np.argmax(proba,1)
        beam_config = ''.join([str(p) for p in pred][::-1])

        if beam_config in beam_configs:
            config = 'good beam configuration'
        else:
            config = 'bad beam configuration'

        for i,on in enumerate(beam_config):
            if on=='1' and string[i]=='1':   
                bbox_kwargs = {'fc': 'g', 'alpha': .5, 'boxstyle': "square"}        
            elif on=='1':
                bbox_kwargs = {'fc': 'y', 'alpha': .5, 'boxstyle': "square"}                    
            elif string[i]=='1':
                bbox_kwargs = {'fc': 'r', 'alpha': .5, 'boxstyle': "square"}                    
            elif on=='0' and string[i]=='0':   
                bbox_kwargs = {'fc': 'w', 'alpha': .5, 'boxstyle': "square"}

            ann_kwargs = {'xycoords': 'axes fraction', 'bbox': bbox_kwargs}    
            beam_axis = fig.axes[plot_order.index(13-i)+1]
            beam_axis.annotate(13-i, xy=(0, 0.9), fontsize=10, **ann_kwargs)

        print ('string', string)
        plot_title = "%s, %s \n $\dot{\\nu}$ = %2.3f Hz/s, ${\\nu}$ = %2.6f MHz" % (source_name, string, drate, freq)
        plot_title+=f'\nPredicted {config}: {beam_config}'
        plt.suptitle(plot_title, fontsize=10)
        plt.margins(0,0)
    #             plt.show()
        savename=source_name+ '_fch1_' + fch1_string + '_dr_' + "{:0.4f}".format(drate) + '_freq_' "{:0.7f}".format(freq_start)+ f"_{config[0]}" + "_f3.png"
        savename= os.path.join(savedir,savename)
        print ('current path', os.getcwd())
        if os.path.isfile(savename):
            savename=source_name + '_fch1_' + fch1_string + '_dr_' + "{:0.4f}".format(drate) + '_freq_' "{:0.7f}".format(freq_start)+ f"_{config[0]}" + "_2nd_f3.png"
            savename= os.path.join(savedir,savename)

        plt.savefig(savename)
        #plt.savefig(source_name + '_dr_' + "{:0.4f}".format(drate) + '_freq_' "{:0.7f}".format(freq_start) + "_f1.png")
        plt.close()

        metadf.loc[len(metadf.index)+1] = data.iloc[index_freq]
        metadf.loc[len(metadf.index),'Candidate Location']=savename    
        metadf.loc[len(metadf.index),'turboSETI_bc']=string
        metadf.loc[len(metadf.index),'ML_bc']=beam_config    
        metadf.loc[len(metadf.index),'Timestamp']=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        metadf.loc[len(metadf.index),'Probabilities'] = str([p[1] for p in proba])
        metadf.loc[len(metadf.index),'Time Samples'] = str(tsamp_counts)
        
        metadf.to_pickle(f"{savedir}/{savedir.split('/')[-1]}.pkl")

    except Exception as Argument:

        f = open(f"{savedir}/ErrorLog{savedir.split('/')[-1]}.txt", "a")

        # writing in the file
        try:
            f.write(f"{fil}, {source_name}"+"\n")
        except:
            pass
        f.write(str(Argument)+'\n')


        # closing the file
        f.close()
        pass