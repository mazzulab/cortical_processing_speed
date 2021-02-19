#Misc
import time, os, sys, pdb
from glob import glob
from fnmatch import fnmatch

#Base
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy import interpolate
import xarray as xr
from sklearn.cluster import KMeans

#Plot 
from matplotlib import pyplot as plt
import plotting as usrplt

#Save
import json, h5py
import scipy.io as sio

run_thresh = 3
behav_dict = {-1:'ambiguous', 0:'rest',1:'running'}


def line(x,m,x0):
    y = m*(x+x0)
    return (y)

def line2(x,m,x0):
    y = m*x + x0
    return (y)

def exp_decay(time, amp, tau):
    return amp*np.exp(-time/tau) 

def autocorr(x,tStart=-1,tEnd=1,bSize=0.01):
    nBins = len(x)
    bin_edges = np.arange(tStart, tEnd, bSize)
    nbins = len(bin_edges)
    max_lag = min(nbins,round(iEnd/binsize))+1

    result = np.correlate(x, x, mode='full')
    result = result[(nBins-K):(nBins+K)]
    return result

def invert_perm(perm):
    inverse = np.zeros(len(perm),dtype=int)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def build_time_window_domain(bin_edges, offsets, callback=None):
    callback = (lambda x: x) if callback is None else callback
    domain = np.tile(bin_edges[None, :], (len(offsets), 1))
    domain += offsets[:, None]
    return callback(domain)

def build_spike_histogram(time_domain, spike_times, unit_ids, dtype=None, binarize=False):

    time_domain = np.array(time_domain)
    unit_ids = np.array(unit_ids)

    tiled_data = np.zeros(
        (time_domain.shape[0], time_domain.shape[1] - 1, unit_ids.size),
        dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype
    )

    starts = time_domain[:, :-1]
    ends = time_domain[:, 1:]

    for ii, unit_id in enumerate(unit_ids):
        data = np.array(spike_times[unit_id])

        start_positions = np.searchsorted(data, starts.flat)
        end_positions = np.searchsorted(data, ends.flat, side="right")
        counts = (end_positions - start_positions)

        tiled_data[:, :, ii].flat = counts > 0 if binarize else counts

    return tiled_data

def get_presentationwise_spike_counts_natural_movies(session, bSize=0.1, 
                                                    stimulus='natural_movie_one_more_repeats', 
                                                    unit_ids=None,
                                                    binarize=False,
                                                    dtype=None,
                                                    large_bin_size_threshold=0.001,
                                                    time_domain_callback=None):
    #The problem with the natural movies is that they break up the stimulus table per image, instead of per movie,
    #making it difficult to use the presentationwise_blah method they have to get a spike_count x-array organized by trial x stimulus x neuron
    #So we're going to do it here.
    # Movie is played at 30Hz, so in a bin-size of 100ms, 3 frames are presented.
    tStart = 0; tEnd = 30.01; #sec, 300 bins
    
    #Get stimulus table per image and find the times for the first and last image in the movie
    presentations_df = session.get_stimulus_table(stimulus)
    uniq_presentation_IDs = presentations_df['stimulus_condition_id'].unique()
    start_scID = uniq_presentation_IDs[0]
    end_scID = uniq_presentation_IDs[-1]
    
    #Get start and end times per movie
    start_times = presentations_df.loc[presentations_df['stimulus_condition_id'] == start_scID]['start_time'].values
    stop_times = presentations_df.loc[presentations_df['stimulus_condition_id'] == end_scID]['stop_time'].values
    nMovie_presentations = len(stop_times)
       
    #Get unit dataframe
    unit_df = session.units
    if unit_ids is None:
        unit_ids = unit_df.index.values
    else:
        unit_df = unit_df.loc[unit_ids]
    nNeurons = len(unit_ids)
    
    #Make bin edges that are separated bSize
    bin_edges = np.arange(tStart, tEnd, bSize) 
    
    #Get actual times of stimulus presentation per trial
    domain = build_time_window_domain(bin_edges, start_times, callback=None)

    #Create an x-array of spike counts per movie based on time domain
    tiled_data = build_spike_histogram(
        domain, session.spike_times, unit_ids, dtype=dtype, binarize=binarize
    )

    tiled_data = xr.DataArray(
        name='spike_counts',
        data=tiled_data,
        coords={
            'movie_presetation_id':np.arange(nMovie_presentations),
            'time_relative_to_stimulus_onset': bin_edges[:-1] + np.diff(bin_edges) / 2,
            'unit_id': unit_ids
        },
        dims=['movie_presetation_id', 'time_relative_to_stimulus_onset', 'unit_id']
    )
    
    return tiled_data, start_times

def get_spikecounts_during_spontaneous_epochs(session,uID_list,bSize=0.5,binarize=False,dtype=None):
    #Get start & end times
    spontaneous_df = session.get_stimulus_table("spontaneous")
    start_times = spontaneous_df['start_time'].values
    stop_times = spontaneous_df['stop_time'].values
    durations = spontaneous_df['duration'].values
    
    #Ensure it was for long enough
    iBlocks = np.where(durations > 5)[0]
    nBlocks = len(iBlocks)
    bSize_run = 0.025
    
    #Get spike times
    spike_times = session.spike_times
    
    spikecount_list = []; timecourse_list = []
    #Loop through spontaneous blocks
    for iEpoch in iBlocks:       
        tStart = start_times[iEpoch]; tStop = stop_times[iEpoch]; duration = tStop-tStart

        #Bin spikes into windows to calculate simple FR vector for each neuron
        bin_edges = np.arange(tStart, tStop, bSize)
        starts = bin_edges[:-1]
        ends = bin_edges[1:]

        tiled_data = np.zeros((bin_edges.shape[0] - 1, len(uID_list)),
            dtype=(np.uint8 if binarize else np.uint16) if dtype is None else dtype)

        #Loop through each neuron
        for ii, unit_id in enumerate(uID_list):
            dTmp = np.array(spike_times[unit_id])
            
            #ignore invalid spike times
            pos = np.where(dTmp > 0)[0]
            data = dTmp[pos]
            
            #Ensure spike times are sorted
            sort_indices = np.argsort(data)

            start_positions = np.searchsorted(data, starts.flat,sorter=sort_indices)
            end_positions = np.searchsorted(data, ends.flat, side="right",sorter=sort_indices)
            counts = (end_positions - start_positions)
#             if any(counts > 100):
#                 pdb.set_trace()
            tiled_data[:, ii].flat = counts > 0 if binarize else counts
               
        #Save matrix to list
        spikecount_list.append(tiled_data.T)
        timecourse_list.append(bin_edges[:-1] + np.diff(bin_edges) / 2)
        
#     print('Spontaneous Epoch Spike Data read in')
    return spikecount_list, timecourse_list

def get_spiketimes_during_spontaneous_epochs(session,uID_list):
    #Get start & end times
    spontaneous_df = session.get_stimulus_table("spontaneous")
    start_times = spontaneous_df['start_time'].values
    stop_times = spontaneous_df['stop_time'].values
    durations = spontaneous_df['duration'].values
    
    #Ensure it was for long enough
    iBlocks = np.where(durations > 5)[0]
    nBlocks = len(iBlocks)

    #Get spike times
    spike_times = session.spike_times
    spiketime_per_epoch = []; timecourse_list = []
    #Loop through spontaneous blocks
    for iEpoch in iBlocks:       
        tStart = start_times[iEpoch]; tStop = stop_times[iEpoch]; duration = tStop-tStart
        print('Epoch {}: {}'.format(iEpoch,duration))
        
        tmp_dict = {}
        spike_times_perneuron = []
        #Loop through each neuron
        for ii, unit_id in enumerate(uID_list):
            dTmp = np.array(spike_times[unit_id])
            
            #ignore invalid spike times
            pos = np.where(dTmp > 0)[0]
            data = dTmp[pos]
            
            #Ensure spike times are sorted
            sort_indices = np.argsort(data)
            
            #Get spike times between start and stop time of this epoch
            pos = np.where((data > tStart) & (data < tStop))[0]
            spike_times_perneuron.append(data[pos])
#             tmp_dict.update(unit_id: data[pos])
               
        #Save matrix to list
        spiketime_per_epoch.append(spike_times_perneuron)
        
#     print('Spontaneous Epoch Spike Data read in')
    return spiketime_per_epoch
    
def get_spiketimes_during_evoked_periods(session,uID_list, tOffset=-1, tEnd=1, stimulus='drifting_gratings_75_repeats'):
    #Get stimulus presentation dataframe 
    stimulus_df = session.get_stimulus_table(stimulus)
    nTrials = stimulus_df.shape[0]

    spike_times = session.spike_times
    spiketime_per_trial = []
    for trial_id in stimulus_df.index.values:
        tStart = stimulus_df.loc[trial_id]['start_time'] + tOffset
        tStop = stimulus_df.loc[trial_id]['start_time'] + tEnd
        
        spike_times_perneuron = []
        #Loop through each neuron
        for ii, unit_id in enumerate(uID_list):
            dTmp = np.array(spike_times[unit_id])
            
            #ignore invalid spike times
            pos = np.where(dTmp > 0)[0]
            data = dTmp[pos]
            
            #Ensure spike times are sorted
            sort_indices = np.argsort(data)
            
            #Get spike times between start and stop time of this epoch
            pos = np.where((data > tStart) & (data < tStop))[0]
            spike_times_perneuron.append(data[pos])
        #Save matrix to list
        spiketime_per_trial.append(spike_times_perneuron)
    return spiketime_per_trial

def get_running_during_spontaneous_epochs(session,session_ID,bSize_spk=0.1,run_thresh=3,plot=True,pdfdoc=None):
    #Get start & end times
    spontaneous_df = session.get_stimulus_table("spontaneous")
    start_times = spontaneous_df['start_time'].values
    stop_times = spontaneous_df['stop_time'].values
    durations = spontaneous_df['duration'].values
    
    #Ensure it was for long enough since there were some spontaneous "epochs" that were as short as 1 second long
    iBlocks = np.where(durations > 5)[0]
    nBlocks = len(iBlocks)
    bSize_run = 0.025
    
    #Locomotion
    run_df = session.running_speed
    running_speed_midpoints = run_df["start_time"] + (run_df["end_time"] - run_df["start_time"]) / 2
    run_df['time'] = running_speed_midpoints
    behavior_list = []; runvec_list = []; nBins_per_epoch = np.zeros((nBlocks,2)); 
    
    behavior_time_list = [[],[]]
    behavior_duration_list = [[],[]]
    behavior_indices_list = [[],[]]
    nBins_tot = 0
    
    timecouse_list = []
    for ii,iEpoch in enumerate(iBlocks):
        #Get running data within start and end times of particular epoch
        tStart = start_times[iEpoch]; tStop = stop_times[iEpoch]
        tmp_df = run_df.loc[(run_df['time'] >= tStart) & (run_df['time'] < tStop)]
    
        #Make bin edges that are separated by 25ms; i.e. 40Hz
        tmpbins = np.arange(tStart, tStop, bSize_run); run_bintimes = tmpbins[:-1] + np.diff(tmpbins) / 2
        tmpbins = np.arange(tStart, tStop, bSize_spk); spk_bintimes = tmpbins[:-1] + np.diff(tmpbins) / 2
        nBins_spk = len(spk_bintimes)
#         timecouse_list.append([run_bintimes,spk_bintimes])

        #Interpolate velocity to the same time intervals used in the count arrays
        fn = interpolate.interp1d(tmp_df['time'].values, tmp_df['velocity'].values,axis=0,bounds_error=False,fill_value='extrapolate')
        running_speed = fn(run_bintimes)
        runvec_list.append(running_speed)
        
        #Break spontaneous period into 1 second windows
        combine_factor = 40 # 0.025*40 = 1s
        nBins_runvec = running_speed.shape[0]
        nBins_filt = np.floor(nBins_runvec/combine_factor).astype(int)
        
        #Find bins where the animal is running/stationary
        running_mask_fullres = np.abs(running_speed) > run_thresh
        running_mask_filtres = np.zeros((nBins_filt),dtype=np.int8)

        for iBool in range(nBins_filt):
            iStart = iBool*combine_factor; iEnd = (iBool+1)*combine_factor
            fBins_animal_running = np.sum(running_mask_fullres[iStart:iEnd])/combine_factor

            #Is the animal running, at rest, or some combination of both in this second
            if fBins_animal_running >= 0.7:
                running_mask_filtres[iBool] = 1
            elif fBins_animal_running <= 0.3:
                running_mask_filtres[iBool] = 0
            else:
                running_mask_filtres[iBool] = -1

        #Fine tuning of running mask
        runningmask = np.zeros((nBins_spk),dtype=np.int8)
        tmp = np.repeat(running_mask_filtres,int(combine_factor/(bSize_spk/bSize_run)),axis=0);nBins_tmp = tmp.shape[0]
        runningmask[:nBins_tmp] = tmp

        behavior_list.append(runningmask)
        behavior_pct = np.array([np.sum(running_mask_filtres == -1), np.sum(running_mask_filtres == 0),np.sum(running_mask_filtres == 1)])*100/nBins_filt
#         print('\ts{} -> Spontaneous Epoch {:2d}, {:5.1f} seconds total->\t ambiguous: {:4.1f}%, Rest: {:4.1f}%, Running: {:4.1f}%'.format(session_ID,iEpoch,durations[iEpoch],*behavior_pct))
        
        #Plot running trace
        if plot:
            running_mask_tmp = np.repeat(running_mask_filtres,int(combine_factor),axis=0)
            xbins = np.arange(nBins_runvec);
            behavior_list2.append(running_mask_tmp)
            fig = usrplt.plot_running_spontaneous_epochs(np.arange(nBins_runvec), running_speed, running_mask_tmp,iEpoch,run_thresh)
            if pdfdoc is not None:
                pdfdoc.savefig(fig)
                plt.close(fig)
        
        for iR in range(2):
            #Sum number of bins per behavior
            nBins_per_epoch[ii,iR] = np.sum(runningmask == iR)
            
            #Block behavior into running and rest periods
            behavior_1hot = np.concatenate(([0],(runningmask == iR).astype(int),[0]))
            behavior_trans = np.diff(behavior_1hot)
            behavior_ends = np.nonzero(behavior_trans == -1)[0]
            behavior_starts = np.nonzero(behavior_trans == +1)[0]
            behavior_durations = behavior_ends - behavior_starts
            indices = np.stack([behavior_starts, behavior_ends],axis=1)

            #Only take periods where the behavior lasts longer than 10 seconds
            indy = np.where(behavior_durations > int(10/bSize_spk))[0]

            behavior_time_list[iR].append(tStart + bSize_spk*indices[indy])
            #Add previous index length since we concatenate all spontaneous epochs together for the spike counts
            #Definitely could be cleaner
            behavior_indices_list[iR].append(nBins_tot + indices[indy])
            behavior_duration_list[iR].append(behavior_durations[indy])
        nBins_tot += runningmask.shape[0]
            
    behavior_indices_list = [np.vstack(behavior_indices_list[iR]) for iR in range(2)]
    behavior_duration_list = [np.hstack(behavior_duration_list[iR]) for iR in range(2)]
    behavior_time_list = [np.vstack(behavior_time_list[iR]) for iR in range(2)]
    
    for iR in range(2):
        #Sort blocks by duration
        sort_indices = np.argsort(behavior_duration_list[iR])[::-1]
        behavior_duration_list[iR] = behavior_duration_list[iR][sort_indices]
        behavior_indices_list[iR] = behavior_indices_list[iR][sort_indices]
        behavior_time_list[iR] = behavior_time_list[iR][sort_indices]


    return runvec_list, behavior_list, behavior_time_list, behavior_duration_list, behavior_indices_list
        
def determine_running_trials(session, tStart=0, tEnd=2.01, bSize=0.025, tOffset=None, tWindow=None, start_times=None,stimulus='drifting_gratings_75_repeats'):
    #Get stimulus presentation dataframe 
    stimulus_df = session.get_stimulus_table(stimulus)
    nTrials = stimulus_df.shape[0]

    #Get running and pupil diameter during stimulus presentation
    running_pw, pupil_diameter = get_presentationwise_behavioralmeasures(session, tStart, tEnd, bSize, start_times, stimulus)
    nTrials, nBins_runvec = running_pw.shape
#     pdb.set_trace()
    #Loop over each trial and classify it as running or rest or ambiguous 
    running_trials = np.zeros((nTrials))
    for iTrial in range(nTrials):
        #Get running data within start and end times of particular trial
        running_speed = running_pw[iTrial]

        #which bins is the animal running faster then run_thresh
        running_mask = np.abs(running_speed) > run_thresh
        
        if tWindow is None:
            #Is the animal running in the first half second of stimulus presentation?
            iStart = 20
            iEnd = 40
        else:
            iStart = int((tWindow+tOffset)/bSize)
        
        #What state is the animal in first half second
        fBins_run_window = np.sum(running_mask[iStart:iEnd])/(iEnd-iStart)
        mSpeed_window = np.mean(np.abs(running_speed[iStart:iEnd]))
        
        #Full stimulation?
        mSpeed_fulltrial = np.mean(np.abs(running_speed))

        #Classify trial as running if animal is running in above threshold in the first half second
        if (fBins_run_window >= 0.7) & (mSpeed_window > run_thresh):
            running_trials[iTrial] = 1
        #Classify trial as a rest trial if animal is initially below threshold and throughout stimulus presentation
        elif (fBins_run_window <= 0.3) & (mSpeed_fulltrial < run_thresh):
            running_trials[iTrial] = 0
        #Otherwise don't use trial
        else:
            running_trials[iTrial] = -1
    
    #Save behavioral classification to stimulus table
    stimulus_df['behavior'] = running_trials
    return stimulus_df, running_pw
    
        
def get_presentationwise_behavioralmeasures(session, tStart, tEnd, bSize=0.025, start_times=None,stimulus='drifting_gratings_75_repeats'):
    ## Inputs:
    # session: allensdk.brain_observatory.ecephys.ecephys_session.EcephysSession object
    # tStart: start time relative to stimulus_onset
    # tEnd: end time relative to stimulus_offset
    # stimulus: name of stimulus we want to get behavioral measures for
    ## Outputs
    # presentations_df: Stimulus dataframe
    # running_speed: x-array of running speed arranged presentationwise 
    # pupil_diameter: x-array of pupil diameter
    ##
    #Make bin edges that are separated by 25ms; i.e. 40Hz
    bin_edges = np.arange(tStart, tEnd, bSize) 

    #Get stimulus presentation dataframe
    stimulus = session.get_stimulus_table(stimulus)

    #Get actual times of stimulus presentation per trial
    if start_times is None:
        start_times = stimulus['start_time'].values
    domain = build_time_window_domain(bin_edges, start_times, callback=None)
    
    nTrials, nBins  = domain.shape
    running_speed = np.zeros((nTrials,nBins-1))
    pupil_diameter = np.zeros((nTrials,nBins-1))
    
    ##===== Behavioral Measures =====##
    #Locomotion
    run_df = session.running_speed
    running_speed_midpoints = run_df["start_time"] + (run_df["end_time"] - run_df["start_time"]) / 2
    run_df['time'] = running_speed_midpoints

#     #Pupil
#     pupil_df = session.get_pupil_data(suppress_pupil_data=False)
#     filtered_pupil_area = pupil_df['filtered_pupil_area'].values
#     pupil_width = pupil_df['pupil_width'].values
#     pupil_time_points = pupil_df.index.values

    #Loop over trials
    for iTrial in range(nTrials):
        spike_bintimes = domain[iTrial,:-1] + np.diff(domain[iTrial]) / 2
        #Get running data within start and end times of particular trial
        tStart = domain[iTrial,0]; tEnd = domain[iTrial,-1]
        tmp_df = run_df.loc[(run_df['time'] >= tStart) & (run_df['time'] < tEnd)]

        #Interpolate velocity to the same time intervals used in the count arrays
        fn = interpolate.interp1d(tmp_df['time'].values, tmp_df['velocity'].values,axis=0,bounds_error=False,fill_value='extrapolate')
        trial_speed = fn(spike_bintimes)
        running_speed[iTrial,:] = trial_speed

#         #Do the same for the pupil diameter
#         indy = (pupil_time_points >= tStart) & (pupil_time_points < tEnd)
        
#         #Interpolate velocity to be 1ms intervals
#         fn = interpolate.interp1d(pupil_time_points[indy], filtered_pupil_area[indy],axis=0,bounds_error=False,fill_value='extrapolate')
#         trial_pupil = fn(spike_bintimes)
#         pupil_diameter[iTrial,:] = trial_pupil
    
    return running_speed, pupil_diameter

def get_putative_celltype(unit_df, nClusters=4,plot=True,pdfdoc=None,session_ID=None):
    ROI_list = ['LD','LGd','LGv','LT','VIS','VISp','VISl','VISrl','LP','VISal','VISpm','VISam','VISli','VISa','VISpor','VISmma','VISmmp']
    #Breat up ROI_list into Cortical and subcortical
    AL_list = [['LD','LGd','LGv','LP','LT'],['VIS','VISa','VISpor','VISp','VISal','VISam','VISl','VISli','VISmma','VISmmp','VISpm','VISrl']]
    areastr_list = ['Subcortical','Cortical']
    
    #Subselect data into only the regions we care about
    unit_df['celltype'] = np.ones(len(unit_df))*-1
    unit_df_subselect = unit_df.loc[(unit_df['waveform_PT_ratio'] < 10) & (unit_df['ecephys_structure_acronym'].isin(ROI_list))]
    
    for AL,AL_str in zip(AL_list,areastr_list):
        #Subselect data
        tmp_df = unit_df_subselect.loc[unit_df_subselect['ecephys_structure_acronym'].isin(AL)]
        unit_IDs = tmp_df.index.values
        if len(tmp_df) < 5:
            continue

        if plot:
            fig, axes = plt.subplots(1,2,figsize=(20,8),gridspec_kw={'width_ratios':[1,1.5]})
            plt.suptitle('s{}: {} Areas'.format(session_ID,AL_str),y=0.95,fontsize=18)
            ax = axes[0]
            ax.set_xlim([0,2])
            ax.hist(tmp_df['waveform_duration'],bins='fd',color=usrplt.cc[8])
            ax.set_xlabel('Waveform Duration')
            ax.set_title('{} neurons, {} areas'.format(len(tmp_df),len(np.unique(tmp_df['ecephys_structure_acronym']))))

        #Cluster waveforms
        X = np.array([tmp_df['waveform_duration']]).T
        kmeans = KMeans(n_clusters=nClusters,n_init=100,max_iter=1000,tol=1E-5).fit(X)
        class_IDs_unsorted = kmeans.labels_
        class_order = invert_perm(np.argsort(kmeans.cluster_centers_.ravel()))
        class_centers = np.sort(kmeans.cluster_centers_.ravel())
        
        #Get colors in order
        class_IDs = np.array([class_order[iClass] for iClass in class_IDs_unsorted])
        #Combine class 1 & 2 into RS neurons
        pos = np.where(class_IDs == 2); class_IDs[pos] = 1
        pos = np.where(class_IDs == 3); class_IDs[pos] = 2
        
        #Check to make sure class 0 is indeed Fast-spiking;
        if class_centers[0] > 0.35:
            pos = np.where(class_IDs == 0); class_IDs[pos] = 1
            
        #Save cell-type 
        unit_df.loc[unit_IDs,'celltype'] = class_IDs
        
        if plot:
            ax = axes[1]
            ax.set_xlim([0,3]);ax.set_ylim([0,2]);
            label_list = ['FS: Putative Inh Neurons','RS-1: Putative Exc Neurons','RS-2: Putative Exc Neurons','4th cluster','5th cluster']
            for iClass in range(nClusters-1):
                indy = np.where(class_IDs == iClass)[0]
                if len(indy) > 1:
                    ax.plot(tmp_df.iloc[indy]['waveform_PT_ratio'],tmp_df.iloc[indy]['waveform_duration'],'o',color=usrplt.cc[iClass],fillstyle='none',ms=5,label=label_list[iClass])
                    ax.hlines(class_centers[iClass],*ax.get_xlim())
            ax.set_xlabel('Waveform Peak:Trough ratio')
            ax.set_ylabel('Waveform Duration (ms)')
            ax.legend(loc=1)
            if pdfdoc is not None:
                pdfdoc.savefig(fig)

    
    if plot:
        fig, ax = plt.subplots(figsize=(16,12))
        label_list = ['FS: Putative Inh Neurons','RS-1: Putative Exc Neurons','RS-2: Putative Exc Neurons','4th cluster','5th cluster']
        for iClass in range(nClusters-1):
            indy = np.where(unit_df['celltype'] == iClass)[0]
            ax.plot(unit_df.iloc[indy]['waveform_PT_ratio'],unit_df.iloc[indy]['waveform_duration'],'o',color=usrplt.cc[iClass],fillstyle='none',ms=5,label=label_list[iClass])
        ax.set_xlabel('Waveform Peak:Trough ratio')
        ax.set_xlim([0,3]);ax.set_ylim([0,2]);
        ax.set_ylabel('Waveform Duration (ms)')
        ax.set_title('s{}: {} neurons, {} areas'.format(session_ID,len(unit_df_subselect),len(np.unique(unit_df_subselect['ecephys_structure_acronym']))))
        ax.legend(loc=1)
        if pdfdoc is not None:
            pdfdoc.savefig(fig)
    #Lump all regular spiking neurons into putative excitatory
    unit_df.loc[unit_df['celltype'] == 2] = 1
    
    
def get_putative_celltype2(unit_df, nClusters=4,plot=True,pdfdoc=None,session_ID=None):
    ROI_list = ['LD','LGd','LGv','LT','VIS','VISp','VISl','VISrl','LP','VISal','VISpm','VISam','VISli','VISa','VISpor','VISmma','VISmmp']
    #Breat up ROI_list into Cortical and subcortical
    AL_list = [['LD','LGd','LGv','LP','LT'],['VIS','VISa','VISpor','VISp','VISal','VISam','VISl','VISli','VISmma','VISmmp','VISpm','VISrl']]
    areastr_list = ['Subcortical','Cortical']
    
    #Subselect data into only the regions we care about
    unit_df['celltype'] = np.ones(len(unit_df))*-1
    unit_df_subselect = unit_df.loc[(unit_df['waveform_PT_ratio'] < 10) & (unit_df['ecephys_structure_acronym'].isin(ROI_list))]
    
    #Subselect data
    tmp_df = unit_df_subselect
    unit_IDs = tmp_df.index.values

    fig, axes = plt.subplots(1,2,figsize=(20,8),gridspec_kw={'width_ratios':[1,1.5]})
    plt.suptitle('s{}'.format(session_ID),y=0.95,fontsize=18)
    ax = axes[0]
    ax.set_xlim([0,2])
    ax.hist(tmp_df['waveform_duration'],bins='fd',color=usrplt.cc[8])
    ax.set_xlabel('Waveform Duration')
    ax.set_title('{} neurons, {} areas'.format(len(tmp_df),len(np.unique(tmp_df['ecephys_structure_acronym']))))

        
    #Cluster waveforms
    X = np.array([tmp_df['waveform_duration']]).T
    if any((tmp_df['waveform_duration'] > 1) & (tmp_df['waveform_PT_ratio'] > 1)):
        nClusters += 1
        
    kmeans = KMeans(n_clusters=nClusters,n_init=100,max_iter=1000,tol=1E-5).fit(X)
    class_IDs_unsorted = kmeans.labels_
    class_order = invert_perm(np.argsort(kmeans.cluster_centers_.ravel()))
    class_centers = np.sort(kmeans.cluster_centers_.ravel())

    #Get colors in order
    class_IDs = np.array([class_order[iClass] for iClass in class_IDs_unsorted])
    #Combine class 1 & 2 into RS neurons
    pos = np.where(class_IDs == 2); class_IDs[pos] = 1
    if nClusters == 4:
        pos = np.where(class_IDs == 3); class_IDs[pos] = 2
    else:
        pos = np.where(class_IDs == 3); class_IDs[pos] = 1
        pos = np.where(class_IDs == 4); class_IDs[pos] = 2

    #Check to make sure class 0 is indeed Fast-spiking;
    if class_centers[0] > 0.35:
        pos = np.where(class_IDs == 0); class_IDs[pos] = 1

    #Save cell-type 
    unit_df.loc[unit_IDs,'celltype'] = class_IDs
#     for iClass in range(nClusters):
#         axes[1].hlines(class_centers[iClass],*ax.get_xlim(),ls='--')
    if plot:
        ax = axes[1]
        ax.grid('on')
        ax.set_xlim([0,3]);ax.set_ylim([0,2]);
        label_list = ['FS: Putative Inh Neurons','RS-1: Putative Exc Neurons','RS-2: Putative Exc Neurons','4th cluster','5th cluster']
        for iClass in range(nClusters-1):
            indy = np.where(class_IDs == iClass)[0]
            if len(indy) > 1:
                ax.plot(tmp_df.iloc[indy]['waveform_PT_ratio'],tmp_df.iloc[indy]['waveform_duration'],'o',color=usrplt.cc[iClass],fillstyle='none',ms=5,label=label_list[iClass])
                ax.hlines(class_centers[iClass],*ax.get_xlim())
        ax.set_xlabel('Waveform Peak:Trough ratio')
        ax.set_ylabel('Waveform Duration (ms)')
        ax.legend(loc=1)
        if pdfdoc is not None:
            pdfdoc.savefig(fig)
            plt.close(fig)

    #Lump all regular spiking neurons into putative excitatory
    unit_df.loc[unit_df['celltype'] == 2] = 1
    


def progprint(iterator,total=None,perline=25,show_times=True):
    times = []
    idx = 0
    if total is not None:
        numdigits = len('%d' % total)
    for thing in iterator:
        prev_time = time.time()
        yield thing
        times.append(time.time() - prev_time)
        sys.stdout.write('.')
        if (idx+1) % perline == 0:
            if show_times:
                avgtime = np.mean(times)
                if total is not None:
                    eta = sec2str(avgtime*(total-(idx+1)))
                    sys.stdout.write((
                        '  [ %%%dd/%%%dd, %%7.2fsec avg, ETA %%s ]\n'
                                % (numdigits,numdigits)) % (idx+1,total,avgtime,eta))
                else:
                    sys.stdout.write('  [ %d done, %7.2fsec avg ]\n' % (idx+1,avgtime))
            else:
                if total is not None:
                    sys.stdout.write(('  [ %%%dd/%%%dd ]\n' % (numdigits,numdigits) ) % (idx+1,total))
                else:
                    sys.stdout.write('  [ %d ]\n' % (idx+1))
        idx += 1
        sys.stdout.flush()
    print('')
    if show_times and len(times) > 0:
        total = sec2str(seconds=np.sum(times))
        print('%7.2fsec avg, %s total\n' % (np.mean(times),total))
        
def sec2str(seconds):
    hours, rem = divmod(seconds,3600)
    minutes, seconds = divmod(rem,60)
    if hours > 0:
        return '%02d:%02d:%02d' % (hours,minutes,round(seconds))
    elif minutes > 0:
        return '%02d:%02d' % (minutes,round(seconds))
    else:
        return '%0.2f' % seconds

