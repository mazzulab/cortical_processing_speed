#Misc
import pdb,glob,fnmatch
import os, time, datetime
import glob, fnmatch

#Base
import numpy as np
import pandas as pd
import scipy.stats as st
import multiprocessing as mp
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.signal import savgol_filter

#Plot
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

#Decoding
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

#User
import plotting as usrplt
import util

nProcesses = 10
nShuffles = 100
save_plots = True

def decode_labels(X,Y,train_index,test_index,clabels,X_test=None,shuffle=True,classifier='LDA'):
    
    #Copy training index for shuffle decoding
    train_index_sh = train_index.copy()
    np.random.shuffle(train_index_sh)
    
    #Split data into training and test sets
    if X_test is None:
        #Training and test set are from the same time interval
        X_train = X[train_index,:]
        X_test = X[test_index,:]
    else:
        #Training and test set are from the different time intervals
        X_train = X[train_index,:]
        X_test = X_test[test_index,:]
        
    #Get class labels
    Y_train = Y[train_index]
    Y_test = Y[test_index]

    #How many classes are we trying to classify?
    class_labels,nTrials_class = np.unique(Y,return_counts=True)
    nClasses = len(class_labels)

    #Initialize Classifier
    if classifier == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif classifier == 'SVM':
        clf = svm.LinearSVC(penalty='l1',dual=False, max_iter=1E6)
    elif classifier == 'QDA':
        clf = QuadraticDiscriminantAnalysis()
        
    #Luca's decoder 
    if (classifier == 'Euclidean_Dist') | (classifier == 'nearest_neighbor'):
        nTrials_test, nNeurons = X_test.shape
        PSTH_train = np.zeros((nClasses,nNeurons))
        #Calculate PSTH templates from training data
        for iStim, cID in enumerate(class_labels):
            pos = np.where(Y_train == cID)[0]
            PSTH_train[iStim] = np.mean(X_train[pos],axis=0)
        
        Y_hat = np.zeros((nTrials_test,),dtype=int)
        for iTrial in range(nTrials_test):
            if classifier == 'Euclidean_Dist':
                #Predict test data by taking the minimum euclidean distance
                dist = [np.sum((X_test[iTrial] - PSTH_train[iStim])**2) for iStim in range(nClasses)]
                Y_hat[iTrial] =  class_labels[np.argmin(dist)]
            else:
                #Predict test data by taking the maximum correlation between the test population vector and training PSTHs
                Rs = [np.corrcoef(PSTH_train[iStim],X_test[iTrial])[0,1] for iStim in range(nClasses)]
                Y_hat[iTrial] =  class_labels[np.argmax(Rs)]
                
        #Get decoding weights
        #???
        decoding_weights = np.zeros((nClasses, nNeurons))
                
    #All other classifiers
    else:
        #Fit model to the training data
        clf.fit(X_train, Y_train)

        #Predict test data
        Y_hat = clf.predict(X_test)
    
        #Get weights
        decoding_weights = clf.coef_

    #Calculate confusion matrix
    kfold_hits = confusion_matrix(Y_test,Y_hat,labels=clabels)
    nClasses, nNeurons = decoding_weights.shape
    
    ##===== Perform Shuffle decoding =====##
    if shuffle:
        kfold_shf = np.zeros((nShuffles,nClasses,nClasses))
        decoding_weights_shf = np.zeros((nShuffles,nClasses,nNeurons))
        
        #Classify with shuffled dataset
        for iS in range(nShuffles):
            #Shuffle training indices
            np.random.shuffle(train_index_sh)
            Y_train_sh = Y[train_index_sh]

            #Initialize Classifier 
            if classifier == 'LDA':
                clf_shf = LinearDiscriminantAnalysis()
            elif classifier == 'QDA':
                clf_shf = QuadraticDiscriminantAnalysis()
            elif classifier == 'SVM':
                clf_shf = svm.LinearSVC(penalty='l1',dual=False,max_iter=1E6) #C=classifier_kws['C']
            
            #"Luca's" decoder 
            if (classifier == 'Euclidean_Dist') | (classifier == 'nearest_neighbor'):
                nTrials_test, nNeurons = X_test.shape
                PSTH_sh = np.zeros((nClasses,nNeurons))
                
                #Calculate PSTH templates from training data
                for iStim, cID in enumerate(class_labels):
                    pos = np.where(Y_train_sh == cID)[0]
                    PSTH_sh[iStim] = np.mean(X_train[pos],axis=0)

                Y_hat = np.zeros((nTrials_test,),dtype=int)
                for iTrial in range(nTrials_test):
                    if classifier == 'Euclidean_Dist':
                        #Predict test data by taking the minimum euclidean distance
                        dist = [np.sum((X_test[iTrial] - PSTH_sh[iStim])**2) for iStim in range(nClasses)]
                        Y_hat[iTrial] =  class_labels[np.argmin(dist)]
                    else:
                        #Predict test data by taking the maximum correlation between the test population vector and training PSTHs
                        Rs = [np.corrcoef(PSTH_sh[iStim],X_test[iTrial])[0,1] for iStim in range(nClasses)]
                        Y_hat[iTrial] =  class_labels[np.argmax(Rs)]
                    
            #All other classifiers
            else:
                #Fit model to the training data
                clf_shf.fit(X_train, Y_train_sh)

                #Predict test data
                Y_hat = clf_shf.predict(X_test)
                
                #Get weights
                decoding_weights_shf[iS] = clf_shf.coef_
                
            #Calculate confusion matrix
            kfold_shf[iS] = confusion_matrix(Y_test,Y_hat,labels=clabels)
            
        #Calculate z-score of decoding weights
        decoding_weights_m_shf = np.mean(decoding_weights_shf,axis=0)
        decoding_weights_s_shf = np.std(decoding_weights_shf,axis=0)
#         decoding_weights_z = (decoding_weights - decoding_weights_m_shf)/decoding_weights_s_shf
        
        decoding_weights_z = np.divide(decoding_weights - decoding_weights_m_shf, decoding_weights_s_shf, out = np.zeros(decoding_weights.shape,dtype=np.float32),where = decoding_weights_s_shf!=0)
   
    else:
        kfold_shf = np.zeros((nClasses,nClasses))
        
    #Return decoding results
    return kfold_hits, kfold_shf, decoding_weights, decoding_weights_z, decoding_weights_m_shf, decoding_weights_s_shf
    
    
def run_decoding_analysis(spike_counts,trsum,trial_indices,
                          trial_indices_perstim,
                          trial_indices_nomatch=None,
                          match_distributions=True,
                          PlotDir=None,fsuffix=None,classifier='LDA'):
    parallel=False; do_shuffle=True
    
    #Save plots
    if PlotDir is None:
        pdfdoc=None
        plot=False
    else:
        plot = True
        if fsuffix is None:
            fname = 'decoding_suppl-plots.pdf'
        else:
            fname = 'decoding_suppl-plots_{}.pdf'.format(fsuffix)
        pdfdoc = PdfPages(os.path.join(PlotDir,fname))

    #Match distributions
    if match_distributions:
        spike_counts_orig = spike_counts.copy()
        spike_counts, nSpikes_removed,_ = match_population_spikecount_distributions(spike_counts,trial_indices,trial_indices,pdfdoc)

    else:
        nSpikes_removed = 0
        
    #For decoding results
    nClasses = 4; nKfold = 5
    nTrials,nBins,nNeurons = spike_counts.shape
    confusion_mat = np.zeros((2,nKfold,nClasses,nClasses))
    confusion_shf = np.zeros((2,nKfold,nClasses,nClasses))
    confusion_z = np.zeros((2,nKfold,nClasses,nClasses))
    CI95_shf = np.zeros((2,nKfold,2,nClasses,nClasses))
    decoding_weights = np.zeros((2,nKfold,nClasses,nNeurons))
    decoding_weights_z = np.zeros((2,nKfold,nClasses,nNeurons))
    decoding_weights_m_shf = np.zeros((2,nKfold,nClasses,nNeurons))
    decoding_weights_s_shf = np.zeros((2,nKfold,nClasses,nNeurons))
    
    uniq_orientations = np.unique(trsum['orientation']).tolist()
    ##===== Loop over behavioral conditions=====##
    for iR,runstr in enumerate(['rest','run']):
        #Get trial indices for condition
        trial_indices_cond = trial_indices[iR]
        
        #Sum spikes in specific window
        X = np.sum(spike_counts[trial_indices_cond],axis=1)
        
        #Get class labels
        Y = np.array(trsum.iloc[trial_indices_cond]['orientation'].values).astype(int)
        nClasses = len(np.unique(Y))

        #Create cross-validation object
        k_fold = StratifiedKFold(n_splits=nKfold)

        #Run the processes in parallel
        if parallel:
            pool = mp.Pool(processes=min(nProcesses,nKfold))
            processes = []
        results = []

        #Loop over kfolds
        for iK, (train_index, test_index) in enumerate(k_fold.split(trial_indices_cond,Y)):
            if parallel:
                processes.append(pool.apply_async(decode_labels,args=(X,Y,train_index,test_index,uniq_orientations,None,do_shuffle,classifier)))
            else:
                tmp = decode_labels(X,Y,train_index,test_index,uniq_orientations,None,do_shuffle,classifier)
                results.append(tmp)

        #Extract results from parallel kfold processing
        if parallel:
            results = [p.get() for p in processes]
            pool.close()

        #Calculate decoding accuracy per kfold
        for iK,rTuple in enumerate(results):
            kfold_hits = rTuple[0] #size [nClasses x nClasses]
            kfold_shf = rTuple[1]  #size [nShuffles,nClasses x nClasses]
            decoding_weights[iR,iK] = rTuple[2] #nClasses x nNeurons
            decoding_weights_z[iR,iK] = rTuple[3] #nClasses x nNeurons
            decoding_weights_m_shf[iR,iK] = rTuple[4] #nClasses x nNeurons
            decoding_weights_s_shf[iR,iK] = rTuple[5] #nClasses x nNeurons
            
            #Normalize confusion matrix
            confusion_mat[iR,iK] = kfold_hits/np.sum(kfold_hits,axis=1).reshape(-1,1)

            if do_shuffle:
                #Loop through shuffles and normalize
                c_shf = np.zeros((nShuffles,nClasses,nClasses))
                for iS in range(nShuffles):
                    c_shf[iS] = kfold_shf[iS]/np.sum(kfold_shf[iS],axis=1).reshape(-1,1)

                #Calculate z-score for this kfold
                m_shf, s_shf = np.mean(c_shf,axis=0), np.std(c_shf,axis=0)
                confusion_shf[iR,iK] = m_shf
                confusion_z[iR,iK] = (confusion_mat[iR,iK] - m_shf)/s_shf

                #Calculate 95% CI for this kfold
                w = 2.576*s_shf/np.sqrt(nShuffles)
                CI95_shf[iR,iK,0] = m_shf-w
                CI95_shf[iR,iK,1] = m_shf+w
                
            if plot:
                #Get signficance of decoding 
                pvalues_kfold = st.norm.sf(confusion_z[iR,iK])
                
                #Plot shuffle distributions
                title = 'Shuffle Distributions for kfold {}, {} behavioral condition'.format(iK,runstr)
                usrplt.plot_decoding_shuffle(confusion_mat[iR,iK], c_shf, pvalues_kfold, title, pdfdoc)
    
    if plot:
        for iR,runstr in enumerate(['rest','run']):
            #Calculate mean decoding performance over kfolds
            mKfold = np.mean(confusion_mat[iR],axis=0)
            mKfoldz = np.mean(confusion_z[iR],axis=0)

            #Get signficance of decoding 
            mPvalues = st.norm.sf(mKfoldz)

            #Plot shuffle distributions
            title = 'Decoding Performance, {} behavioral condition'.format(runstr)
            usrplt.plot_confusion_matrices(mKfold, mKfoldz, mPvalues, uniq_orientations, title, pdfdoc)
        pdfdoc.close()

    return (confusion_mat,confusion_shf,confusion_z,CI95_shf,nSpikes_removed, decoding_weights,decoding_weights_z, decoding_weights_m_shf, decoding_weights_s_shf)

def match_trials(trsum,target_contrasts):
    #Determine whether there are enough running & rest trials per stimulus to decode properly
    uniq_orientations = np.unique(trsum['orientation']).tolist()
    
    #Match the number of trials across behavioral conditions
    trial_match = []  
    for ori in uniq_orientations:
        print('\n\t{:4.0f}\xb0 Orientation ->'.format(ori),end='\t')
        for iB in [-1,0,1]:
            indy_list = np.concatenate([np.where((trsum['contrast'] == cst) & (trsum['orientation'] == ori) & (trsum['behavior'] == iB))[0] for cst in target_contrasts])
            nTrials_cond = len(indy_list)
            if iB > -1:
                trial_match.append(nTrials_cond)
            print('{}: {:3d} trials,'.format(util.behav_dict[iB],nTrials_cond),end=' ')
            
    mTrials = np.min(trial_match)
    print('\n\t{} trials matched between conditions'.format(mTrials))
    return mTrials
    
def get_trial_indices(trsum,target_contrasts,spike_counts=None,iWindow=slice(500,-1)):
    np.random.seed(1)
    uniq_orientations = np.unique(trsum['orientation']).tolist()
    #Match the number of trials across behavioral conditions
    mTrials = match_trials(trsum,target_contrasts)
    
    trial_indices = []
    trial_indices_nomatch = []
    trial_indices_perstim = []
    for iR,runstr in enumerate(['rest','run']):
        #Concatenate trial indices from different stimuli
        indy_list = []; indy_list_nomatch = []
        for ori in uniq_orientations:
            tmp_indy = np.concatenate([np.where((trsum['contrast'] == cst) & (trsum['orientation'] == ori) & (trsum['behavior'] == iR))[0] for cst in target_contrasts])
            
            #Sort trials based on population spike counts
            if spike_counts is not None:
                #Sort based on number of population spikes
                population_spikes = np.sum(np.sum(spike_counts[tmp_indy,iWindow,:],axis=1),axis=-1)

                #if rest, sort from largest to smallest population spikes
                #if run, sort from smallest to largest population spikes
                if iR == 0:
                    indy2indy = np.argsort(population_spikes)[::-1]
                else:
                    indy2indy = np.argsort(population_spikes)
                    
                #Take the first mTrials
                indy_sorted = tmp_indy[indy2indy]
                indy_list.append(indy_sorted[:mTrials]) 
            else:
                np.random.shuffle(tmp_indy)
                indy_list.append(tmp_indy[:mTrials])
            #Grab all trials
            indy_list_nomatch.append(tmp_indy)
            
        trial_indices.append(np.concatenate(indy_list))
        trial_indices_nomatch.append(np.concatenate(indy_list_nomatch))
        trial_indices_perstim.append(indy_list)
        
    return trial_indices, trial_indices_nomatch, trial_indices_perstim
    
def match_population_spikecount_distributions(spike_counts,trial_indices,trial_indices_plot=None,pdfdoc=None,title=None):
    np.random.seed(1)
    
    nTrials_full, nBins_slice, nNeurons = spike_counts.shape 
    if trial_indices_plot is not None:
        trial_indices_plot = trial_indices
    
    preMatch_param_list = []
    ## Fit population spike count distributions from both behavioral conditions PRE distribution matching ##
    for iR,runstr in enumerate(['Rest','Run']):
        population_spikes = np.sum(np.sum(spike_counts[trial_indices[iR]],axis=1),axis=-1)
#         pdb.set_trace()
        #First fit with a normal distribution
        normal_params = st.norm.fit(population_spikes)
        lognormal_params = st.lognorm.fit(population_spikes,loc=normal_params[0],scale=normal_params[1])
        
        #Save
        preMatch_param_list.append(lognormal_params)

    #Plot before distribution matching
    fig, axes = usrplt.plot_distribution_match(spike_counts,trial_indices_plot,preMatch_param_list)
    fig.suptitle(title)
    ## Match distributions ##
    #Get log normal parameters from rest condition
    params_match = preMatch_param_list[0]

    #Get population spike counts from running condition
    population_spikes_run = np.sum(np.sum(spike_counts[trial_indices[1]],axis=1),axis=-1)
    #Get indices of sorted trials (descending order)
    indy_sort_run = np.argsort(population_spikes_run)[::-1]
#     pdb.set_trace()
    nTrials = len(trial_indices[1])
    #Draw samples from rest distribution
#     population_spikes_dist = st.lognorm.rvs(params_match[0], loc=params_match[1], scale=params_match[2], size=nTrials)
    population_spikes_dist = np.sum(np.sum(spike_counts[trial_indices[0]],axis=1),axis=-1)
    indy_sort_dist = np.argsort(population_spikes_dist)[::-1]
#     pdb.set_trace()
    nTrials = min(len(indy_sort_run),len(indy_sort_dist))
    ## REMOVE SPIKES FROM RUNNING DISTRIBUTION ##
    nSpikes_removed = np.zeros(nTrials)
    for ii in range(nTrials):
        iTrial = indy_sort_run[ii]
        pc_trial = population_spikes_run[iTrial]
        pc_match = population_spikes_dist[indy_sort_dist[ii]]

        if pc_match < pc_trial:
            nSpikes = int(pc_trial - pc_match)

            #Unravel count_matrix
            count_matrix = spike_counts[trial_indices[iR][iTrial]].ravel()

            #Get bins where there is a spike
            pos = np.where(count_matrix >= 1)[0]
            #Shuffle indices
            np.random.shuffle(pos)

            #Delete spikes
            count_matrix[pos[:nSpikes]] = 0
            nSpikes_removed[ii] = nSpikes
#     print('{} spikes removed to match distributions of population spike counts for rest & running'.format(np.sum(nSpikes_removed)))
    
    postMatch_param_list = []
    ## Fit population spike count distributions from both behavioral conditions POST distribution matching ##
    for iR,runstr in enumerate(['Rest','Run']):
        population_spikes = np.sum(np.sum(spike_counts[trial_indices[iR]],axis=1),axis=-1)
        
        #First fit with a normal distribution
        normal_params = st.norm.fit(population_spikes)
        lognormal_params = st.lognorm.fit(population_spikes,loc=normal_params[0],scale=normal_params[1])
        
        #Save
        postMatch_param_list.append(lognormal_params)
    
    #Plot matched distributions
    fig, axes = usrplt.plot_distribution_match(spike_counts,trial_indices,postMatch_param_list,fig, axes)
    
    #Save to pdf
    if pdfdoc is not None:
        pdfdoc.savefig(fig)
        plt.close(fig)

    return spike_counts, nSpikes_removed, fig

def match_population_spikecount_distributions_perstim(spike_counts,trial_indices,trial_indices_plot=None,pdfdoc=None,title=None):
    np.random.seed(1)
    
    nTrials_full, nBins_slice, nNeurons = spike_counts.shape 
    if trial_indices_plot is not None:
        trial_indices_plot = trial_indices
        
    for iStim in range(4):
        preMatch_param_list = []
        ## Fit population spike count distributions from both behavioral conditions PRE distribution matching ##
        for iR,runstr in enumerate(['Rest','Run']):
            population_spikes = np.sum(np.sum(spike_counts[trial_indices[iR][iStim]],axis=1),axis=-1)
    #         pdb.set_trace()
            #First fit with a normal distribution
            normal_params = st.norm.fit(population_spikes)
            lognormal_params = st.lognorm.fit(population_spikes,loc=normal_params[0],scale=normal_params[1])

            #Save
            preMatch_param_list.append(lognormal_params)

        #Plot before distribution matching
        fig, axes = usrplt.plot_distribution_match(spike_counts,[trial_indices[0][iStim],trial_indices[1][iStim]],preMatch_param_list)
        fig.suptitle(title)
        ## Match distributions ##
        #Get log normal parameters from rest condition
        params_match = preMatch_param_list[0]

        #Get population spike counts from running condition
        population_spikes_run = np.sum(np.sum(spike_counts[trial_indices[1][iStim]],axis=1),axis=-1)
        #Get indices of sorted trials (descending order)
        indy_sort_run = np.argsort(population_spikes_run)[::-1]

        nTrials = len(trial_indices[1][iStim])
        #Draw samples from rest distribution
    #     population_spikes_dist = st.lognorm.rvs(params_match[0], loc=params_match[1], scale=params_match[2], size=nTrials)
        population_spikes_dist = np.sum(np.sum(spike_counts[trial_indices[0][iStim]],axis=1),axis=-1)
        indy_sort_dist = np.argsort(population_spikes_dist)[::-1]

        nTrials = min(len(indy_sort_run),len(indy_sort_dist))
        ## REMOVE SPIKES FROM RUNNING DISTRIBUTION ##
        nSpikes_removed = np.zeros(nTrials)
        for ii in range(nTrials):
            iTrial = indy_sort_run[ii]
            pc_trial = population_spikes_run[iTrial]
            pc_match = population_spikes_dist[indy_sort_dist[ii]]

            if pc_match < pc_trial:
                nSpikes = int(pc_trial - pc_match)

                #Unravel count_matrix
                count_matrix = spike_counts[trial_indices[iR][iStim][iTrial]].ravel()

                #Get bins where there is a spike
                pos = np.where(count_matrix >= 1)[0]
                #Shuffle indices
                np.random.shuffle(pos)

                #Delete spikes
                count_matrix[pos[:nSpikes]] = 0
                nSpikes_removed[ii] = nSpikes
#                 pdb.set_trace()
    #     print('{} spikes removed to match distributions of population spike counts for rest & running'.format(np.sum(nSpikes_removed)))

        postMatch_param_list = []
        ## Fit population spike count distributions from both behavioral conditions POST distribution matching ##
        for iR,runstr in enumerate(['Rest','Run']):
            population_spikes = np.sum(np.sum(spike_counts[trial_indices[iR][iStim]],axis=1),axis=-1)

            #First fit with a normal distribution
            normal_params = st.norm.fit(population_spikes)
            lognormal_params = st.lognorm.fit(population_spikes,loc=normal_params[0],scale=normal_params[1])

            #Save
            postMatch_param_list.append(lognormal_params)

        #Plot matched distributions
        fig, axes = usrplt.plot_distribution_match(spike_counts,[trial_indices[0][iStim],trial_indices[1][iStim]],postMatch_param_list,fig, axes)
        
        #Save to pdf
        if pdfdoc is not None:
            pdfdoc.savefig(fig)
            plt.close(fig)

    return spike_counts, nSpikes_removed, fig


def calculate_decoding_speed(decoding_performance_1ms,shf_upperbound_1ms,stop_times_1ms,max_fraction=0.8):

    ##===== Calculate the encoding speed =====##
    #Define interval with transient
    transient_window = np.where((stop_times_1ms > 0) & (stop_times_1ms < 0.75))[0]
    bin_at_transientMax = transient_window[np.argmax(decoding_performance_1ms[transient_window])]
    max_dc_cond = np.max(decoding_performance_1ms[transient_window])
    bin_at_XpercentMax = np.where(decoding_performance_1ms > max_fraction*max_dc_cond)[0][0]
    #Find bins that are above chance
    bins_above_chance = decoding_performance_1ms > (shf_upperbound_1ms)
    
    indy = np.array([]); R2_list = []; param_list = []; indy_list = []
       
    #Make sure it's the transient is actually going up
    for iB,bBool in enumerate(bins_above_chance):
        if iB > bin_at_transientMax: break
#         if (all(bins_above_chance[iB:(iB+25)])) & (stop_times_1ms[iB] > 0):  & (np.sum(bins_above_chance[iB:(iB+50)])/50 > 0.75) & (~bins_above_chance[iB-1])
        if (all(bins_above_chance[iB:(iB+75)]))  & (~bins_above_chance[iB-1]) & (bins_above_chance[iB]): # & (stop_times_1ms[iB] > 0.025): 
            indy = np.arange(iB-1,bin_at_XpercentMax+1)
            if (max_fraction*max_dc_cond < 0.4) or (len(indy) < 5): #(len(indy) < 25) & 
                indy = np.arange(iB-1,bin_at_transientMax+1)

            slope, intercept, r_value, _, _ = st.linregress(stop_times_1ms[indy],decoding_performance_1ms[indy])
            params_line = [slope, intercept]; R2 = r_value**2
            return params_line, indy, R2
                
    if len(indy) == 0:
        params_line = np.zeros((2,1)) #params_line[:] = np.nan
        window = np.zeros((2,1901))
        R2 = 0
        return params_line, window, R2
    
def calculate_decoding_latency(decoding_performance_1ms,shf_upperbound_1ms,stop_times_1ms,mMax_dc_cond,max_fraction=0.8,method='both'):

    #Find bins that are above chance
    bin_at_XpercentMax = np.where(decoding_performance_1ms > max_fraction*mMax_dc_cond)[0][0]
    bins_above_chance = decoding_performance_1ms > (shf_upperbound_1ms)
    
    #Interval with transient
    transient_window = np.where((stop_times_1ms > 0) & (stop_times_1ms < 0.75))[0]
    bin_at_transientMax = transient_window[np.argmax(decoding_performance_1ms[transient_window])]

    firstBin = np.nan

    #Get two measures of latency: 
    #1) When did the decoding trace rise above chance for more than 50ms
    for iB,bBool in enumerate(bins_above_chance):
        if iB > bin_at_transientMax: break
#         if (all(bins_above_chance[iB:(iB+25)])) & (stop_times_1ms[iB] > 0):  & (np.sum(bins_above_chance[iB:(iB+50)])/50 > 0.75)
        if (all(bins_above_chance[iB:(iB+75)]))  & (~bins_above_chance[iB-1]) & (bins_above_chance[iB]): # & (stop_times_1ms[iB] > 0.025):
            firstBin = stop_times_1ms[iB]*1E3
            break

    indy = np.array([])
    if method == 'both':
        #2) Average rise time difference between the decoding transients    
        for iB,bBool in enumerate(bins_above_chance):
            if iB > bin_at_transientMax: break
#             if (all(bins_above_chance[iB:(iB+10)])) & (stop_times_1ms[iB] > 0): & (np.sum(bins_above_chance[iB:(iB+50)])/50 > 0.75) 
            if (all(bins_above_chance[iB:(iB+75)]))  & (~bins_above_chance[iB-1]) & (bins_above_chance[iB]): # & (stop_times_1ms[iB] > 0.025): 
                indy = np.arange(iB,bin_at_XpercentMax+1)
                if len(indy) <= 10:
                    indy = np.arange(iB,bin_at_transientMax+1)
                break

    if len(indy) == 0:
        xy_coords = np.nan
        decIntervals = np.nan
        return decIntervals, xy_coords, firstBin

    #Create a vector of decoding times at which the decoding pecame significant to 80% of the max decoding between the 2 conditions
    decIntervals = np.linspace(stop_times_1ms[iB]-0.0005,stop_times_1ms[bin_at_XpercentMax],100)
    
    #Get the decoding values at these time points
    fn = interpolate.interp1d(stop_times_1ms[indy],decoding_performance_1ms[indy],bounds_error=False,fill_value='extrapolate')
    decoding_values = fn(decIntervals)
    xy_coords = np.array((decIntervals,decoding_values)).T
    return decIntervals, xy_coords, firstBin

def analyze_decoding_timecourse(dc_hits_tc,shf_hits_tc,shf_CI95_tc,start_times,areaname,contrast_str,tWindow=0.1,max_fraction=0.8,plot=True,pdfdoc=None):

    ##===== Increase resolution via interpolation =====##
    stop_times = start_times + tWindow
    tStart = start_times[0]; tStop = start_times[-1]
    start_times_1ms = np.arange(tStart,tStop+0.001,0.001)
    stop_times_1ms = start_times_1ms + tWindow

    #Decoding performance
    fn = interpolate.interp1d(stop_times,dc_hits_tc,axis=0,bounds_error=False,fill_value='extrapolate')
    dc_hits_tc_1ms = fn(stop_times_1ms)
    
    #Shuffle Performance 
    fn = interpolate.interp1d(stop_times,shf_hits_tc,axis=0,bounds_error=False,fill_value='extrapolate')
    shf_hits_tc_1ms = fn(stop_times_1ms)
    
    #95% CI Bounds
    fn = interpolate.interp1d(stop_times,shf_CI95_tc,axis=0,bounds_error=False,fill_value='extrapolate')
    shf_CI95_tc_1ms = fn(stop_times_1ms)
    shf_upperbounds_1ms = shf_CI95_tc_1ms[:,:,1]
    
    ##===== Smooth data =====##
    window_len = 21    
    dc_hits_tc_1ms_orig = dc_hits_tc_1ms.copy()
    for iR,runstr in enumerate(['Rest','Run']):
        shf_upperbounds_1ms[:,iR] = savgol_filter(shf_CI95_tc_1ms[:,iR,1], window_len, 2)
        for iStim in range(dc_hits_tc.shape[-1]):
            dc_hits_tc_1ms[:,iR,iStim] = savgol_filter(dc_hits_tc_1ms[:,iR,iStim], window_len, 2)
            
    #Take mean over orientations to get overall decoding performance
    decoding_performance_1ms = np.mean(dc_hits_tc_1ms,axis=-1)
    
    #Plot decoding timecourse
    if plot:
        fig, axes = usrplt.plot_decoding_session(dc_hits_tc_1ms,shf_hits_tc_1ms,shf_CI95_tc_1ms,stop_times_1ms,areaname,contrast_str)
    
    #===== Calculate and plot speed =====##
    iB2 = None
    fit_handles = []; decoding_speeds = []; decoding_peaks = []
    #Calculate and plot speed
    for iR,runstr in enumerate(['Rest','Run']):
        decoding_peaks.append(np.max(decoding_performance_1ms[:,iR]))
        
        #Calculate decoding speed for each 
        params_line, window, R2 = calculate_decoding_speed(decoding_performance_1ms[:,iR],shf_upperbounds_1ms[:,iR],stop_times_1ms,max_fraction)
        
        #Plot
        if all(params_line):
            decoding_speeds.append(params_line[0])
            if plot:
                tmp_handle = usrplt.plot_speed(axes[1], stop_times_1ms, params_line, window, R2, runstr)
                fit_handles.append(tmp_handle)
        elif not all(params_line):
            decoding_speeds.append(np.nan)
            
    ##===== Calculate and plot latency =====##
    #Get the smaller of the transient decoding peaks
    transient_window = np.where((stop_times_1ms > 0) & (stop_times_1ms < 0.75))[0]
    mMax_dc_cond  = np.min([np.max(decoding_performance_1ms[transient_window,0]),np.max(decoding_performance_1ms[transient_window,1])])
    
    xy_list = []
    decoding_latencies = np.zeros(2)
    firstbin_latencies = np.zeros(2)
    for iR,runstr in enumerate(['Rest','Run']):
        #Calculate latency
        decIntervals, xy_coords, firstbin_latencies[iR] = calculate_decoding_latency(decoding_performance_1ms[:,iR],shf_upperbounds_1ms[:,iR],stop_times_1ms,mMax_dc_cond,max_fraction,'both')
        
        #Second method for calculating latency
        decoding_latencies[iR] = np.mean(decIntervals)*1E3
        xy_list.append(xy_coords)


    #Calculate delta-latency for plotting
    deltalatencies = np.zeros(2)
    #Calculate the simple measure of delta-latency
    deltalatencies[0] = firstbin_latencies[1] - firstbin_latencies[0]
    #Calculate the 2nd order measure of delta-latency
    deltalatencies[1] = decoding_latencies[1] - decoding_latencies[0]
    plot_shaded_region = True
    
    #Plot
    if plot:
        tmp_handle = usrplt.plot_latency(axes[1], xy_list,deltalatencies,firstbin_latencies, max_fraction*mMax_dc_cond,plot_shaded_region)
        fit_handles.append(tmp_handle)
        axes[1].legend(handles=fit_handles,loc=1,framealpha=1,shadow=True);

    if pdfdoc is not None:
        pdfdoc.savefig(fig)
#         plt.close(fig)

    decoding_measures = []
    for iR,runstr in enumerate(['Rest','Run']):
        decoding_measures.append([areaname,contrast_str,runstr,decoding_speeds[iR],firstbin_latencies[iR],decoding_latencies[iR],decoding_peaks[iR]])
    delta_decoding_measures = [contrast_str,*deltalatencies,decoding_peaks[1]-decoding_peaks[0]]
    
    return decoding_measures, delta_decoding_measures, fig