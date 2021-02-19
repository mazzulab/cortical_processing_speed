#Misc
import time, os, sys, pdb
from glob import glob
from fnmatch import fnmatch
#Save
import json, h5py
import scipy.io as sio
#Base
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import curve_fit
from scipy import interpolate
#Plot 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
#User
import plotting as usrplt
import util

#Equations for gain fits
def sigmoid(x,beta,h0,mR):
    y = mR / (1 + np.exp(-beta*(x-h0)))
    return (y)

def softRELU(x,R0,a):
    y = R0*np.log(1+np.exp(x/a))
    return (y)

sns.set_style("ticks")
plt.rcParams['ytick.labelsize'] = 18 
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['font.style'] = "normal"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = 'Helvetica'
# plt.rcParams.update({'figure.autolayout': True})
run_thresh = 3
behav_dict = {-1:'ambiguous', 0:'rest',1:'running'}
celltype_dict = {-1:'ambiguous', 0:'inhibitory',1:'excitatory'}

def calculate_transfer_curve_spont(spike_counts, method = 'sigmoid'):
    #How many datapoints do we have
    nPoints = spike_counts.shape[0]

    #Build quantile-quantile plot between distribution of firing rates and assumed normal distribution of inputs
    #draw gaussian distribution of input currents
    input_currents = np.array(sorted(np.random.normal(size=nPoints)))
    sc_sorted = np.array(sorted(spike_counts))
    max_counts = np.max(spike_counts)
    
    #Calculate initial estimate of inflection point
    iBins_at_0spks = np.where(sc_sorted > 0)[0]
    if len(iBins_at_0spks) < 1:
        pos_x = int(nPoints/2)
    else:
        #Start fit right before it goes up from 0
        pos_x = int(iBins_at_0spks[0] + (nPoints-iBins_at_0spks[0])/2)
            
    if method == 'sigmoid':
        #Sigmoid parameter estimates
        p0_est = [2, input_currents[pos_x], max_counts]
        p0_bounds = [[0,-2,1],[5,3,max_counts+max_counts/2]]

        #Fit sigmoid
        params_fit, _ = curve_fit(sigmoid,input_currents,sc_sorted,p0=p0_est,bounds=p0_bounds,method='dogbox',maxfev=1*10**9)
    
        #Slope at inflection point
        gain = params_fit[0]*params_fit[2]/4
        
        #Get cure fit
        yhat = sigmoid(input_currents,*params_fit)
        
    else:
#         input_currents = np.array(sorted(np.random.normal(loc=10,size=nPoints)))
        slope, intercept, r_value, _, _ = st.linregress(input_currents[iBins_at_0spks],sc_sorted[iBins_at_0spks])
        
        xMax = np.max(input_currents)
        #softRELU parameter estimates
        p0_est = [slope, 1]
        p0_bounds = [[0,0],[100,100]]
        
        #Fit softRELU
        params_fit, _ = curve_fit(softRELU,input_currents,sc_sorted,p0=p0_est,bounds=p0_bounds,method='dogbox',maxfev=1*10**9)
#         params_fit, _ = curve_fit(softRELU,input_currents,sc_sorted,method='lm',maxfev=1*10**9)
    
        #Slope
        gain = params_fit[0]/params_fit[1]
        
        #Get cure fit
        yhat = softRELU(input_currents,*params_fit)
        input_currents = input_currents
    
    #Evaluate Fit
    SS_res = np.sum((sc_sorted - yhat)**2)
    SS_tot = np.sum((sc_sorted - np.mean(sc_sorted))**2)
    R2 = 1 - SS_res/SS_tot

    return params_fit, gain, R2, input_currents

def run_gain_analysis_spont(spikecount_xarr, running_mask_spont, units_df, count_thresh=0.25, uID_list=None, pdfdoc=None,method='sigmoid'):
    if uID_list is None:
        uID_list = units_df.index.values
    
    #Get indices per behavior
    indy_dict = pd.DataFrame(running_mask_spont,columns=['behavior']).groupby('behavior').indices
    bSize = 0.5
    
    #Loop over neurons
    nNeurons = len(uID_list)
    params_all = np.zeros((nNeurons,6,2))
    
    for iNeuron, unitID in enumerate(uID_list):
        #Initialize figure
        fig, axes = plt.subplots(1,2,figsize=(7,3),gridspec_kw = {'wspace':0.5})
        plt.autoscale(enable=True,tight=True)
#         plt.tight_layout()
        iCT = int(units_df.loc[unitID]['celltype'])
        celltype = celltype_dict[iCT]
        plt.suptitle('Putative {} neuron in {}, SNR of {:.2f}, ID: {}'.format(celltype,units_df.loc[unitID]['ecephys_structure_acronym'],units_df.loc[unitID]['snr'],unitID),y=0.945,fontsize=12)
        
        #Loop over behavior
        max_counts = []        
        for iR, runstr in enumerate(['rest','run']):
            #Get indices for behavior
            indy = indy_dict[iR]
            
            #Get firing rate data for this neuron
            tmp = np.array(spikecount_xarr.loc[unitID])
            spk_counts_neuron = np.array(tmp[indy])/bSize
            
            #Calculate mean and max FR
            mean_spkcount = np.mean(spk_counts_neuron)
            sc_sorted = np.array(sorted(spk_counts_neuron))
            max_counts.append(np.max(spk_counts_neuron))
            
            #Ensure neuron fires enough
            if (mean_spkcount < count_thresh):
                params_fit = np.zeros(3)*np.nan
                gain = np.nan; R2 = np.nan
                input_currents = np.array(sorted(np.random.normal(size=len(indy))))
            else:
                #Calculate gain
                params_fit, gain, R2, input_currents = calculate_transfer_curve_spont(spk_counts_neuron,method)
                
            #Save fits
            nParams = len(params_fit)
            params_all[iNeuron,:nParams,iR] = [*params_fit]
            params_all[iNeuron,-3:,iR] = [gain, R2, mean_spkcount]

            #Plot
            plot_gain_spont(fig,axes, input_currents, sc_sorted, params_fit, iR, runstr,method)
            
        #If this neuron could not be fit in one of the behavioral conditions, throw it out
        if any(np.isnan(params_all[iNeuron].ravel())):
            params_all[iNeuron] = np.nan

        mc = np.max(max_counts)
        axes[0].set_xlim([0,mc])
#         axes[1].set_xlim([-3,6])
#         axes[1].set_xticks([-3,0,3,6])
        axes[1].set_ylim([0,mc+mc/2])

#         plt.show()
#         pdb.set_trace()
        if pdfdoc is not None:
            pdfdoc.savefig(fig)
        plt.close(fig)
    return params_all
        
def plot_gain_spont(fig, axes, input_currents, sc_sorted, params_fit, iR, runstr,method='sigmoid'):
    
    max_counts = np.max(sc_sorted)
    #Plotting params
    mark = 'o' if iR == 0 else 'X'
    line = '-' if iR == 0 else ':'
    
    ax = axes[0]
    #Plot histogram of firing rates
    ax.hist(sc_sorted,histtype='step',color=usrplt.state_colors[iR],LineWidth=3,density=True)
    
    #Plot distribution of firing rates obtained from passing a standard normal distribution 
    #through the sigmoidal transfer function shown in the next axis
    if all(~np.isnan(params_fit)):
        if method == 'sigmoid':
            sns.kdeplot(sigmoid(input_currents,*params_fit),ax=ax,color=usrplt.cc[8],lw=4,ls=line,zorder=0,alpha=0.75)
            x0 = params_fit[1]
            
            #For plotting line
            m = params_fit[0]*params_fit[2]/4
            y0 = sigmoid(x0,*params_fit)
            
        elif method == 'softRELU':
            sns.kdeplot(softRELU(input_currents,*params_fit),ax=ax,color=usrplt.cc[8],lw=4,ls=line,zorder=0,alpha=0.75)
            R0 = params_fit[0]
            x0 = params_fit[1]
            
            #For plotting line
            m = R0*np.e/(x0+x0*np.e)
            y0 = softRELU(x0,*params_fit)

    #Plot mean spike count for the neuron
    ax.set_xlabel('Firing Rate [spks/s]')#,fontsize=16)
    ax.set_ylabel('Prob. Density')#,fontsize=16)
    ax.set_xlim([0,max_counts])

    #plot relationship between distribution of firing rates and assumed normal distribution of inputs
    ax = axes[1]
    ax.plot(input_currents,sc_sorted,marker=mark,color=usrplt.state_colors[iR],fillstyle='none',ms=5)
    ax.set_xlabel('Input Current [au]')#,fontsize=16)
    ax.set_ylabel('Firing Rate [spks/s]')#,fontsize=16)
    
    #Plot sigmoid fit over points
    x = np.arange(-4,8,0.01);
    if all(~np.isnan(params_fit)):
        #Plot fit over data
        if method == 'sigmoid':
            ax.plot(x,sigmoid(x,*params_fit),color=usrplt.cc[8],lw=3,zorder=1,alpha=0.75,ls=line)
            #Plot slope at inflection point
            ax.plot(x,m*(x-x0)+y0,color=usrplt.cc[5],ls=line,lw=3,zorder=2)
        elif method == 'softRELU':
            ax.plot(x,softRELU(x,*params_fit),color=usrplt.cc[8],lw=3,zorder=3,alpha=0.75,ls=line)
            
#     ax.set_xlim([-4,8])
    ax.set_ylim([0,max_counts+max_counts/2])

    


