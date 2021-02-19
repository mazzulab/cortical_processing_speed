#Misc
import time, os, sys, pdb
from glob import glob
#Base
import numpy as np
import pandas as pd
import scipy.stats as st
#Plot
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.collections as mcoll

from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

#User
import util
# import decoding as dc
state_colors = ['#808080','#8B5E3C']

color_names=['windows blue',
             'red',
             'amber',
             'faded green',
             'dusty purple',
             'orange',
             'steel blue',
             'pink',
             'greyish',
             'mint',
             'clay',
             'light cyan',
             'forest green',
             'pastel purple',
             'salmon',
             'dark brown',
             'lavender',
             'pale green',
             'dark red',
             'gold',
             'dark teal',
             'rust',
             'fuchsia',
             'pale orange',
             'cobalt blue',
             'mahogany',
             'cloudy blue',
             'dark pastel green',
             'dust',
             'electric lime',
             'fresh green',
             'light eggplant',
             'nasty green']
 
color_palette = sns.xkcd_palette(color_names)
cc = sns.xkcd_palette(color_names)

cc2 = sns.xkcd_palette(['greyish','dark brown'])

#Default Plotting Options
sns.set_style("ticks")

plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

Run_Dict = {0:'Rest',1:'Run'}
Run_Dict_inv = {'Rest': 0, 'Run': 1}
behav_dict = {-1:'ambiguous', 0:'rest',1:'running'}
nOri = 12

def plot_running_spontaneous_epochs(xbins, running_speed, running_mask,iEpoch, run_thresh=3):
    
    nBins_runvec = running_speed.shape[0]; duration = nBins_runvec*0.025
    fig,ax=plt.subplots(figsize=(24,6))
    fit_handles = []
    for iBehavior, lColor in zip([1, 0, -1],['k','r',cc[8]]):
        fit_handles.append(mlines.Line2D([], [], color= lColor,lw=3, ls='-', label=behav_dict[iBehavior]))
        indy = np.where(running_mask == iBehavior)[0]
        periods = np.where(np.diff(indy)!=1)[0]+1
        if len(periods) < 1:
            ax.plot(xbins[indy],running_speed[indy],'-',color=lColor)
        else:
            sub_indy = np.split(indy,periods)
            for indy in sub_indy:
                ax.plot(xbins[indy],running_speed[indy],'-',color=lColor)
    ax.set_title('Running trace during spontaneous epoch {}, {:.1f} seconds total'.format(iEpoch,duration))
    ax.set_xticks([int(nBins_runvec/4),int(nBins_runvec/2),int(3*nBins_runvec/4),nBins_runvec])
    ax.set_xticklabels([int(duration/4),int(duration/2),int(3*duration/4),int(duration)])
    ax.plot([0,nBins_runvec],[run_thresh,run_thresh],color=cc[0])
    ax.plot([0,nBins_runvec],[-run_thresh,-run_thresh],color=cc[0])
    ax.set_xlabel('Time (s)')
    ax.autoscale(tight=True)
    ax.legend(handles=fit_handles,framealpha=1,shadow=True)
    return fig

def plot_running_pertrial(running_pw,SaveDir,fsuffix,run_thresh=2):
  
    #Save running traces
    pdfdoc = PdfPages(os.path.join(SaveDir,'running_speed_{}.pdf'.format(fsuffix)))
    nTrials,nBins_runvec = running_pw.shape
    
    #Loop over each trial and classify it as running or rest or ambiguous 
    for iTrial in range(nTrials):
        #Get running data within start and end times of particular trial
        running_speed = running_pw[iTrial]

        #Is the animal running in the first 1sec of stimulus presentation?
        mSpeed_first500ms = np.mean(np.abs(running_speed[20:40]))
        mSpeed_full2000ms = np.mean(np.abs(running_speed))
        

        fig,ax=plt.subplots(figsize=(24,6))
        ax.set_title('mean speed in first half-second: {:.2f}, mean speed across trial: {:.2f}'.format(mSpeed_first500ms,mSpeed_full2000ms))
        ax.plot(running_speed,'-k')
        ax.set_xticks([0,20,40,60,80,100]); ax.set_xticklabels([-0.5,0,0.5,1,1.5,2])
        ax.plot([0,running_speed.shape[0]],[run_thresh,run_thresh],color=cc[0])
        ax.plot([0,running_speed.shape[0]],[-run_thresh,-run_thresh],color=cc[0]);ax.set_xlabel('Time (s)')
        ax.autoscale(tight=True)

        pdfdoc.savefig(fig)
        plt.close(fig)
    pdfdoc.close()
    
def plot_spike_counts(data_array, running_vec,
                      pupil_vec, time_coords,
                      cbar_label, title, 
                      xlabel='time relative to stimulus offset (s)',
                      ylabel='unit', xtick_step=20, cmap = 'viridis'): 
    
    fig, axes = plt.subplots(3,1,gridspec_kw = {'height_ratios':[10,3,3],'hspace': 0.05},figsize=(8, 16))
    div = make_axes_locatable(axes[0])
    cbar_axis = div.append_axes("right", 0.1, pad=-0.05)

    #Spike count axes
    ax = axes[0]
    img = ax.imshow(
        data_array.T, 
        interpolation='none',
        cmap = cmap,
        aspect = 'auto'
    )
    plt.colorbar(img, cax=cbar_axis)
    nNeurons = data_array.shape[-1]
    cbar_axis.set_ylabel(cbar_label, fontsize=16)

    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')

    reltime = np.array(time_coords)
    ax.set_xticks([])
    ax.set_yticks([0,round(nNeurons/2),nNeurons])
    ax.set_yticklabels([0,round(nNeurons/2),nNeurons],fontsize=12)
    
    ax.set_title(title, fontsize=18,y=1.025)
    
    #Plot running speed
    ax = axes[1]
    ax.plot(running_vec,'-r')
    ax.set_xticks([])
    ax.set_ylabel('running (cm/s)', fontsize=14)
    
    
    #Plot running speed
    ax = axes[2]
    ax.plot(pupil_vec,'-k')
    ax.set_ylabel('pupil area (*min area)', fontsize=14)
    ax.set_xticks(np.arange(0, len(reltime), xtick_step))
    ax.set_xticklabels([f'{mp:1.3f}' for mp in reltime[::xtick_step]], rotation=45,fontsize=12)
    ax.set_xlabel(xlabel, fontsize=14,fontweight='bold')
    
    
    ax = axes[0]
    ax.plot([200,200],[0,200],'--r')
    ax.plot([0,300],[72,72],':w',linewidth=2)
    ax.set_ylim([0,199])
    axes[1].plot([200,200],[running_vec.min(),20],'--r')
    axes[1].set_ylim(ymin=running_vec.min(),ymax=20)
    
    ax = axes[2]
    ax.plot([200,200],[pupil_vec.min(),pupil_vec.max()],'--r')

    ax.set_xticks(np.arange(0,320,25))
    ax.set_xticklabels(np.arange(-2,1.25,.25))
    axes[1].autoscale(tight=True)
    axes[2].autoscale(tight=True)

    return fig, axes

def fill_betweenxy(xy1,xy2,ax,ymax=None, **kwargs):

#     N = len(xy1)
#     Y = np.zeros((2 * N + 2, 2), float)
#     Y[0] = xy2[-1]
#     Y[N+1] = xy2[0]
#     Y[1:N + 1] = xy2
#     Y[N + 2:] = xy1
#     Y[N+2] = xy1[0]
    
#     pdb.set_trace()
    N = len(xy1)
    Y = np.zeros((2 * N + 2, 2), float)
    Y[0] = xy2[0]
    Y[1:N+1] = xy1
    Y[N+1] = xy2[-1]
    Y[N + 2:] = np.flip(xy2,axis=0)
    Y[-1] = xy1[0]
#     Y[N+2] = xy1[0]
    
    if Y is not None:
        pos = np.where(Y[:,1] > ymax)[0]
        Y[pos,1] = ymax
    collection = mcoll.PolyCollection([Y], **kwargs)

    # now update the datalim and autoscale
    ax.dataLim.update_from_data_xy(xy1, ax.ignore_existing_data_limits,
                                     updatex=True, updatey=True)
    ax.ignore_existing_data_limits = False
    ax.dataLim.update_from_data_xy(xy2, ax.ignore_existing_data_limits,
                                     updatex=True, updatey=False)
    ax.add_collection(collection, autolim=False)
    ax.autoscale_view()
    return collection

def plot_decoding_session(dc_hits_tc,shf_hits_tc,shf_CI95_tc,stop_times,areaname,contrast_str):
    
    #Take mean over orientations to get overall decoding performance
    dc_hits_tc_mean = np.mean(dc_hits_tc,axis=-1)
    shf_hits_tc_mean = np.mean(shf_hits_tc,axis=-1) 
    shf_CI95_tc_mean = np.mean(shf_CI95_tc,axis=1)

    #Plot decoding results
    fig,axes = plt.subplots(1,2,figsize=(24,8),gridspec_kw={'width_ratios':(16,8)})
    plt.suptitle('{}, {}-contrast drifting gratings'.format(areaname,contrast_str),fontsize=20)
    axes[0].set_title('Differential decoding performance per behavior')
    axes[1].set_title('Zoomed into initial stimulus onset')
    axes[0].set_xlabel('Right bound of 100ms moving window');axes[1].set_xlabel('Time relative to stimulus onset (s)')
    
    #Legend strings
    mean_strs = [r'$<\vec \mu^{rest}_{i}>_{orientation}$',r'$<\vec \mu^{run}_{i}>_{orientation}$']
    sess_strs = [r'$\vec \mu^{rest}_{i}$',r'$\vec \mu^{run}_{i}$']
#     mean_strs = [r'$<\vec \mu^{rest}_{s}>_{session}$',r'$<\vec \mu^{run}_{s}>_{session}$']
#     sess_strs = [r'$\vec \mu^{rest}_{s}$',r'$\vec \mu^{run}_{s}$']
    nDCcurves = dc_hits_tc.shape[-1]
    #Loop over axes
    for i,ax in enumerate(axes):
        m = 'None'# '.' if i == 1 else
        
        fit_handles = []
        for iR,runstr in enumerate(['Rest','Run']):
            #Plot traces
            ax.plot(stop_times,dc_hits_tc_mean[:,iR],ls='-',marker=m,lw=2,color=cc[iR],label=runstr,zorder=2)
#             ax.plot(stop_times,decoding_performance_1ms[:,iR],ls='--',lw=2,color=cc[iR],label=runstr,zorder=1,alpha = 0.25)
            ax.plot(stop_times,shf_hits_tc_mean[:,iR],ls='--',lw=2,color=cc[8],zorder=0)
            #Create legend handles
            tmp_handle = mlines.Line2D([], [], color=cc[iR],lw=3,alpha=1, ls='-', label='{}'.format(mean_strs[iR]));fit_handles.append(tmp_handle)
            #Plot individual orientations/sessions
            if nDCcurves > 1:
                for iStim in range(nDCcurves): ax.plot(stop_times,dc_hits_tc[:,iR,iStim],color=cc[iR],alpha=0.1,lw=1.5,ls='-',zorder=1)
                tmp_handle = mlines.Line2D([], [], color=cc[iR],lw=1.5,alpha=0.60, ls='-', label='    {}'.format(sess_strs[iR]));fit_handles.append(tmp_handle)

            #Show the mean 95 confidence interval over behavioral conditions
            shf_color = 2 if iR == 0 else 8
            ax.fill_between(stop_times,y1=shf_CI95_tc[:,iR,0],y2=shf_CI95_tc[:,iR,1],color=cc[shf_color],alpha=0.25)
#         #Show the mean 95 confidence interval over behavioral conditions
#         ax.fill_between(stop_times,y1=shf_CI95_tc_mean[:,0],y2=shf_CI95_tc_mean[:,1],color=cc[8],alpha=0.5)
        #Common plot parameters
        ax.set_ylabel('Decoding Performance')
        ax.autoscale(tight=True);ax.set_ylim([0.175,1])
    axes[1].set_xlim([0,0.35])
    axes[0].legend(handles=fit_handles,loc=1,framealpha=1,shadow=True);

    return fig, axes

def plot_speed(ax, stop_times, params_line, window, R2, runstr):
    iR = Run_Dict_inv[runstr]
    speed = util.line2(stop_times,*params_line)
    ax.plot(stop_times[window],speed[window],'--k',lw=3.5,zorder=3)
    tmp_handle = mlines.Line2D([], [], color= cc[iR],lw=3, ls='--', label='{:4s} - - > encoding speed: {:3.1f}, R2: {:.2f}'.format(runstr,params_line[0],R2))
    return tmp_handle

def plot_latency(ax, xy_latencies,deltalatencies,firstbin_latencies,ymax=None,plot_shaded_region=True):

#     for iR in range(2):
#         ax.vlines(firstbin_latencies[iR]*1E-3,*ax.get_ylim(),ls='-.',color=cc[iR])
    if all(~np.isnan(deltalatencies)) & plot_shaded_region:
#         ax.hlines(ymax,*ax.get_xlim(),ls=':',color='k')
        
        #Fill inbetween decoding curves
#         for i in [-1,0]:
#             plt.plot(xy_latencies[0][i,0],xy_latencies[0][i,1],'.k')
#             plt.plot(xy_latencies[1][i,0],xy_latencies[1][i,1],'.k')
        fill_betweenxy(xy_latencies[0],xy_latencies[1],ax,ymax=ymax,color=cc[2],alpha=0.25)
        tmp_handle = mpatches.Patch(color=cc[2],alpha=0.25, label='\u0394 Latency: {:.0f}ms      \u0394FB: {:.0f}ms'.format(deltalatencies[1],deltalatencies[0]))
    else:
        tmp_handle = mpatches.Patch(color=cc[2],alpha=0, label='\u0394 FB: {:.2f}ms'.format(deltalatencies[0]))

    return tmp_handle

def plot_decoding_measures(results_df,anticipation_df,hierarchy_scores,contrast_str='low',title=None):
    
    fit_handles = []
    cst_color = cc[8] if contrast_str == 'low' else 'k'
    fig,axes = plt.subplots(1,4,figsize=(19,6),gridspec_kw = {'wspace': 0.25,'width_ratios': [6,6,6,1]})
    if title is None:
        plt.suptitle('{}-contrast drifting gratings'.format(contrast_str),fontsize=20)
    else:
        plt.suptitle(title,fontsize=20)

    hierarchy_vec = [hierarchy_scores[area] for area in hierarchy_scores if area in np.unique(anticipation_df.index.values)]
    areas_sorted = [area for area in hierarchy_scores if area in np.unique(anticipation_df.index.values)]
    nAreas = len(areas_sorted)
    ls_list = ['--','-']; 
    for iR,runstr in enumerate(['Rest','Run']):
        #Speed
        axes[0].scatter(hierarchy_vec,results_df.loc[results_df['behavior'] == runstr].loc[areas_sorted]['speed'],s=60,c=cc[:nAreas],zorder=3)
        axes[0].plot(hierarchy_vec,results_df.loc[results_df['behavior'] == runstr].loc[areas_sorted]['speed'],color=cst_color,ls=ls_list[iR])
        #Performance
        axes[2].scatter(hierarchy_vec,results_df.loc[results_df['behavior'] == runstr].loc[areas_sorted]['peak'],s=60,c=cc[:nAreas],zorder=3)
        axes[2].plot(hierarchy_vec,results_df.loc[results_df['behavior'] == runstr].loc[areas_sorted]['peak'],color=cst_color,ls=ls_list[iR])
        fit_handles.append(mlines.Line2D([], [], color=cst_color,lw=2, ls=ls_list[iR], label=runstr))
    #Latency
    axes[1].scatter(hierarchy_vec,anticipation_df.loc[areas_sorted]['dLatency'],marker='o',s=60,c=cc[:nAreas],zorder=3)
    axes[1].plot(hierarchy_vec,anticipation_df.loc[areas_sorted]['dLatency'],'-.',color=cst_color)
#     axes[1].scatter(hierarchy_vec,anticipation_df.loc[areas_sorted]['dFirstBin'],marker='X',s=60,c=cc[:nAreas],zorder=3)
#     axes[1].plot(hierarchy_vec,decoding_latencies[:,0],'-.',color=cst_color)
    
    #Labels 
    for ax in axes[:-1]:
        ax.set_xticks([-0.5,-0.25, 0,0.25, 0.5])
        ax.set_xlabel('Anatomical Hierarchy Score')

    axes[0].set_ylabel('Decoding Speed')
    axes[1].set_ylabel('Decoding Latency (ms)')
    axes[2].set_ylabel('Max Decoding Performance')
    axes[2].set_ylim([0.25,1.1])

    #Add area names to legend
    for ii, area in enumerate(areas_sorted):
        fit_handles.append(mlines.Line2D([], [], c=cc[ii],marker='o',ms=8,ls='None',label=area))
    axes[3].legend(handles=fit_handles,loc=10,bbox_to_anchor=(-0.01, 0.5),framealpha=1,shadow=True)
    axes[3].axis('off')
    
    return fig

def plot_delta_PSTH(PSTHs,area=None):
    fig,axes = plt.subplots(1,6,figsize=(12,4),gridspec_kw={'width_ratios':[5,5,5,5,5,1]})
    plt.suptitle('{} \u0394PSTH: Run - Rest'.format(area),y=1.04)
    delta_PSTHs = PSTHs[:,1] - PSTHs[:,0]
    for iStim,ori in enumerate([0,45,90,135]):
        bb = np.percentile(np.abs(delta_PSTHs),99)
        bb = np.max(np.abs(delta_PSTHs))
        sns.heatmap(delta_PSTHs[iStim].T,ax=axes[iStim],vmin=-bb,vmax=bb,cmap='RdBu_r',cbar=False)

        if iStim > 0: 
            axes[iStim].set_yticks([])
        else:
            axes[iStim].set_ylabel('Neurons')
#         axes[iStim].set_xticks([0,10,19])
#         axes[iStim].set_xticklabels([-.5,0,.5],rotation=45)
        axes[iStim].set_xlabel('Time (s)')
        axes[iStim].set_title('{:.0f}\u00B0 DG'.format(ori))
        
    sns.heatmap(np.mean(delta_PSTHs,axis=0).T,ax=axes[-2],vmin=-bb,vmax=bb,cmap='RdBu_r',cbar=True,cbar_ax=axes[-1]) 
#     axes[-2].set_xticks([0,10,19]);axes[-2].set_yticks([])
#     axes[-2].set_xticklabels([-.5,0,.5],rotation=45)
    axes[-2].set_xlabel('Time (s)')
    axes[-2].set_title('Mean \u0394PSTH'.format(ori))
    return fig

def plot_distribution_match(spike_counts,trial_indices,param_list,fig=None, axes=None):
    
    sns.set_style("ticks")
    plt.rcParams['ytick.labelsize'] = 18 
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titleweight'] = 'normal'
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = 'Helvetica'

    #Plot distribitions before matching
    ls_list = [':','--']; handles = []
    if fig is None:
        fig,axes = plt.subplots(1,2,figsize=(12,6),gridspec_kw={'wspace': 0.25})
        ax = axes[0]
        ax.set_title('Before',fontsize=18)
        legend = False
    else:
    #Plot distribitions after matching
        ax = axes[1]
        ax.set_title('After',fontsize=18)
        legend = True
    
    for iR,runstr in enumerate(['Rest','Run']):
        population_spikes = np.sum(np.sum(spike_counts[trial_indices[iR]],axis=1),axis=-1)
    
        #Plot data
        sns.distplot(population_spikes,kde=False,norm_hist=True,color=state_colors[iR],ax=ax,label=runstr)

        #Plot fitted lognormal distribution
        lognormal_params = param_list[iR]
        x=np.linspace(np.min(population_spikes)-np.min(population_spikes)%50,np.max(population_spikes),10000)
        pdf_fitted = st.lognorm.pdf(x, lognormal_params[0], loc=lognormal_params[1], scale=lognormal_params[2]) # fitted distribution
        ax.plot(x ,pdf_fitted,'k',ls=ls_list[iR],lw=2)

        handles.append(mpatches.Patch(color=state_colors[iR],alpha=0.25,label=runstr))
        handles.append(mlines.Line2D([], [], marker='',color='k',ls=ls_list[iR]))

#     ax.set_xlim([50,550])
#     ax.set_ylim([0,0.03])
    if legend:
        ax.legend(handles=handles)
#         ax.set_yticks([])
#         ax.set_ylabel('')
#     else:
#         ax.set_ylabel('p(x)')
#         ax.set_yticks([0,0.03])
    ax.set_xlabel('Population Counts')
    
#     ax.set_xlim([0,125])
#     ax.set_xticks([0,125])
    
    return fig, axes

def plot_decoding_shuffle(decoding_accuracy, shuffles, pvalues,title=None,pdfdoc=None):
    
    nClasses = decoding_accuracy.shape[-1]
    ## Plot shuffle distributions ##
    fig,axes = plt.subplots(1,nClasses,figsize=(18,6))
    plt.suptitle(title,y=1.01)

    #Plot the shuffle distribution with the mean decoding performance for that class
    for i in range(nClasses):
        ax = axes[i]
        sns.distplot(shuffles[:,i,i],color=cc[i],ax=ax)
        if pvalues[i,i] < 0.01:
            ax.set_title('element [{},{}], pval: {:.1e}'.format(i,i,pvalues[i,i]))
        else:
            ax.set_title('element [{},{}], pval: {:.2f}'.format(i,i,pvalues[i,i]))

        ax.vlines(decoding_accuracy[i,i], *ax.get_ylim(),LineWidth=2.5,label='Data: {:.2f}'.format(decoding_accuracy[i,i]))
        ax.vlines(np.mean(shuffles,axis=0)[i,i], *ax.get_ylim(),LineWidth=2.5,LineStyle = '--',label='Shuffle: {:.2f}'.format(np.mean(shuffles,axis=0)[i,i]))
        ax.set_xlim(xmin=0)
        ax.legend()

    if pdfdoc is not None:
        pdfdoc.savefig(fig)
        plt.close(fig)

def plot_confusion_matrices(decoding_accuracy, decoding_zscores, pvalues, class_labels=None, title=None, pdfdoc=None):

    ##===== Plotting =====##
    fig,axes = plt.subplots(1,3,figsize=(21,7),gridspec_kw={'hspace': 0.2})
    plt.suptitle(title,y=0.87)

    #Plot actual decoding performance, averaged across kfolds
    sns.heatmap(decoding_accuracy,annot=True,fmt='.2f',annot_kws={'fontsize': 16},cbar=True,square=True,cbar_kws={'shrink': 0.5},ax=axes[0])
    axes[0].set_title('Decoding accuracy')

    #Plot z-scored decoding performance, averaged across kfolds
    sns.heatmap(decoding_zscores,annot=True,fmt='.2f',annot_kws={'fontsize': 16},cmap=sns.color_palette("RdBu", 100),vmin=-2,vmax=2,cbar=True,square=True,cbar_kws={'shrink': 0.5},ax=axes[1])
    axes[1].set_title('Z-scored decoding accuracy')

    #Get the p-values from the mean z-score; i.e. survival fraction
    sns.heatmap(pvalues,annot=True,fmt='.2e',annot_kws={'fontsize': 16},cmap='gray_r',cbar=True,square=True,cbar_kws={'shrink': 0.5},ax=axes[2])
    axes[2].set_title('p-values')

    #Labels
    if class_labels is not None:
        for ax in axes:
            ax.set_ylabel('Actual Stimulus Condition'); ax.set_xlabel('Decoded Stimulus Condition')
            ax.set_xticklabels(class_labels, rotation=45) 
            ax.set_yticklabels(class_labels, rotation=45)

    if pdfdoc is not None:
        pdfdoc.savefig(fig)
        plt.close(fig)


    
