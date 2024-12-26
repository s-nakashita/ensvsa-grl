#https://matplotlib.org/stable/gallery/text_labels_and_annotations/label_subplots.html#sphx-glr-gallery-text-labels-and-annotations-label-subplots-py
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import xarray as xr
import re

import os
import sys

titles = {'te':'total','ke':'kinetic','pe':'potential','lh':'latent heat'}
def read_en(fname,nmem,normalize=False):
    buf = np.fromfile(fname,dtype=">f4").reshape(-1,nmem+1)
    nlen = buf.shape[0]
    nfh = nlen // 4
    data = dict()
    for v in titles.keys():
        data[v] = []
    i = 0
    for ifh in range(nfh):
        for v in titles.keys():
            data[v].append(buf[i,])
            i+=1
    for v in titles.keys():
        data[v] = np.array(data[v])
    if normalize:
        scale = 1.0 / data['te'][0,:]
        data['te'] = data['te'] * scale[None,:]
        data['ke'] = data['ke'] * scale[None,:]
        data['pe'] = data['pe'] * scale[None,:]
        data['lh'] = data['lh'] * scale[None,:]
    return data

from plotargs import plotargs, regions

from pathlib import Path
from datetime import datetime

config = plotargs()
init = config.init
nmem = config.nmem
base = config.base
vnormtype = config.vnormtype
generalized = config.generalized
inormtype = config.inormtype
iregtype = config.iregtype
ext = config.ext
valid = config.valid
smode = config.smode
emode = config.emode

fstart = config.fstart
fend = config.fend
incfh = config.incfh

lon0,lon1,lat0,lat1=regions[iregtype.upper()]

slon=config.slon
elon=config.elon
slat=config.slat
elat=config.elat

glob = config.glob

if generalized:
    ntype = f'i{inormtype}{iregtype}v{vnormtype}'
else:
    ntype = vnormtype
if smode == emode:
    expn = f"m{smode:02d}"
else:
    expn = f"m{smode:02d}-{emode:02d}"

hostdir = Path('data')
datadirs = {
    'D1':hostdir/f'd1',
    'D2':hostdir/f'd2',
    'D3':hostdir/f'd3'
}

fnames1 = {
    'D1':{
        'EnSV':f'tevol-{ntype}-mMM-v{valid:02d}h.grd',
        'NL':f'nl-tevol-{ntype}-mMM-v{valid:02d}h{ext}.grd',
    },
    'D2':{
        'EnSV':f'tevol-{ntype}-mMM-v{valid:02d}h.grd',
        'NL':f'nl-tevol-{ntype}-mMM-v{valid:02d}h{ext}.grd'
    },
    'D3':{
        'EnSV':f'tevol-i{inormtype}allv{vnormtype}-mMM-v{valid:02d}h.grd',
        'NL':f'nl-tevol-i{inormtype}allv{vnormtype}-mMM-v{valid:02d}h{ext}.grd'
    }
}
fnames2 = {
    'D1':{
        'EnSV':f'tevol-{iregtype}-{ntype}-mMM-v{valid:02d}h.grd',
        'NL':f'nl-tevol-{iregtype}-{ntype}-mMM-v{valid:02d}h{ext}.grd',
    },
    'D2':{
        'EnSV':f'tevol-{iregtype}-{ntype}-mMM-v{valid:02d}h.grd',
        'NL':f'nl-tevol-{iregtype}-{ntype}-mMM-v{valid:02d}h{ext}.grd'
    },
    'D3':{
        'EnSV':f'tevol-all-i{inormtype}allv{vnormtype}-mMM-v{valid:02d}h.grd',
        'NL':f'nl-tevol-all-i{inormtype}allv{vnormtype}-mMM-v{valid:02d}h{ext}.grd'
    }
}

contribs = {
    'D1':{1:21.0,2:11.8,3:9.5},
    'D2':{1:13.6,2:10.5,3:7.2},
    'D3':{1:11.9,2:9.9,3:6.4}
}

figdir = Path('.')
if not figdir.exists():
    figdir.mkdir(parents=True)

dwlist = ['EnSV','NL$+$','NL$-$','NLmean']
entdict = dict()
enidict = dict()
for d in datadirs.keys():
    datadir = datadirs[d]
    entdict[d] = dict()
    enidict[d] = dict()
    for m in range(smode,emode+1):
        entdict[d][m] = dict()
        enidict[d][m] = dict()
        for dw in dwlist:
            if dw=='EnSV':
                fname1=re.sub('MM',f'{m:02d}',fnames1[d][dw])
                fname2=re.sub('MM',f'{m:02d}',fnames2[d][dw])
                print(datadir/fname1)
                print(datadir/fname2)
                data1 = read_en(datadir/fname1,nmem)
                data2 = read_en(datadir/fname2,nmem)
            else:
                fname1 = re.sub('MM',f'{m:02d}',fnames1[d]['NL'])
                fname2 = re.sub('MM',f'{m:02d}',fnames2[d]['NL'])
                print(datadir/fname1)
                print(datadir/fname2)
                data1 = read_en(datadir/fname1,2)
                data2 = read_en(datadir/fname2,2)
            entdict[d][m][dw] = data1
            enidict[d][m][dw] = data2

nrows = 3
ncols = emode - smode + 1
nmode = f'm{smode:d}-{emode:d}'
figwidth = ncols*3
figheight = 8
#colors = plt.get_cmap('tab10')(np.linspace(0.0,0.1*(ncols-1),ncols))
colors = {'EnSV':'dimgray','NL$+$':'r','NL$-$':'b','NLmean':'k'}
markers = {'EnSV':'.','NL$+$':'$+$','NL$-$':'$-$','NLmean':''}
styles = {'EnSV':'solid','NL$+$':'dashed','NL$-$':'dotted','NLmean':'solid'}
widths = {'EnSV':1.5,'NL$+$':1.5,'NL$-$':2.0,'NLmean':1.0}
captions = [
    '(a)','(b)','(c)','(d)','(e)',
    '(f)','(g)','(h)','(i)','(j)',
    '(k)','(l)','(m)','(n)','(o)']
for v in ['te']:
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=[figwidth,figheight],sharey=True,sharex=True,constrained_layout=True)
    lines = []
    labels = []
    for dw in dwlist:
        lines.append(Line2D([0],[0],c=colors[dw],ls=styles[dw],marker=markers[dw],lw=widths[dw]))
        labels.append(dw)
    for icol, d in enumerate(entdict.keys()):
        for irow, m in enumerate(entdict[d].keys()):
            ax = axs[irow,icol]
            for dw in entdict[d][m].keys():
                ent = entdict[d][m][dw]
                eni = enidict[d][m][dw]
                taxis = np.arange(ent[v].shape[0])
                iv = np.argmin(np.abs(taxis-valid))
                if dw=='EnSV':
                    z  = ent[v][:,0] / eni[v][0,0]
                    ax.semilogy(taxis,z,marker=markers[dw],ls=styles[dw],lw=widths[dw]
                    ,c=colors[dw])
                    #,c=colors[m-1])
                    #if icol==0:
                    #    lines.append(Line2D([0],[0],c=colors[m-1]))
                    #    labels.append(f'mode{m}')
                    ze = np.zeros(nmem)
                    for imem in range(nmem):
                        ze[imem] = ent[v][iv,imem+1]/eni[v][0,imem+1]
                    ax.boxplot(np.log(ze)[:,None],positions=[taxis[iv]],widths=2.0,showfliers=False) #,whis=(0,100))
                else:
                    if dw=='NL$+$':
                        z = ent[v][:,0] / eni[v][0,0]
                    elif dw=='NL$-$':
                        z = ent[v][:,1] / eni[v][0,1]
                    else:
                        z = ent[v][1:,2] / (eni[v][0,0]+eni[v][0,1]) * 2.0
                    ax.semilogy(taxis[taxis.size-z.size:],z,marker=markers[dw],ls=styles[dw],lw=widths[dw]
                    ,c=colors[dw])
                    #,c=colors[m-1])
                print(f"{d}({v}) {dw} mode{m} final={z[iv]}")
            #ax.set_title(f'{d} mode{m} ({contribs[d][m]:.1f}%)')
            icap = icol*nrows+irow
            #ax.annotate(
            #    f'{captions[icap]} {d} mode{m} ({contribs[d][m]:.1f}%)',
            #    xy=(0,1),xycoords='axes fraction',
            #    xytext=(0.5,-0.5),textcoords='offset fontsize',
            #    fontsize='medium', va='top', #ha='right',
            #    bbox=dict(facecolor='0.7',edgecolor='none',pad=3.0)
            #)
            ax.text(
                0.0,1.0,
                f'{captions[icap]} {d} mode{m} ({contribs[d][m]:.1f}%)',
                transform=(
                    ax.transAxes + ScaledTranslation(-12/72, +7/72, fig.dpi_scale_trans)
                ),fontsize=12,va='bottom'
            )
            #if irow==2:
            #    ax.set_xlabel('forecast hours')
            if icol==0:
                #ax.set_ylabel(r'$\frac{\|\mathbf{z}\|^2_{\mathbf{G}_\mathrm{v}}}{\|\mathbf{y}\|^2_{\mathbf{G}_\mathrm{a}}}$')
                ax.annotate(r'$\frac{\|\mathbf{z}\|^2_{\mathbf{G}_\mathrm{v}}}{\|\mathbf{y}\|^2_{\mathbf{G}_\mathrm{a}}}$',
                xy=(-0.25,0.5),xycoords='axes fraction',ha='center',va='center',fontsize=12)
    axs[nrows-1,ncols-1].annotate('[h]',xy=(1.05,-0.035),xycoords='axes fraction',ha='left',va='top',fontsize=11)
    axs[0,ncols-1].legend(lines,labels,loc='upper left',bbox_to_anchor=(1.01,1.0))
    for ax in axs.flatten():
        ax.set_ylim(1e-1,1e2)
        ax.vlines([valid],0,1,ls='solid',colors='gray',alpha=0.5,transform=ax.get_xaxis_transform(),zorder=0)
        ax.set_xticks(taxis[::3])
        ax.set_xticklabels(taxis[::3])
        ax.grid()
    fig.savefig(figdir/"figure2.pdf",dpi=600)
    figtitle=f"{titles[v]} {init.strftime('%Y-%m-%d %HZ')}"
    figtitle=figtitle+f" lon:{slon:.1f}-{elon:.1f} lat:{slat:.1f}-{elat:.1f}"
    #fig.suptitle(figtitle)
    fig.savefig(figdir/"figure2.png",dpi=300)
    plt.show(block=False)
    plt.close()
