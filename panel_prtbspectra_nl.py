import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import colormaps
import sys
import os

def wnum2wlen(wnum):
    wnum = np.array(wnum, float)
    near_zero = np.isclose(wnum, 0)
    wlen = np.zeros_like(wnum)
    wlen[near_zero] = np.inf
    wlen[~near_zero] = 2.0 * np.pi / wnum[~near_zero] * 1.0e-3 # km
    return wlen

def wlen2wnum(wlen):
    return wnum2wlen(wlen)

from pathlib import Path
from datetime import UTC, datetime, timedelta
import xarray as xr
import pandas as pd
from plotargs import plotargs, regions

dxdict = {'D1':27000.0,'D2':9000.0,'D3':3000.0}
ylims = {'ke':[1.0e-2,1.0e8],'tv':[1.0e-2,1.0e8],'tvn':[1.0e-2,1.0e8],'q':[1.0e-7,1.0e1],'cw':[1.0e-9,1.0e-3],'w':[1.0e-4,5.0e2]}

config = plotargs()
d = config.domain
init = config.init
base = config.base
vnormtype = config.vnormtype
generalized = config.generalized
inormtype = config.inormtype
iregtype = config.iregtype
ext = config.ext
valid = config.valid
smode = config.smode
emode = config.emode

fs = config.fstart
fe = config.fend
incfh = config.incfh
if fs==0 and fe==valid:
    f0 = 0
    frange=''
    fhrange = [0,1]+np.arange(incfh,fe+incfh,incfh).tolist()
else:
    f0 = fs
    frange=f'_f{fs:02d}-{fe:02d}'
    fhrange = np.arange(fs,fe+incfh,incfh)

lon0=config.lon0
lon1=config.lon1
lat0=config.lat0
lat1=config.lat1

slon=config.slon
elon=config.elon
slat=config.slat
elat=config.elat

plreg = config.plreg
v = 'ke'

if generalized:
    ntype = f'i{inormtype}{iregtype}v{vnormtype}'
else:
    ntype = vnormtype

hostdir = Path('data')

datadirs = {
    'D1':hostdir/'d1',
    'D2':hostdir/'d2',
    'D3':hostdir/'d3',
}

dwlist = ['EnSV','NL$+$','NL$-$']

dx = dxdict[d]
datadir = datadirs[d]
sig = np.loadtxt(datadir/'sig.txt')

#base
if plreg is not None:
    if plreg == 'd3' and d=='D3':
        fbase = f'kebase_psd.nc'
    else:
        fbase = f'kebase_psd_{plreg}.nc'
else:
    fbase = 'kebase_psd.nc'
dsb = xr.open_dataset(datadir/fbase)

psddict = {}
for m in range(smode,emode+1):
    psddict[m] = dict()
    for dw in dwlist:
        expn = f"m{m:02d}"
        if plreg is not None:
            if plreg == 'd3' and d=='D3':
                fname = f'kepsd_prtb_{expn}_{dw}.nc'
            else:
                fname = f'kepsd_prtb_{expn}_{dw}_{plreg}.nc'
        else:
            fname = f'kepsd_prtb_{expn}_{dw}.nc'
        print(datadir/fname)
        ds = xr.open_dataset(datadir/f'{fname}')
        psddict[m][dw] = ds
if plreg == 'd3':
    dstmp = xr.open_dataset(datadirs['D3']/'kebase_psd.nc')
    wnummin = dstmp.wnums[1]
    wnummax = dstmp.wnums[-1]
else:
    dstmp = xr.open_dataset(datadirs['D1']/'kebase_psd.nc')
    wnummin = dstmp.wnums[1]
    dstmp = xr.open_dataset(datadirs['D3']/'kebase_psd.nc')
    wnummax = dstmp.wnums[-1]

nlev = dsb.level.size
intlev = 15 #nlev//3
slev0 = 1
elev0 = 1+intlev
slev1 = elev0
elev1 = slev1 + intlev
slev2 = elev1
elev2 = nlev+1
slevlist = [slev0,slev1,slev2]
elevlist = [elev0,elev1,elev2]

figdir = Path('.')

ncols = 3
nrows = 1 #emode - smode + 1
figwidth = ncols * 3
figheight = nrows * 3 + 1
#colors = colormaps['tab10'](np.linspace(0.0,0.4,emode-smode+1))
colors = colormaps['jet'](np.linspace(0.0,1.0,len(fhrange)))
markers = {'EnSV':'','NL$+$':'$+$','NL$-$':'$-$'}
styles = {'EnSV':'solid','NL$+$':'dashed','NL$-$':'dotted'}
widths = {'EnSV':1.0,'NL$+$':1.5,'NL$-$':2.0}
captions = ['(a)','(b)','(c)']
captions = {
#    'D1':['(a)','(b)','(c)'],
    'D1':['(a)','(b)','(a)'],
#    'D2':['(d)','(e)','(f)'],
    'D2':['(c)','(d)','(b)'],
#    'D3':['(g)','(h)','(i)'],
    'D3':['(e)','(f)','(c)'],
}
for ilev in range(1): #len(slevlist)):
    slev = slevlist[ilev]
    elev = elevlist[ilev]
    dx = dxdict[d]
    for m in psddict.keys():
        for dw in ['NL$+$']:
            nmode = f'_{d.lower()}_{dw}_m{m:02d}'
            # first row: spectra
            # second row: ratio to base field
            # third row: (only for NL) ratio to EnSV
            fig, (ax0,ax1,ax2) = plt.subplots(nrows=3,figsize=[4.5,9],constrained_layout=True)
            ds = psddict[m][dw]
            dsref = psddict[m]['EnSV']
            for fh in fhrange:
                i = fhrange.index(fh)
                wnum = dsb.wnums[1:]
                vb = dsb[v].sel(ft=fh,level=slice(slev,elev)).mean(axis=0)
                ax0.loglog(wnum,vb[1:],lw=1.0,ls='solid',\
                    c=colors[i],alpha=0.5)
                wnum = ds.wnums[1:]
                vp = ds[v].sel(ft=fh,level=slice(slev,elev)).mean(axis=0)
                ax0.loglog(wnum,vp[1:],label=f'FT{fh:02d}',\
                    ls='dotted',c=colors[i])
                ax1.semilogx(wnum,vp[1:]/vb[1:]*100,label=f'FT{fh:02d}',\
                    ls='dotted',c=colors[i])
                if dw!='EnSV':
                    vref = dsref[v].sel(ft=fh,level=slice(slev,elev)).mean(axis=0)
                    ax2.semilogx(wnum,vp[1:]/vref[1:],label=f'FT{fh:02d}',\
                    ls='dotted',c=colors[i])
            #
            kthres = 2.0 * np.pi / 7.0 / dx #Skamarock 2004
            for irow,ax in enumerate([ax0,ax1,ax2]):
                ax.fill_between(wnum,0,1,where=wnum>kthres,
                color='gray',alpha=0.1,transform=ax.get_xaxis_transform(),zorder=0)
                if irow==2:
                    ax.set_xlabel(r'$k_h=\sqrt{k_x^2+k_y^2}$ [radian/m]')
                sax = ax.secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
                if irow==0:
                    sax.set_xlabel(r'$2\pi/k_h$ [km]')
                ax.grid()
                ax.set_xlim(wnummin,wnummax)
                #icap = icol
                #caption = captions[icap]
                caption = captions[d][irow]
                annotate_dict = {
                    'xy':(1,1),'xycoords':'axes fraction',
                    'xytext':(-0.5,-0.5),'textcoords':'offset fontsize',
                    'fontsize':'medium','va':'top','ha':'right',
                    'bbox':dict(facecolor='0.7',edgecolor='none',pad=3.0)
                }
                if plreg=='d3' or irow>=1:
                    annotate_dict.update(
                        xy=(0,1),
                        xytext=(0.5,-0.5),
                        ha='left'
                    )
                ax.annotate(
                    f'{caption} {d} {dw} mode{m}',
                    **annotate_dict
                )
            if v=='ke' or v=='tv' or v=='tvn':
                xref = 2.0*np.pi / np.array([1.0e7,3.0e5])
                yamp1 = np.pi**3 * 1.0e-11
                yref = yamp1*(xref**(-3))
                #yref = yref * 1e7 / yref[0]
                ax0.loglog(xref,yref,c='k',lw=1,zorder=0)
                #if expn=='msm2msm3_da':
                #    xy=(wnum[-100],yref[-100]*2.0); ha='center'; va='bottom'
                #else:
                #    xy=(wnum[-1],yref[-1]); ha='left'; va='center'
                if plreg=='d3':
                    xanno = 1.0e-5
                    yanno = yamp1*(xanno**(-3))
                    xy = (xanno, yanno); ha='left'; va='bottom'
                else:
                    xanno = 2.0*np.pi / 1.0e6
                    yanno = yamp1*(xanno**(-3))
                    xy = (xanno, yanno); ha='left'; va='bottom'
                ax0.annotate(r'$k^{-3}$',xy=xy,xycoords='data',\
                        ha=ha,va=va,fontsize=12)
                xref = 2.0*np.pi / np.array([3.0e5,1.0e3])
                yamp2 = (yamp1**(5./9.))*(10.0**(20./9.))
                yref = yamp2*(xref**(-5./3.))
                #yref = yref * 1e7 / yref[0]
                ax0.loglog(xref,yref,c='k',ls='dashed',lw=1,zorder=0)
                #if expn=='msm2msm3_da':
                #    xy=(wnum[-100],yref[-100]*2.0); ha='center'; va='bottom'
                #else:
                #    xy=(wnum[-1],yref[-1]); ha='left'; va='center'
                xanno = 2.0*np.pi / 2.0e4
                yanno = yamp2*(xanno**(-5./3.))
                xy=(xanno,yanno); ha='left'; va='bottom'
                #xy=(xref[0],yref[0]); ha='left'; va='bottom'
                ax0.annotate(r'$k^{-5/3}$',xy=xy,xycoords='data',\
                    ha=ha,va=va,fontsize=12)
            #if expn=='rsm2rsm27_da':
            ax0.set_ylim(ylims[v])
            if base=='mean':
                #if d=='D1':
                #    ax1.set_ylim(0.0,0.6)
                #elif d=='D2':
                #    ax1.set_ylim(0.0,3.5)
                #else:
                #    ax1.set_ylim(0.0,9.0)
                ax1.set_ylim(0.0,100.0)
                ax2.set_ylim(0.0,10.0)
            else:
                #if d=='D1':
                #    ax1.set_ylim(0.0,0.3)
                #elif d=='D2':
                #    ax1.set_ylim(0.0,1.0)
                #else:
                ax1.set_ylim(0.0,200.0)
                ax2.set_ylim(0.0,12.5)
            ax0.set_ylabel(dsb[v].attrs["units"])
            ax1.set_ylabel(rf"{v.upper()}$^\prime$/{v.upper()}"+r"$_\mathrm{b}$ [%]")
            if dw=='EnSV':
                ax2.remove()
                ax1.set_xlabel(r'$k_h=\sqrt{k_x^2+k_y^2}$ [radian/m]')
            else:
                ax2.set_ylabel("ratio to EnSV")
            ax0.legend(loc='upper left',bbox_to_anchor=(1.01,1.0),facecolor='0.7')
            figtitle = fr'$\sigma$={sig[slev-1]:.2f}-{sig[elev-2]:.2f} [{slev}-{elev-1}]'
            vrange=f'z{slev}-{elev-1}'
            fig.savefig(figdir/f'figure4+S3{captions[d][-1]}.pdf',dpi=600)
            fig.savefig(figdir/f'figure4+S3{captions[d][-1]}.png',dpi=300)
            plt.show(block=False)
            plt.close()
