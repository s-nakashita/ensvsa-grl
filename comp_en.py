import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation
import cartopy.crs as ccrs
from metpy.units import units
import metpy.calc as mpcalc
from metpy.interpolate import log_interpolate_1d
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['figure.titlesize'] = 18
import xarray as xr
import re

import os
import sys

titles = {'te':'total','ke':'kinetic','pe':'potential','lh':'latent heat'}
def read_en(fname,nlev,nlat,nlon):
    buf = np.fromfile(fname,dtype=">f4").reshape(-1,nlat,nlon)
    data = dict()
    data['peps'] = buf[0,:,:]
    nstart=1
    for v in titles.keys():
        data[v] = buf[nstart:nstart+nlev,:,:]
        nstart+=nlev
    return data

## custom colormap from NCL spread_15lev
from matplotlib.colors import ListedColormap
color_list = [(255,225,225),(255,210,210),(255,165,165),\
    (255,120,120),(255,75,75),(255,0,0),(255,100,0),(255,150,0),\
    (255,200,0),(255,255,0),(140,255,0),(0,255,0),(0,205,95),\
    (0,145,200)]
ncolors = len(color_list)
color_list = np.array(color_list) / 255.0
mycmap = ListedColormap(color_list,"energy",N=ncolors).with_extremes(over=(0,0,1),under=(1,1,1),bad='gray')

from plotargs import plotargs

from pathlib import Path
from datetime import datetime

config = plotargs()
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

fh = config.fh

lon0=config.lon0
lon1=config.lon1
lat0=config.lat0
lat1=config.lat1

slon=config.slon
elon=config.elon
slat=config.slat
elat=config.elat

glob = config.glob

if generalized:
    ntype = f'i{inormtype}{iregtype}v{vnormtype}'
else:
    ntype = vnormtype

hostdir = Path('data')
datadirs = {
    'D1':hostdir/f'd1',
    'D2':hostdir/f'd2',
    'D3':hostdir/f'd3'
}

headers = {
    'D1':f'{ntype}-mMM-v{valid:02d}h',
    'D2':f'{ntype}-mMM-v{valid:02d}h',
    'D3':f'i{inormtype}allv{vnormtype}-mMM-v{valid:02d}h'
}
contribs = {
    'D1':{1:21.0,2:11.8,3:9.5},
    'D2':{1:13.6,2:10.5,3:7.2},
    'D3':{1:11.9,2:9.9,3:6.4}
}

paramsdict = dict()
endict = dict()
for d in datadirs.keys():
    datadir = datadirs[d]
    params = dict()
    lon = np.loadtxt(datadir/'rlon.txt')
    lat = np.loadtxt(datadir/'rlat.txt')
    sig = np.loadtxt(datadir/'sig.txt')
    nlon = lon.size
    nlat = lat.size
    nlev = sig.size
    sigh = np.loadtxt(datadir/'sigh.txt')
    vwgts = sigh[:-1] - sigh[1:]
    params['vwgt'] = vwgts
    lon0tmp=max(lon0,np.nanmin(lon))
    lon1tmp=min(lon1,np.nanmax(lon))
    lat0tmp=max(lat0,np.nanmin(lat))
    lat1tmp=min(lat1,np.nanmax(lat))
    print("lon{:.2f}-{:.2f} lat{:.2f}-{:.2f}".format(lon0tmp,lon1tmp,lat0tmp,lat1tmp))
    dlon = dlat = 5
    latb = np.ceil(lat0tmp/float(dlat))*dlat
    latt = np.floor(lat1tmp/float(dlat))*dlat
    lonl = np.ceil(lon0tmp/float(dlon))*dlon
    lonr = np.floor(lon1tmp/float(dlon))*dlon
    lonlist=list(range(int(lonl),int(lonr)+dlon,dlon))
    latlist=list(range(int(latb),int(latt)+dlat,dlat))
    print(lonlist)
    print(latlist)
    params['lonlist'] = lonlist
    params['latlist'] = latlist
    i0=np.argmin(np.abs(lon-lon0tmp))
    if lon[i0] < lon0tmp: i0=i0+1
    i1=np.argmin(np.abs(lon-lon1tmp))
    if lon[i1] < lon1tmp: i1=i1+1
    j0=np.argmin(np.abs(lat-lat0tmp))
    if lat[j0] < lat0tmp: j0=j0+1
    j1=np.argmin(np.abs(lat-lat1tmp))
    if lat[j1] < lat1tmp: j1=j1+1
    if glob:
        params['sij'] = (i0,i1,j0,j1)
        i0 = 0
        i1 = lon.size - 1
        j0 = 0
        j1 = lat.size - 1
    print("plot lon:{:.1f}[{:d}]-{:.1f}[{:d}] lat:{:.1f}[{:d}]-{:.1f}[{:d}]"\
        .format(lon[i0],i0,lon[i1],i1,\
            lat[j0],j0,lat[j1],j1))
    params['ij'] = (i0,i1,j0,j1)
    params['lon'] = lon[i0:i1+1]
    params['lat'] = lat[j0:j1+1]
    paramsdict[d] = params

    endict[d] = dict()
    for m in range(smode,emode+1):
        head=re.sub('mMM',f'm{m:02d}',headers[d])
        endict[d][m] = dict()
        for v in ['te']:
            fname = f"{head}{ext}.f{fh:02d}.{v}.npy"
            endict[d][m][v] = np.load(datadir/fname)
        #print(endict[d][m])

figdir = Path('.')
if not figdir.exists():
    figdir.mkdir(parents=True)

nmode = f'm{smode:d}-{emode:d}'
nrows = emode - smode + 1
ncols = 3
figwidth = ncols*3
figheight = nrows*3
captions = {
    'D1':{
        1:'(a)',2:'(b)',3:'(c)'
    },
    'D2':{
        1:'(d)',2:'(e)',3:'(f)'
    },
    'D3':{
        1:'(g)',2:'(h)',3:'(i)'
    }
}
#captions = [
#    '(a)','(b)','(c)','(d)','(e)',
#    '(f)','(g)','(h)','(i)','(j)',
#    '(k)','(l)','(m)','(n)','(o)']
for v in ['te']:
    fig = plt.figure(figsize=(figwidth,figheight),constrained_layout=True)
    axs = []
    vlim = -999.
    for i, d in enumerate(endict.keys()):
        icol = i + 1
        for j, m in enumerate(endict[d].keys()):
            irow = j + 1
            iplot = ncols*j + icol
            if glob and d=='D1':
                ax = fig.add_subplot(nrows,ncols,iplot,projection=ccrs.PlateCarree(central_longitude=180.0))
            else:
                ax = fig.add_subplot(nrows,ncols,iplot,projection=ccrs.PlateCarree())
            axs.append(ax)
            #iplot += 1
            z = endict[d][m][v]
            if glob:
                si0, si1, sj0, sj1 = paramsdict[d]['sij']
                vmax = np.nanmax(z[sj0:sj1+1,si0:si1+1])
            else:
                vmax = np.nanmax(z)
            l10vmax = np.log10(vmax)
            vbase = np.floor(l10vmax)
            c = 10**(l10vmax - vbase)
            ccut = int(c*10)*0.1
            #ccut = np.floor(vmax/(10**vbase))
            vlimtmp = ccut*(10**vbase)
            vlim = max(vlim,vlimtmp)
    vmin = vlim / 10.0
    vmax = vmin * 10.0
    ncontour = int(vmax/vmin)
    #plevels = np.linspace(vmin,vmax,ncontour)
    if v!='pe':
        plevels = np.linspace(10.0,80.0,8)
    else:
        plevels = np.linspace(1.0,8.0,8)
    print(f"{v} plevels={plevels}")

    k = 0
    plist = []
    for j, d in enumerate(endict.keys()):
        lon = paramsdict[d]['lon']
        lat = paramsdict[d]['lat']
        nlon = lon.size
        nlat = lat.size
        lonlist = paramsdict[d]['lonlist']
        latlist = paramsdict[d]['latlist']
        for i, m in enumerate(endict[d].keys()):
            ax = axs[k]
            k += 1
            z = endict[d][m][v]
            print(z.shape,np.nanmin(z),np.nanmax(z))
            if v=='te':
                # contour levels are determined for each mode
                vmax = np.nanmax(z)
                l10vmax = np.log10(vmax)
                vbase = np.floor(l10vmax)
                c = 10**(l10vmax - vbase)
                ccut = int(c*10)*0.1
                #ccut = np.floor(vmax/(10**vbase))
                vlim = ccut*(10**vbase)
                vmin = vlim / 10.0
                vmax = vmin * 10.0
                ncontour = int(vmax/vmin)
                plevels = np.linspace(vmin,vmax,ncontour)
            p = ax.contourf(lon,lat,z,\
                plevels,transform=ccrs.PlateCarree(),\
                cmap=mycmap,extend='both',zorder=0)
            plist.append(p)
            if v=='te':
                c = fig.colorbar(p,ax=ax,orientation='horizontal',shrink=0.5,pad=0.01)
                c.ax.annotate(r'J kg$^{-1}$',xy=(1.1,0.5),xycoords='axes fraction',va='center',fontsize=9)
                c.ax.set_xticks([])
                c.ax.set_xticks([plevels[0],plevels[-1]])
                c.ax.tick_params(labelsize=9)
            ## target region
            ax.plot([slon,elon,elon,slon,slon],
                    [slat,slat,elat,elat,slat],
                    lw=0.5,c='k',#ls='dashed',
                    transform=ccrs.PlateCarree())
            ax.add_patch(Rectangle((slon,slat),elon-slon,elat-slat,fc="k",alpha=0.2,
                transform=ccrs.PlateCarree()))
            
            ax.coastlines(color='saddlebrown',lw=1.0)
            if not glob:
                ax.set_xticks(lonlist,crs=ccrs.PlateCarree())
                ax.set_yticks(latlist,crs=ccrs.PlateCarree())
                ax.tick_params(labelsize=9)
            else:
                ax.set_extent([lon[0],lon[-1],lat[0],lat[-1]],ccrs.PlateCarree())
            #ax.grid()
            #if d==dw:
            #    ax.set_title(f"{d}")
            #else:
            #icap = k-1
            #caption = captions[icap]
            try:
                caption = captions[d][m]
            except KeyError:
                caption = ''
            #ax.set_title(f"{captions[icap]} {d} mode{m} ({contribs[d][m]:.1f}%)",loc='left')
            # Use ScaledTranslation to put the label
            # - at the top left corner (axes fraction (0, 1)),
            # - offset 12 pixels left and 7 pixels up (offset points (-12, +7)),
            # i.e. just outside the axes.
            ax.text(
                0.0,1.0,f"{caption} {d} mode{m} ({contribs[d][m]:.1f}%) FT{fh:02d}",
                transform=(
                    ax.transAxes + ScaledTranslation(-12/72, +7/72, fig.dpi_scale_trans)
                ),fontsize=12,va='bottom')
            #ax.annotate(
            #    f'{captions[icap]} {d} mode{m} ({contribs[d][m]:.1f}%)',
            #    xy=(0,1),xycoords='axes fraction',
            #    xytext=(0.5,-0.5),textcoords='offset fontsize',
            #    fontsize='medium', va='top', #ha='right',
            #    bbox=dict(facecolor='0.7',edgecolor='none',pad=3.0)
            #)
        if v=='te':
            #c.ax.annotate(r'J kg$^{-1}$',xy=(1.1,0.5),xycoords='axes fraction',va='center',fontsize=14)
            print("")
        else:
            c = fig.colorbar(p,ax=axs,orientation='horizontal',aspect=40,shrink=0.5,pad=0.01)
            c.ax.annotate('%',xy=(1.1,0.5),xycoords='axes fraction',va='center',fontsize=14)
            #c.ax.set_xticks([])
            #c.ax.set_xticks([plevels[0],plevels[-1]])
            c.ax.tick_params(labelsize=14)
    figtitle=titles[v]+f" {init.strftime('%Y-%m-%d %HZ')}+{fh:02d}h"
    fig.savefig(figdir/"figure1.pdf",dpi=600)
    #fig.suptitle(figtitle)
    fig.savefig(figdir/"figure1.png",dpi=300)
    plt.show(block=False)
    plt.close()
    #exit()
