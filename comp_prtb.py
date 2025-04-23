import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation
import cartopy.crs as ccrs
from metpy.units import units
import metpy.calc as mpcalc
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['figure.titlesize'] = 20

import os
import sys
import xarray as xr
from plotargs import plotargs
import re

from pathlib import Path
from datetime import datetime

config = plotargs()
init = config.init
cbase = config.base
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

if generalized:
    ntype = f'i{inormtype}{iregtype}v{vnormtype}'
else:
    ntype = vnormtype

hostdir = Path('data')
datadirs = {
    'D1':hostdir/'d1',
    'D2':hostdir/'d2',
    'D3':hostdir/'d3'
}

headers = {
    'D1':f'{ntype}-mMM-v{valid:02d}h',
    'D2':f'{ntype}-mMM-v{valid:02d}h',
    'D3':f'i{inormtype}allv{vnormtype}-mMM-v{valid:02d}h'
}

figdir = Path('.')
if not figdir.exists():
    figdir.mkdir(parents=True)

var = ["ugrdprs","vgrdprs","spfhprs","tmpprs"]
unitnames = {"ugrdprs":"m/s","vgrdprs":"m/s","spfhprs":"kg/kg","tmpprs":"degK"}

paramsdict = dict()
basedict = dict()
prtbdict = dict()
for d in datadirs.keys():
    datadir = datadirs[d]
    params = dict()
    lon = np.loadtxt(datadir/'rlon.txt')
    lat = np.loadtxt(datadir/'rlat.txt')
    nlon = lon.size
    nlat = lat.size
    if d == 'D1':
        params['intuv'] = 4 #3 #5
        params['intuvp'] = 4 #3
    elif d == 'D2':
        params['intuv'] = 12 #9 #15
        params['intuvp'] = 12 #9
    else:
        params['intuv'] = 36 #24 #40
        params['intuvp'] = 36 #24

    lon0tmp=max(lon0,np.nanmin(lon))
    lon1tmp=min(lon1,np.nanmax(lon))
    lat0tmp=max(lat0,np.nanmin(lat))
    lat1tmp=min(lat1,np.nanmax(lat))
    print("lon{:.2f}-{:.2f} lat{:.2f}-{:.2f}".format(lon0,lon1,lat0,lat1))
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
    i1=np.argmin(np.abs(lon-lon1tmp))
    j0=np.argmin(np.abs(lat-lat0tmp))
    j1=np.argmin(np.abs(lat-lat1tmp))
    print("plot lon:{:.1f}[{:d}]-{:.1f}[{:d}] lat:{:.1f}[{:d}]-{:.1f}[{:d}]"\
        .format(lon[i0],i0,lon[i1],i1,\
            lat[j0],j0,lat[j1],j1))
    params['ij'] = (i0,i1,j0,j1)
    params['lon'] = lon[i0:i1+1]
    params['lat'] = lat[j0:j1+1]
    paramsdict[d] = params

plot_var = ["spfhprs"]
pvarnames = {
    "tmpprs":r'$Z$, $\mathbf{u}$, $T^\prime$, $\mathbf{u}^\prime$',
    "spfhprs":r'$\theta_\mathrm{e}$, $\mathbf{u}$, $q^\prime$, $\mathbf{u}^\prime$'
    }
fignames = {"tmpprs":"t+uv+hgt","spfhprs":"q+uv+thte"}
plot_lev = [925,850,700,500,300,200]
plot_lev = [850]

nrows = emode - smode + 1
ncols = 3
if nrows < 5:
    nmode = f'_m{smode:d}-{emode:d}'
else:
    nmode = ''
figwidth = ncols*3
figheight = nrows*3
captions = {
    'D1':{1:'(a)',2:'(b)',3:'(c)'},
    'D2':{1:'(d)',2:'(e)',3:'(f)'},
    'D3':{1:'(g)',2:'(h)',3:'(i)'},
}
for pvar in plot_var:
    for lev in plot_lev:
        if pvar=='spfhprs' and lev < 500: continue
        fig = plt.figure(figsize=(figwidth, figheight),constrained_layout=True)
        #iplot = 1
        axs = []
        zb = {}
        ub = {}
        vb = {}
        qb = {}
        zp = {}
        up = {}
        vp = {}
        vlim = -999.
        for j, d in enumerate(datadirs.keys()):
            icol = j + 1
            i0, i1, j0, j1 = paramsdict[d]['ij']
            datadir = datadirs[d]
            u=np.load(datadir/f'ub{lev}.fh{fh:02d}.npy')
            ub[d] = u[j0:j1+1,i0:i1+1]*units('m/s')
            v=np.load(datadir/f'vb{lev}.fh{fh:02d}.npy')
            vb[d] = v[j0:j1+1,i0:i1+1]*units('m/s')
            q=np.load(datadir/f'qb{lev}.fh{fh:02d}.npy')
            qb[d] = q[j0:j1+1,i0:i1+1]*units('kg/kg')
            pbase = 'thteprs'
            z=np.load(datadir/f'eptb{lev}.fh{fh:02d}.npy')
            zb[d] = z * units('m/s')
            unit = r'g kg$^{-1}$'
            cmapname = 'BrBG'
            uvcolor = 'crimson'
            up[d] = {}
            vp[d] = {}
            zp[d] = {}
            for i, m in enumerate(range(smode,emode+1)):
                if cbase == 'mean':
                    if m == 2 and d != 'D1':
                        psign = -1.0
                    else:
                        psign = 1.0
                elif cbase == 'cntl':
                    if m==2 and d=='D3':
                        psign = -1.0
                    else:
                        psign = 1.0
                irow = i + 1
                iplot = ncols*i + icol
                ax = fig.add_subplot(nrows,ncols,iplot,projection=ccrs.PlateCarree())
                axs.append(ax)
                #iplot += 1
                u=np.load(datadir/f'u{lev}.m{m:02d}.fh{fh:02d}.npy')*units('m/s')
                v=np.load(datadir/f'v{lev}.m{m:02d}.fh{fh:02d}.npy')*units('m/s')
                q=np.load(datadir/f'q{lev}.m{m:02d}.fh{fh:02d}.npy')*units('kg/kg')
                up[d][m] = psign*(u[j0:j1+1,i0:i1+1] - ub[d])
                vp[d][m] = psign*(v[j0:j1+1,i0:i1+1] - vb[d])
                zp[d][m] = psign*(q[j0:j1+1,i0:i1+1] - qb[d])
                if pvar == 'spfhprs':
                    zp[d][m] = zp[d][m].to(units['g/kg'])
                vlim = max(vlim,abs(np.nanmin(zp[d][m].magnitude)),np.nanmax(zp[d][m].magnitude))
        if vlim > 1.0:
            vlim = np.floor(vlim)
        plevels = np.linspace(-vlim,vlim,11)
        if pvar=='spfhprs':
            plevels = np.linspace(-1.0,1.0,11)
        print(f"{pvar}{lev} plevels={plevels}")
        cmap = plt.get_cmap(cmapname)
        cmapuse = cmap(np.linspace(0.0,1.0,plevels.size+1)[1:-1])

        k = 0
        plist = []
        for j, d in enumerate(datadirs.keys()):
            lon = paramsdict[d]['lon']
            lat = paramsdict[d]['lat']
            nlon = lon.size
            nlat = lat.size
            lonlist = paramsdict[d]['lonlist']
            latlist = paramsdict[d]['latlist']
            intuv = paramsdict[d]['intuv']
            intuvp = paramsdict[d]['intuvp']
            for i, m in enumerate(range(smode,emode+1)):
                ax = axs[k]
                k += 1
                p = ax.contourf(lon,lat,zp[d][m],\
                    plevels,transform=ccrs.PlateCarree(),\
                    colors=cmapuse,extend='both',zorder=0)
                p.cmap.set_under(cmap(0.0))
                p.cmap.set_over(cmap(1.0))
                p.changed()
                plist.append(p)
                ax.contour(lon,lat,zb[d],6,\
                    transform=ccrs.PlateCarree(),\
                    colors='k',linewidths=1.0,zorder=1)
                ax.barbs(lon[slice(intuv,nlon,intuv)],lat[slice(intuv,nlat,intuv)],\
                    ub[d][slice(intuv,nlat,intuv),slice(intuv,nlon,intuv)],
                    vb[d][slice(intuv,nlat,intuv),slice(intuv,nlon,intuv)],color='k',
                    length=5,transform=ccrs.PlateCarree(),zorder=2)
                q = ax.quiver(\
                    lon[slice(intuvp,nlon,intuvp)],\
                    lat[slice(intuvp,nlat,intuvp)],\
                    up[d][m][slice(intuvp,nlat,intuvp),slice(intuvp,nlon,intuvp)],\
                    vp[d][m][slice(intuvp,nlat,intuvp),slice(intuvp,nlon,intuvp)],\
                    color=uvcolor,edgecolor='k',scale=20,
                    transform=ccrs.PlateCarree(),zorder=3)
                ax.quiverkey(q, X=0.875, Y=1.05, U=1.0, label=r'1 m s$^{-1}$')
                ## target region
                ax.plot([slon,elon,elon,slon,slon],
                    [slat,slat,elat,elat,slat],
                    lw=0.5,c='k',#ls='dashed',
                    transform=ccrs.PlateCarree())
                ax.add_patch(Rectangle((slon,slat),elon-slon,elat-slat,fc="k",alpha=0.2,
                    transform=ccrs.PlateCarree()))
                
                ax.coastlines(color='tan',lw=0.5)
                ax.set_xticks(lonlist,crs=ccrs.PlateCarree())
                ax.set_yticks(latlist,crs=ccrs.PlateCarree())
                ax.tick_params(labelsize=9)
                ax.grid()

                #if d==dw:
                #    ax.set_title(f"{d}")
                #else:
                #    ax.set_title(f"{d} wgt:{dw}")
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
                    0.0,1.0,f"{caption} {d} mode{m} FT{fh:02d}",
                    transform=(
                        ax.transAxes + ScaledTranslation(-12/72, +7/72, fig.dpi_scale_trans)
                    ),fontsize=12,va='bottom')
        c = fig.colorbar(p,ax=axs,orientation='horizontal',aspect=40,shrink=0.5,pad=0.01)
        c.ax.annotate(unit,xy=(1.1,0.5),xycoords='axes fraction',va='center',fontsize=14)
        c.ax.set_xticks([])
        c.ax.set_xticks([plevels[0],plevels[4],plevels[6],plevels[-1]])
        c.ax.tick_params(labelsize=14)
        fig.savefig(figdir/"figureS1.pdf",dpi=600)
        #fig.suptitle(pvarnames[pvar]+f" {lev} hPa {init.strftime('%Y-%m-%d %HZ')}+{fh:02d}h")
        fig.savefig(figdir/"figureS1.png",dpi=300)
        plt.show(block=False)
        plt.close()
        #exit()
