import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation
import cartopy.crs as ccrs
from metpy.units import units
import metpy.calc as mpcalc
from metpy.interpolate import log_interpolate_1d
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.titlesize'] = 16
import xarray as xr
import re
import os
import sys

titles = {'te':'total','ke':'kinetic','pe':'potential','lh':'latent heat'}

def read_en1(fname,nmem,normalize=False):
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

## custom colormap from NCL spread_15lev
from matplotlib.colors import ListedColormap
color_list = [(255,225,225),(255,210,210),(255,165,165),\
    (255,120,120),(255,75,75),(255,0,0),(255,100,0),(255,150,0),\
    (255,200,0),(255,255,0),(140,255,0),(0,255,0),(0,205,95),\
    (0,145,200)]
ncolors = len(color_list)
color_list = np.array(color_list) / 255.0
mycmap = ListedColormap(color_list,"energy",N=ncolors).with_extremes(over=(0,0,1),under=(1,1,1),bad='gray')

from plotargs import plotargs, regions
#from filt2d import filt2d

from pathlib import Path
from datetime import datetime

config = plotargs()
d = config.domain
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

#fh = config.fh
fstart = config.fstart
fend = config.fend
incfh = config.incfh
fhrange = np.arange(fstart,fend+incfh,incfh)

lon0=config.lon0
lon1=config.lon1
lat0=config.lat0
lat1=config.lat1

slon=config.slon
elon=config.elon
slat=config.slat
elat=config.elat

glob = config.glob
plreg = config.plreg
if glob:
    lon0, lon1, lat0, lat1 = regions[iregtype.upper()]
elif plreg is None:
    if iregtype != 'all':
        plreg = iregtype
    else:
        plreg = d
    lon0, lon1, lat0, lat1 = regions[plreg.upper()]

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
    'D1':hostdir/'d1',
    'D2':hostdir/'d2',
    'D3':hostdir/'d3'
}
headers = {
    'D1':{
        'EnSV':f'{ntype}-{expn}-v{valid:02d}h',
        'NL$+$':f'nl-{ntype}-{expn}-v{valid:02d}h{ext}',
        'NL$-$':f'nl-{ntype}-{expn}-v{valid:02d}h{ext}',
        'NLmean':f'nl-{ntype}-{expn}-v{valid:02d}h{ext}'
    },
    'D2':{
        'EnSV':f'{ntype}-{expn}-v{valid:02d}h',
        'NL$+$':f'nl-{ntype}-{expn}-v{valid:02d}h{ext}',
        'NL$-$':f'nl-{ntype}-{expn}-v{valid:02d}h{ext}',
        'NLmean':f'nl-{ntype}-{expn}-v{valid:02d}h{ext}',
    },
    'D3':{
        'EnSV':f'i{inormtype}allv{vnormtype}-{expn}-v{valid:02d}h',
        'NL$+$':f'nl-i{inormtype}allv{vnormtype}-{expn}-v{valid:02d}h{ext}',
        'NL$-$':f'nl-i{inormtype}allv{vnormtype}-{expn}-v{valid:02d}h{ext}',
        'NLmean':f'nl-i{inormtype}allv{vnormtype}-{expn}-v{valid:02d}h{ext}',
    }
}
suffices = {
    'EnSV':'.te.npy',
    'NL$+$':'.te.npy',
    'NL$-$':'.isign.te.npy',
    'NLmean':'.mean.te.npy',
}
fnames1 = {
    'D1':{
        'EnSV':f'tevol-{ntype}-{expn}-v{valid:02d}h.grd',
        'NL':f'nl-tevol-{ntype}-{expn}-v{valid:02d}h{ext}.grd',
    },
    'D2':{
        'EnSV':f'tevol-{ntype}-{expn}-v{valid:02d}h.grd',
        'NL':f'nl-tevol-{ntype}-{expn}-v{valid:02d}h{ext}.grd'
    },
    'D3':{
        'EnSV':f'tevol-i{inormtype}allv{vnormtype}-{expn}-v{valid:02d}h.grd',
        'NL':f'nl-tevol-i{inormtype}allv{vnormtype}-{expn}-v{valid:02d}h{ext}.grd'
    }
}
centers = {
    1:{'D1':(128.0,30.2),'D2':(128.0,30.2),'D3':(128.0,30.2)},
    2:{'D1':(123.0,29.0),'D2':(122.5,29.8),'D3':(122.5,29.8)},
    3:{'D1':(127.8,30.0),'D2':(124.7,30.5),'D3':(124.7,30.8)},
    4:{'D1':(124.0,30.0),'D2':(127.5,30.0),'D3':(127.0,29.5)},
    5:{'D1':(125.8,30.5),'D2':(124.8,30.1),'D3':(129.5,29.7)},
}
center = centers[smode]
clength = 7.0e5

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
i1=np.argmin(np.abs(lon-lon1tmp))
j0=np.argmin(np.abs(lat-lat0tmp))
j1=np.argmin(np.abs(lat-lat1tmp))
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

endict = dict()
lamdict = dict()
for fh in fhrange:
    endict[fh] = dict()
    for dw in headers[d].keys():
        head=headers[d][dw]
        suffix=suffices[dw]
        fname=f"{head}.f{fh:02d}{suffix}"
        endict[fh][dw] = dict()
        endict[fh][dw]['te']=np.load(datadir/fname)
        #print(endict[fh][dw])
        if fh==fend and dw!='NLmean':
            lamdict[dw] = dict()
            if dw=='EnSV':
                fname1 = fnames1[d][dw]
                print(datadir/fname1)
                data1 = read_en1(datadir/fname1,nmem)
                imem = 0
            else:
                fname1 = fnames1[d]['NL']
                print(datadir/fname1)
                data1 = read_en1(datadir/fname1,2)
                if dw=='NL$+$':
                    imem = 0
                elif dw=='NL$-$':
                    imem = 1
                else:
                    imem = 2
            for v in data1.keys():
                if v=='peps': continue
                evoe0 = data1[v][valid,imem]/data1[v][0,imem]
                lamdict[dw][v] = np.log(evoe0)/float(valid)
print(lamdict)

figdir = Path('.')
if not figdir.exists():
    figdir.mkdir(parents=True)

captions = {
    'EnSV':{
#        1:'(d) ',2:'(e) ',3:'(f) ',
        1:'(a) ',2:'(a) ',3:'(a) ',
    },
    'NL$+$':{
#        1:'(g) ',2:'(h) ',3:'(i) ',
        1:'(b) ',2:'(b) ',3:'(b) ',
    },
    'NL$-$':{
#        1:'(j) ',2:'(k) ',3:'(l) ',
        1:'(c) ',2:'(c) ',3:'(c) ',
    },
    'NLmean':{
        1:'(d) ',2:'(d) ',3:'(d) ',
    }
}
captions = {
    0:'(a) ',3:'(a) ',6:'(b) ',
    9:'(c) ',12:'(d) ',15:'(e) '
}
nrows = 1
ncols = 4
vlim0dict = dict()
for fh in fhrange:
    for v in ['te']:
        fig = plt.figure(figsize=(12,3.1),constrained_layout=True)
        axs = []
        vlim = -999.
        irow = 1
        lam = -999.
        for i, dw in enumerate(endict[fh].keys()):
            icol = i + 1
            iplot = icol
            if glob and d=='D1':
                ax = fig.add_subplot(nrows,ncols,iplot,projection=ccrs.PlateCarree(central_longitude=180.0))
            else:
                ax = fig.add_subplot(nrows,ncols,iplot,projection=ccrs.PlateCarree())
            axs.append(ax)
            #iplot += 1
            if fh==0 or d=='D1':
                z = endict[fh][dw][v]
                if glob:
                    si0, si1, sj0, sj1 = params['sij']
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
            else:
                # assume exponential growth
                try:
                    lam = max(lam,lamdict[dw][v])
                except KeyError:
                    pass
        if fh==0 or d=='D1':
            vlim0dict[v] = vlim
        else:
            vmax = vlim0dict[v] * np.exp(lam*float(fh))
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
        print(f"{v} FT{fh:02d} plevels={plevels}")

        k = 0
        plist = []
        lon = params['lon']
        lat = params['lat']
        nlon = lon.size
        nlat = lat.size
        lonlist = params['lonlist']
        latlist = params['latlist']
        for i, dw in enumerate(endict[fh].keys()):
            ax = axs[k]
            k += 1
            z = endict[fh][dw][v]
            print(z.shape,np.nanmin(z),np.nanmax(z))
            p = ax.contourf(lon,lat,z,\
                    plevels,transform=ccrs.PlateCarree(),\
                    cmap=mycmap,extend='both',zorder=0)
            plist.append(p)

            #ax.plot(center[d][0],center[d][1],lw=0.0,marker='*',c='k',transform=ccrs.PlateCarree())
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
            #if fh<=3:
            #    caption = captions[dw][smode]
            if dw=='EnSV':
                caption = captions[fh]
            else:
                caption = '    '
            ax.text(
                0.0,1.0,f"{caption}{d} {dw} mode{smode:d} FT{fh:02d}",
                transform=(
                    ax.transAxes + ScaledTranslation(-12/72, +7/72, fig.dpi_scale_trans)
                ),fontsize=12,va='bottom'
            )
            #ax.set_title(f"{dw}")
        lcbar = 4 #3
        c = fig.colorbar(p,ax=axs[:lcbar],orientation='horizontal',aspect=40,shrink=0.5,pad=0.01)
        c.ax.annotate(r'J kg$^{-1}$',xy=(1.1,0.5),xycoords='axes fraction',va='center',fontsize=14)
        c.ax.set_xticks([])
        c.ax.set_xticks([plevels[0],plevels[-1]])
        c.ax.tick_params(labelsize=12)
        figtitle=f"{d} mode{expn[1:]} "+titles[v]+f" {init.strftime('%Y-%m-%d %HZ')}+{fh:02d}h"
        fig.savefig(figdir/f"figure3{captions[fh][:-1]}.pdf",dpi=600)
        #fig.suptitle(figtitle)
        fig.savefig(figdir/f"figure3{captions[fh][:-1]}.png",dpi=300)
        plt.show(block=False)
        plt.close()
    #exit()
