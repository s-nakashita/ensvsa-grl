import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
from pathlib import Path
from plotargs import plotargs, regions

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

slon=config.slon
elon=config.elon
slat=config.slat
elat=config.elat

lon0=config.lon0
lon1=config.lon1
lat0=config.lat0
lat1=config.lat1
plreg=config.plreg

glob=config.glob

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
ntypes = {
    'D1':f'i{inormtype}{iregtype}v{vnormtype}',
    'D2':f'i{inormtype}{iregtype}v{vnormtype}',
    'D3':f'i{inormtype}allv{vnormtype}'
}

figdir = Path('.')
if not figdir.exists():
    figdir.mkdir(parents=True)

levels = ['all'] #,'z1-15','z16-30','z31-42']
titles = {'ke':'kinetic','lh':'latent heat','pe':'potential'}
etypes = {
    'theta':r'$\Theta=\frac{\|\mathbf{z}^{+}+\mathbf{z}^{-}\|}{\frac{1}{2}(\|\mathbf{z}^{+}\|+\|\mathbf{z}^{-}\|)}$',
    }
captions = {
    'ke':{
        'D1':'(a) ','D2':'(b) ','D3':'(c) '
    },
    'lh':{
        'D1':'(d) ','D2':'(e) ','D3':'(f) '
    }
}
nrows = emode - smode + 1
nmode = f'm{smode}-{emode}'
colors = plt.get_cmap('tab10')(np.linspace(0.0,0.1*(nrows-1),nrows))
for etype in etypes.keys():
    for v in ['ke']:
        for l in range(len(levels)):
            fig, axs = plt.subplots(ncols=3,figsize=[9,4],sharey=True,constrained_layout=True)
            for d, ax in zip(datadirs.keys(),axs):
                datadir = datadirs[d]
                ntype = ntypes[d]
                for m in range(smode,emode+1):
                    if glob:
                        data = np.load(datadir/f'{etype}-{v}-all-{ntype}-m{m:02d}-v{valid:02d}h{ext}.npy')
                    elif plreg is not None:
                        if d=='D3' and plreg == 'd3':
                            print(datadir/f'{etype}-{v}-all-{ntype}-m{m:02d}-v{valid:02d}h{ext}.npy')
                            data = np.load(datadir/f'{etype}-{v}-all-{ntype}-m{m:02d}-v{valid:02d}h{ext}.npy')
                        else:
                            print(datadir/f'{etype}-{v}-{plreg}-{ntype}-m{m:02d}-v{valid:02d}h{ext}.npy')
                            data = np.load(datadir/f'{etype}-{v}-{plreg}-{ntype}-m{m:02d}-v{valid:02d}h{ext}.npy')
                    else:
                        data = np.load(datadir/f'{etype}-{v}-{ntype}-m{m:02d}-v{valid:02d}h{ext}.npy')
                    taxis = np.arange(data[:,l].size)
                    ax.plot(taxis,data[:,l],c=colors[m-1],label=f'{m}')
                    print(f"{v}({levels[l]}) {d} mode{m} FT15={data[15,l]}")
                try:
                    caption = captions[v][d]
                except KeyError:
                    caption = ''
                ax.set_title(f'{caption}{d}, {v.upper()}',x=0.01,ha='left')
                #ax.hlines([0,1,2],0,1,colors='gray',lw=2.0,transform=ax.get_yaxis_transform(),zorder=0)
                ax.hlines([2],0,1,colors='gray',lw=1.5,transform=ax.get_yaxis_transform(),zorder=0)
                # Hoheneger and Schär (2007, BAMS)
                ax.hlines([np.sqrt(3.0)/2],0,1,colors='gray',lw=3.0,transform=ax.get_yaxis_transform(),zorder=0)
            axs[0].set_ylabel(etypes[etype])
            axs[2].legend(title='mode',loc='upper left',bbox_to_anchor=(1.01,1.0))
            for ax in axs:
                ax.grid()
                ax.vlines([valid],0,1,ls='solid',colors='gray',alpha=0.5,transform=ax.get_xaxis_transform(),zorder=0)
                ax.set_xticks(taxis[::3])
                ax.set_xticklabels(taxis[::3])
                #ax.set_xlabel('forecast hours')
            axs[2].annotate('[h]',xy=(1.05,-0.035),xycoords='axes fraction',ha='left',va='top',fontsize=11)
            figtitle=f"{titles[v]} {init.strftime('%Y-%m-%d %HZ')} "
            figtitle=figtitle+f"lon:{lon0:.1f}-{lon1:.1f} lat:{lat0:.1f}-{lat1:.1f}"
            #fig.suptitle(figtitle+" "+levels[l])
            fig.savefig(figdir/"figureS2.pdf",dpi=600)
            fig.savefig(figdir/"figureS2.png",dpi=300)
            plt.show(block=False)
            plt.close()