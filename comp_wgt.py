import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import stats
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
import argparse
plt.rcParams['font.size'] = 16

def getsv(fname,nlen):
    buf = np.fromfile(fname,dtype=">f4").reshape(-1,nlen)
    return buf[0,]

def getwgt(fname,nlen):
    buf = np.fromfile(fname,dtype=">f4").reshape(-1,nlen)
    return buf[1:,]

parser = argparse.ArgumentParser()
parser.add_argument("-nm","--nmem",type=int,default=40,\
    help="# of ensemble members")
parser.add_argument("-t","--target",type=str,default='',\
    help="ESA target")
parser.add_argument("-vn","--vnorm",type=str,default="te",\
    help="verification norm type")
parser.add_argument("-in","--inorm",type=str,default="te",\
    help="initial norm type")
parser.add_argument("-ir","--ireg",type=str,default='d3',\
    help="initial target region")
parser.add_argument("-i","--init",type=str,default="2022061812",\
    help="initial date (YMDH)")
parser.add_argument("-v","--vt",type=int,default=15,\
    help="verification hours")
parser.add_argument("-inv","--invert",type=bool,default=False,\
    help="sign of weights")
parser.add_argument("-b","--base",type=str,default='cntl',\
    help='base field type')
parser.add_argument("-mmd","--maxmode",type=int,default=5,\
    help="maximum mode for plotting weights")

domain = {'rsm2rsm27_da':'D1','rsm2msm9_da':'D2','msm2msm3_da':'D3','rsm2msm9_dad2':'D2b','msm2msm3_dad3':'D3b'}
expndict = {'d1':'rsm2rsm27_da','d2':'rsm2msm9_da','d3':'msm2msm3_da'}

def calccorr(w1,w2):
    var1 = np.sum((w1-w1.mean())**2)
    var2 = np.sum((w2-w2.mean())**2)
    cov = np.sum((w1-w1.mean())*(w2-w2.mean()))
    return cov / np.sqrt(var1) / np.sqrt(var2)

if __name__ == "__main__":
    argsin = parser.parse_args()
    nmem = argsin.nmem
    target = argsin.target
    vnorm=argsin.vnorm
    inorm=argsin.inorm
    ireg=argsin.ireg
    ymdh = argsin.init
    init = datetime(int(ymdh[0:4]),int(ymdh[4:6]),int(ymdh[6:8]),int(ymdh[8:]))
    valid = argsin.vt
    invert = argsin.invert
    base = argsin.base
    svdict = dict()
    wgtdict = dict()
    dlist = ['d1','d2','d3']
    for d in dlist:
        expn = expndict[d]
        iregtmp = ireg
        if expn=='msm2msm3_da':
            iregtmp = 'all'
        datadir = Path(f'data/{d}')
        wgtfile = f'weight-i{inorm}{iregtmp}v{vnorm}-v{valid:02d}h.grd'
        ndof = nmem
        print(datadir/wgtfile)
        sv = getsv(datadir/wgtfile,ndof)
        svdict[expn] = sv
        wgts = getwgt(datadir/wgtfile,ndof)
        wgtdict[expn] = wgts

    figdir = Path('.')
    if not figdir.exists():
        figdir.mkdir(parents=True)
    #savedir = Path(os.environ['HOME']+'/Writing/dissertation/esa')
    savedir = figdir
    
    fig, axs = plt.subplots(ncols=2,figsize=(10,4),constrained_layout=True)
    xaxis = np.arange(ndof)+1
    width=0.25
    xaxisb = xaxis - width
    icol=0
    cmap=plt.get_cmap('tab10')
    for expn,sv in svdict.items():
        #axs[0].bar(xaxisb,sv**2,width=width,label=domain[expn])
        #xaxisb = xaxisb + width
        axs[0].plot(xaxis,sv**2,label=domain[expn])
        pall = np.sum(sv**2)
        contrib = sv.copy()
        ccf = np.zeros_like(contrib)
        for i in range(ndof):
            contrib[i] = sv[i]**2 / pall
            ccf[i] = np.sum(sv[:i+1]**2) / pall
        axs[1].plot(xaxis,ccf,label=domain[expn])
        axs[1].hlines([ccf[4]],0,1,ls='dashed',colors=cmap(icol),\
        transform=axs[1].get_yaxis_transform())
        icol += 1
        df = pd.DataFrame(
            {
                'eig':sv**2,
                'contrib':contrib,
                'ccf':ccf
            },index=xaxis
        )
        df.to_csv(savedir/f'eig_{domain[expn]}.csv')
    axs[0].set_ylabel(r'$\lambda=\|\mathbf{z}\|^2_{\mathbf{G}_\mathrm{v}}/\|\mathbf{y}\|^2_{\mathbf{G}_\mathrm{a}}$')
    axs[0].set_xlabel('mode')
    axs[0].grid(True,axis='x',zorder=0)
    axs[1].set_ylabel('cumulative fraction')
    axs[1].set_ylim(0.0,1.0)
    axs[1].grid()
    axs[0].legend()
    axs[1].legend()
    if inorm is not None:
        fig.suptitle(f'norm={inorm} reg={ireg} {nmem} member')
    else:
        fig.suptitle(f'{nmem} member')
    fig.savefig(figdir/'eval.png',dpi=300)
    plt.show()
    plt.close()

    # statistically significance test
    tval = stats.t.ppf(0.995, ndof-2) #99%
    r2 = tval*tval / (tval*tval + ndof - 2)
    print(f"99% significance > {np.sqrt(r2):.3f}")
    textcolors = ['black','white']

    nmode = min(ndof,5)
    for i in range(len(dlist)-1):
        clist = []
        for j in range(i+1,len(dlist)):
            expn1 = expndict[dlist[i]]
            expn2 = expndict[dlist[j]]
            w1 = wgtdict[expn1]
            w2 = wgtdict[expn2]
            corrmat = np.zeros((nmode,nmode))
            for irow in range(nmode):
                for icol in range(nmode):
                    corrmat[irow,icol] = calccorr(w1[irow],w2[icol])
            
            fig, ax = plt.subplots(figsize=[8,7],constrained_layout=True)
            im = ax.imshow(corrmat,vmin=-1.0,vmax=1.0,cmap='bwr')
            fig.colorbar(im,ax=ax,shrink=0.6,pad=0.01,label='correlation')
            valfmt = ticker.StrMethodFormatter('{x:.3f}')
            threshold = im.norm(np.sqrt(r2))
            kw={'ha':'center','va':'center','color':'k','fontsize':14}
            for irow in range(nmode):
                for icol in range(nmode):
                    kw.update(color=textcolors[int(im.norm(abs(corrmat[irow,icol]))>threshold)])
                    text = ax.text(icol,irow,valfmt(corrmat[irow,icol],None),\
                        **kw)
            #ax.spines[:].set_visible(False)
            ax.set_xticks(np.arange(nmode))
            ax.set_xticklabels(np.arange(1,nmode+1))
            ax.set_yticks(np.arange(nmode))
            ax.set_yticklabels(np.arange(1,nmode+1))
            ax.set_xticks(np.arange(nmode+1)-.5,minor=True)
            ax.set_yticks(np.arange(nmode+1)-.5,minor=True)
            ax.grid(which='minor',color='w',linestyle='-',lw=3.0)
            ax.tick_params(which='minor',left=False,bottom=False)
            ax.set_title(domain[expn1],fontsize=16)
            ax.set_ylabel(domain[expn2])
            ax.tick_params(top=True,bottom=False,labeltop=True,labelbottom=False)
            if inorm is not None:
                fig.suptitle(f'norm={inorm} reg={ireg} {nmem} member')
            else:
                fig.suptitle(f'{nmem} member')
            fig.savefig(figdir/f'wgtcorr_{domain[expn1]}-{domain[expn2]}.png',dpi=150)
            plt.show()
            plt.close()
            #exit()
            df = pd.DataFrame({'corr':np.diag(corrmat)},index=np.arange(1,nmode+1))
            df.to_csv(savedir/f'wgtcorr_{domain[expn1]}-{domain[expn2]}.csv')
