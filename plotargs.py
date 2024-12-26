import argparse
from datetime import datetime
from pathlib import Path

domain = {'rsm2rsm27_da':'D1','rsm2msm9_da':'D2','msm2msm3_da':'D3'}
regions = {
    'D1':(74.67,185.9,-16.77,63.05),
    'D2':(109.1,153.1,14.9,47.0),
    'D3':(120.0,134.6,23.6,34.8)
}
class plotargs():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-e","--expn",type=str,default='rsm2rsm27_da',\
            help="experiment name")
        self.parser.add_argument("-i","--init",type=str,default="2022061812",\
            help='initial date (YMDH)')
        self.parser.add_argument("-nm","--nmem",type=int,default=40,\
            help="ensemble size")
        self.parser.add_argument("-b","--base",type=str,default="cntl",\
            help="base field")
        self.parser.add_argument("-vn","--vnorm",type=str,default="te",\
            help="verification norm")
        self.parser.add_argument("-in","--inorm",type=str,default="te",\
            help="initial norm")
        self.parser.add_argument("-ir","--ireg",type=str,default='d3',\
            help="initial target region")
        self.parser.add_argument("-ext","--ext",type=str,default=None,\
            help="extention for file name")
        self.parser.add_argument("-vt","--valid",type=int,default=15,\
            help="verification time (hours)")
        self.parser.add_argument("-sm","--smode",type=int,default=1,\
            help="first sensitive mode")
        self.parser.add_argument("-em","--emode",type=int,default=None,\
            help="last sensitive mode")

        self.parser.add_argument("-nl","--nlfcst",action="store_true",\
            help="nonlinear perturbed forecast")
        self.parser.add_argument("-is","--isign",action="store_true",\
            help="flip perturbation sign")
        self.parser.add_argument("-pr","--preg",type=str,default="all",\
            help="region for adding perturbation")
        self.parser.add_argument("-pv","--pvar",type=str,default="all",\
            help="variables for adding perturbation")
        self.parser.add_argument("-qadj","--qadj",action="store_true",\
            help="negative moisture and supersaturation adjustment")
        self.parser.add_argument("-sps","--saveps",action="store_true",\
            help="avoid perturbing surface pressure")
        self.parser.add_argument("-fil","--filter",action="store_true",\
            help="spatial filter")
        self.parser.add_argument("-fcl","--fclen",type=int,default=1000,\
            help="correlation length scale for spatial filter [km]")

        self.parser.add_argument("-t","--target",type=str,default='',\
            help="ESA target")
        self.parser.add_argument("-slon","--slon",type=float,default=127.0,\
            help="western edge of target region")
        self.parser.add_argument("-elon","--elon",type=float,default=130.0,\
            help="eastern edge of target region")
        self.parser.add_argument("-slat","--slat",type=float,default=29.5,\
            help="southern edge of target region")
        self.parser.add_argument("-elat","--elat",type=float,default=32.5,\
            help="northern edge of target region")
        self.parser.add_argument("-lon0","--lon0",type=float,default=-999.,\
            help="western edge of plotted region")
        self.parser.add_argument("-lon1","--lon1",type=float,default=999.,\
            help="eastern edge of plotted region")
        self.parser.add_argument("-lat0","--lat0",type=float,default=-999.,\
            help="southern edge of plotted region")
        self.parser.add_argument("-lat1","--lat1",type=float,default=999.,\
            help="northern edge of plotted region")
        self.parser.add_argument("-plr","--plreg",type=str,default="d3",\
            help="plotted region type")

        self.parser.add_argument("-fs","--fstart",type=int,default=0,\
            help="initial forecast hour")
        self.parser.add_argument("-fe","--fend",type=int,default=0,\
            help="final forecast hour")
        self.parser.add_argument("-ifh","--incfh",type=int,default=1,\
            help="interval for plotting (hours)")
        self.parser.add_argument("-fh","--fh",type=int,default=0,\
            help="forecast hour")

        self.parser.add_argument("-ddir","--datadir",type=str,default='.',\
            help="data directory")
        self.parser.add_argument("-fdir","--figdir",type=str,default='.',\
            help="figure directory")
        self.parser.add_argument("-bdir","--basedir",type=str,default='.',\
            help="base directory")

        self.parser.add_argument("-wdh","--wdh",type=str,default='',\
            help="file header")
        
        self.parser.add_argument("-norm","--normalize",action="store_true",\
            help="(for plot_tevol) normalization")
        self.parser.add_argument("-gl","--glob",action="store_true",\
            help="(for plot_enprof) calculate in entire domain")
        self.parser.add_argument("-v","--var",type=str,default="ke",\
            help="(for plot_spectra) plotted variable")

        argsin = self.parser.parse_args()
        self.expn = argsin.expn
        self.domain = domain[self.expn]
        ymdh = argsin.init 
        self.init = datetime(int(ymdh[:4]),int(ymdh[4:6]),int(ymdh[6:8]),int(ymdh[8:]))
        self.nmem = argsin.nmem
        self.base = argsin.base
        self.vnormtype = argsin.vnorm
        self.inormtype = argsin.inorm
        if self.inormtype is not None:
            self.generalized = True
        else:
            self.generalized = False
        self.iregtype = argsin.ireg
        if self.iregtype == self.domain.lower():
            self.iregtype = 'all'
        self.ext = argsin.ext

        self.valid = argsin.valid
        self.smode = argsin.smode
        self.emode = argsin.emode
        if self.emode is None:
            self.emode = self.smode

        self.nlfcst = argsin.nlfcst
        self.isign = argsin.isign
        self.preg = argsin.preg
        self.pvar = argsin.pvar
        self.qadj = argsin.qadj
        self.sps = argsin.saveps
        self.fil = argsin.filter
        self.fcl = argsin.fclen
        if self.ext is None:
            self.ext = ''
            if self.fil:
                self.ext = f'.fil{self.fcl}'
            if self.sps:
                if self.ext == '':
                    self.ext = '.nops'
                else:
                    self.ext = '.nops'+self.ext
            if self.qadj:
                if self.ext == '':
                    self.ext = '.qadj'
                else:
                    self.ext = '.qadj'+self.ext
            if self.pvar != 'all':
                if self.ext == '':
                    self.ext = '.' + self.pvar
                else:
                    self.ext = '.' + self.pvar + self.ext
            if self.preg != 'all':
                if self.ext == '':
                    self.ext = '.' + self.preg
                else:
                    self.ext = '.' + self.preg + self.ext

        self.target = argsin.target
        self.slon = argsin.slon
        self.elon = argsin.elon
        self.slat = argsin.slat
        self.elat = argsin.elat

        self.lon0 = argsin.lon0
        self.lon1 = argsin.lon1
        self.lat0 = argsin.lat0
        self.lat1 = argsin.lat1
        self.plreg = argsin.plreg
        if self.plreg is not None and \
            self.lon0 == -999. and self.lon1 == 999. \
                and self.lat0 == -999. and self.lat1 == 999.:
            if self.plreg == 'all':
                self.lon0, self.lon1, self.lat0, self.lat1 = regions[self.domain]
            else:
                self.lon0, self.lon1, self.lat0, self.lat1 = regions[self.plreg.upper()]

        self.fstart = argsin.fstart
        self.fend = argsin.fend
        self.incfh = argsin.incfh
        self.fh = argsin.fh

        self.datadir = Path(argsin.datadir)
        self.figdir = Path(argsin.figdir)
        self.basedir = Path(argsin.basedir)

        self.wdh = argsin.wdh

        self.normalize = argsin.normalize
        self.glob = argsin.glob
        self.var = argsin.var
