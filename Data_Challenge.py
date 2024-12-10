import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter, NullFormatter 
plt.rc('font', family='serif') 
plt.rc('mathtext',fontset='cm')
from astropy.io import ascii
import scipy.special as sps
import scipy.integrate as integrate
import scipy.optimize as opt
import random

data = ascii.read('/Users/mjbandem/Data Challenge/dc-data_set.txt')
ndata = len(data)
x = data['col1'] ; y = data['col2'] ; ey = data['col3']



def power(x, A, P, phi, C):
    y = A*np.sin((2*np.pi*x)/P + phi) + C
    return y

def Chi2(model,*args):
  x = np.array(args[0])
  data = np.array(args[1])
  edata = np.array(args[2])

  chi2 = sum( (data-model)**2/edata**2 )
  return chi2

def mcmc(x,y,ey, x0,nmcmc):
  args = [x, y, ey]
  # define first values
  A0=x0[0] ; P0=x0[1] ; phi0=x0[2] ; C0=x0[3] ; dof=len(x)-len(x0)
  # define how to step through parameter space
  scale = 0.5
  Astep = 0.5 #scale*np.sqrt(np.var((y-C0)/np.sin((2*np.pi*x)/P0 - phi0),ddof=1))
  Pstep = 0.1
  phistep = scale*np.sqrt(np.var(np.arcsin((y-C0)/A0)/(2*np.pi*x)-1/P0, ddof=1))
  Cstep = 0.4 #scale*np.sqrt(np.var(y-A0*np.sin((2*np.pi*x)/P0 - phi0),ddof=1))
  print(Astep, Pstep, phistep, Cstep)
  mod0 = power(x, A0, P0, phi0, C0)
  Chi0 = Chi2(mod0,*args)
  nadjust = round(0.2*nmcmc)
  Achain = [A0] ; Pchain = [P0] ; phichain = [phi0] ; Cchain=[C0] ; Chichain = [Chi0] ; it=1 # define chains of accepted values
  for ii in range(nmcmc):
    if ii==0: Ac=A0 ; Pc=P0 ; phic=phi0 ; Cc=C0 ; Chic=Chi0  # on first step, use guess as current value
        

    if ii==nadjust:
      Astep=0.5*np.sqrt(np.var(Achain,axis=0)) # adjust to sqrt of variance
      Pstep=0.5*np.sqrt(np.var(Pchain,axis=0)) # adjust to sqrt of variance
      phistep=0.5*np.sqrt(np.var(phichain,axis=0))
      Cstep=0.5*np.sqrt(np.var(Cchain,axis=0))
      oo=Chichain==min(Chichain) # identify best fit thus far
      Ac=Achain[oo][0] # Reset to best value yet
      Pc=Pchain[oo][0] # Reset to best value yet
      phic=phichain[oo][0]
      Cc=Cchain[oo][0]
      print('Adjusting Steps to = ' + str([Astep,Pstep,phistep,Cstep]))

    
    # Perturb values away from their current chain values
    At=np.random.normal(Ac, Astep) # trial perturbation
    Pt=np.random.normal(Pc, Pstep)
    phit=np.random.normal(phic, phistep)
    Ct=np.random.normal(Cc, Cstep)

    if phit < -2*np.pi : phit = phic
    if phit > 2*np.pi : phit = phic
 
    modt = power(x, At, Pt, phit, Ct)
    Chit = Chi2(modt,*args)
    lnprat = 0.5*(Chic-Chit)
    uu = random.uniform(0,1)  # Draw random number between 0 and 1
    if np.log(uu)<lnprat:   # "Accept" trial if U<ratio of P2/P1
      Ac=At ; Pc=Pt ; phic=phit ; Cc=Ct ; Chic=Chit  # reset current chain values
      it = it + 1
        
    #print(Cc, Ct, lnprat) # can run this to see if the current value is changing 

    Achain=np.append(Achain,Ac)
    Pchain=np.append(Pchain,Pc)
    phichain=np.append(phichain,phic)
    Cchain=np.append(Cchain,Cc)
    Chichain=np.append(Chichain,Chic)
    
  print('Fraction of Accepted Trials: '+str(it/nmcmc))

  return [Achain,Pchain,phichain,Cchain,Chichain]

x0 = [30, 5, 0, 85]
nu = ndata - len(x0)
nmcmc = 20000
A_mcmc, P_mcmc, phi_mcmc, C_mcmc, Chi2_mcmc = mcmc(x,y,ey,x0,nmcmc)

#Display MCMC chains and burn cutoff

Nmax = 2000 # Adjust this to plot up the first Nmax trials and search for convergence

fig = plt.figure(figsize=[8,10])
plt.subplots_adjust(left=0.12,bottom=0.12,right=0.97,top=0.99)

gs = gridspec.GridSpec(5,1,height_ratios=[1,1,1,1,1])
gs.update(hspace=0.00)

ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0],sharex=ax1)
ax3 = plt.subplot(gs[2,0],sharex=ax1)
ax4 = plt.subplot(gs[3,0],sharex=ax1)
ax5 = plt.subplot(gs[4,0],sharex=ax1)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax4.get_xticklabels(), visible=False)

# set tick attributes (direction inwards, tick width and length for both axes)
ylabels = [r'$A$',r'$P$',r'$\phi$', r'$C$', r'$\chi^2$']
xlabels = [' ',' ' , ' ', ' ', 'Trial Number']
it = 0
fnt=18
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_xlabel(xlabels[it],fontsize=fnt)
    ax.set_ylabel(ylabels[it],fontsize=fnt)
    ax.minorticks_on()
    ax.tick_params(which='both',axis='both',direction='in',top='True',right='True',labelsize=14)
    ax.tick_params(which='major', length=10)
    ax.tick_params(which='minor', length=5)
    it = it+1

i10 = 250

l1, = ax1.plot(np.linspace(1,len(A_mcmc),len(A_mcmc)),A_mcmc,color='black', label = 'walker 1')
b1, = ax1.plot([i10,i10],[-1,1000],linestyle='dotted',color='g')
ax1.set_xlim(0,Nmax)
ax1.set_ylim(min(A_mcmc), max(A_mcmc))

l2, = ax2.plot(np.linspace(1,len(P_mcmc),len(P_mcmc)),P_mcmc,color='black', label = 'walker 1')
b2, = ax2.plot([i10,i10],[-1,1000],linestyle='dotted',color='g')
ax2.set_xlim(0,Nmax)
ax2.set_ylim(min(P_mcmc), max(P_mcmc))

l3, = ax3.plot(np.linspace(1,len(phi_mcmc),len(phi_mcmc)),phi_mcmc,color='black', label = 'walker 1')
b3, = ax3.plot([i10,i10],[-1,1000],linestyle='dotted',color='g')
ax3.set_xlim(0,Nmax)
ax3.set_ylim(0,2*np.pi)

l4, = ax4.plot(np.linspace(1,len(C_mcmc),len(C_mcmc)),C_mcmc,color='black', label = 'walker 1')
b4, = ax4.plot([i10,i10],[-1,1000],linestyle='dotted',color='g')
ax4.set_xlim(0,Nmax)
ax4.set_ylim(min(C_mcmc), max(C_mcmc))

l5, = ax5.plot(np.linspace(1,len(Chi2_mcmc),len(Chi2_mcmc)),Chi2_mcmc,color='black')
b5, = ax5.plot([i10,i10],[0.01,1e5],linestyle='dotted',color='g')

ax5.semilogy()
ax5.set_ylim(0.3,max(Chi2_mcmc))

plt.savefig('/Users/mjbandem/Data Challenge/chains.png')

Aburn=A_mcmc[i10:] ; Pburn=P_mcmc[i10:] ; phiburn=phi_mcmc[i10:] ; Cburn=C_mcmc[i10:] ; Chiburn=Chi2_mcmc[i10:] ;                                                                                             
oo=Chiburn==min(Chiburn) # identify best fit thus far
Abest=Aburn[oo][0] # Reset to best value yet
Pbest=Pburn[oo][0]
phibest=phiburn[oo][0]
Cbest=Cburn[oo][0]
Chibest=Chiburn[oo][0]
print(Abest, Pbest, phibest, Cbest,Chibest)

import corner

# Set up the parameters of the problem.
data = np.transpose(np.array([Aburn, Pburn, phiburn, Cburn]))

# Plot it using corner
figure = corner.corner(
    data,
    labels=[
        r"$A$",
        r"$P$",
        r"$\phi$",
        r"$C$",
   ],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 14},
)
figure.savefig('/Users/mjbandem/Data Challenge/DCcorner.png')
params = np.zeros(4)
con16 = np.zeros(4)
con84 = np.zeros(4)
paramname = ['A', 'P' , 'phi' , 'C']
for ii in range(4):
    q_16, q_50, q_84 = corner.quantile(data[:,ii], [0.16, 0.5, 0.84])
    dx_down, dx_up = q_50-q_16, q_84-q_50
    params[ii] = q_50 ; con16[ii] = dx_down ; con84[ii] = dx_up
    print('Median value  and 16-84% confidence interval of ' + paramname[ii] + ': ' + str(params[ii]) + ' +' + str(con84[ii]) + ' -' + str(con16[ii]) )

xx = np.linspace(0, 16, 1000)
yy = power(xx, params[0], params[1], params[2], params[3])

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.17,bottom=0.15,right=0.97,top=0.97)

fnt=18
ax.set_xlim(0, np.max(x)+1)
ax.set_ylim(np.min(y)-5, np.max(y)+5)
ax.set_xlabel(r'$x$',fontsize=fnt)
ax.set_ylabel(r'$y$',fontsize=fnt)

ax.tick_params(which='both',axis='both',direction='in',top=True,right=True,labelsize=14)
ax.tick_params(which='major', length=10)
ax.minorticks_on()
ax.tick_params(which='minor', length=5)

ax.errorbar(x, y, ey, fmt = 'o', label=r'data')
ax.plot(xx,yy, label=r'model')

ax.legend(loc='upper center',ncol=1,frameon=False,fontsize='large')

plt.savefig('/Users/mjbandem/Data Challenge/data_and_Model.png')

def chi2_dist(nu):
  ''' Build Chi^2 distribution Function given degrees of freedom nu '''
  npts = 1000
  sig_q = np.sqrt(2*nu) # variance of the chi^2 distribution
  qq=np.linspace(0,nu+10*sig_q,npts)
  Pq = 1/2**(nu/2)*1./sps.gamma(nu/2)*np.exp(-qq/2)*qq**(nu/2 - 1.)

  CDF = np.zeros(npts)
  for ii in range(npts):
    fint = lambda x:1/2**(nu/2)*1./sps.gamma(nu/2)*np.exp(-x/2)*x**(nu/2 - 1.)
    intf = integrate.quad(fint, 0, qq[ii])
    CDF[ii] = intf[0]

  return [qq,Pq,CDF]

#assuming the model is a the null hypothesis
Pq = chi2_dist(nu)
pnull = 1-np.interp(Chibest,Pq[0],Pq[2])
print("The Pnull (assuming the model is the Null): " + str(pnull))