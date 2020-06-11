#=================================================================
# AE2220-II Computational Modelling.
# Linearised acoustics solver
# RKLS in time, 2nd order central + artificial visc in space
#=================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as pltani

#------------------------------------------------------
# Input parameters
#------------------------------------------------------
nx         = 1000;         # number of mesh points (min 50)
alpha      = 0.8;         # c delta t / delta x
kdt        = 0.04;        # Artificial diffusion*dt*dx*dx
glsrk      = np.array([1.0, -0.27159831, 0.5, 1]) # g1, g2, g3, g4 for LSRK
c_pse = 0.7292282756505188
c_amp = -0.2707086185437866

#------------------------------------------------------
# Fixed parameters
#------------------------------------------------------
plot_skip  = 10;          # Plot every plot_skip time steps
plot_int   = 50;          # Plot display interval [ms]
ro         = 1;           # Reference density
co         = 1;           # Reference speed of sound
xSrc       = 0.3;         # Distance of source to wall
wSrc       = 0.05;        # Width of source

#------------------------------------------------------
# Computed parameters
#------------------------------------------------------
deltaX = 1./(nx-1);                    # Mesh spacing
deltaT = alpha*deltaX;                 # delta t for c=1
nt     = int(4./deltaT);               # number of time steps
nStage = len(glsrk);                   # Number of RK stages

#------------------------------------------------------
# Make mesh and initialise solution vectors 
#------------------------------------------------------
x = np.linspace(0, 1.0, nx)  # Mesh coordinate vector
R1_n   = np.zeros(nx);       # Invariant 1 at time level n
R1_st  = np.zeros(nx);       # Invariant 1 for marching
R1_np1 = np.zeros(nx);       # Invariant 1 at time level n+1
R2_n   = np.zeros(nx);       # Invariant 2 at time level n
R2_st  = np.zeros(nx);       # Invariant 2 for marching
R2_np1 = np.zeros(nx);       # Invariant 2 at time level n+1
u_np1  = np.zeros(nx);       # Velocity at time level n+1
p_np1  = np.zeros(nx);       # Pressure at time level n+1


#------------------------------------------------------
# Acoustic source
#------------------------------------------------------
def src(xp,t):
   f = 0.; fon=0;
   if abs(np.sin(5*t-0.21))>0.95:
    fon = 0.05
    xs=xSrc-xp;
    if abs(xs)<wSrc/2.:
       f = 20*np.sin(0.5*np.pi*xs/wSrc)**2;
   return f, fon

#------------------------------------------------------
# Controller
#------------------------------------------------------
def control(t):
   c = c_amp*np.exp( -15*(np.sin(5*(t-c_pse)) )**2);
   if (t<0.6): c=0;
   return c
 

#------------------------------------------------------
#  Prepare for time march
#------------------------------------------------------
t =0;                                  # Set initial time
plot_t   = np.array([0])               # times for animated plot
plot_f   = np.array([0])               # mid-src forcing signal
plot_c   = np.array([0])               # controller pressure signal
plot_m   = np.array([0])               # microphone pressure signal
plot_u   = np.matrix(R1_n)             # velocity distributions
plot_p   = np.matrix(R2_n)             # pressure distributions
rms_m    = 0.                          # Microphone rms pressure
nrms     = 0.                          # Number of rms contributions
fon      = 0.                          # Force on indicator


#------------------------------------------------------
#  March for nt steps , while saving solution
#------------------------------------------------------
print('------------------------------------------------------------------')
print ('Marching for',nt,'time steps with a',nStage,'stage RKLS scheme');
print ('------------------------------------------------------------------')
for n in range(nt):

  for st in range(nStage):  

    tst = t+deltaT*glsrk[st];  # Evaluation time for this stage

    if (st==0): 
      R1_st=R1_n;   R2_st=R2_n;
    else:
      R1_st=R1_np1; R2_st=R2_np1;

    # Interior update
    for i in range(1, nx-1):
      f,fon = src(x[i],tst); 
      R1_np1[i] = (R1_n[i] + glsrk[st]*(-alpha*(R1_st[i+1]-R1_st[i-1])/2
                                        +kdt*(R1_n[i+1]-2*R1_n[i]+R1_n[i-1]) 
                                        +deltaT*f ));
      R2_np1[i] = (R2_n[i] + glsrk[st]*(+alpha*(R2_st[i+1]-R2_st[i-1])/2 
                                        +kdt*(R2_n[i+1]-2*R2_n[i]+R2_n[i-1]) 
                                        +deltaT*f ));

    # Numerical conditions
    R1_np1[-1] = R1_n[-1] -glsrk[st]*alpha*(R1_st[-1]-R1_st[-2]);  # Right
    R2_np1[0]  = R2_n[0]  +glsrk[st]*alpha*(R2_st[1] -R2_st[0] );  # Left

    # Actuator on wall
#   R1_np1[0] = R2_np1[0] + control(tst);
    # Actuator in flow
    R1_np1[0] = control(tst);
      
  t = t + deltaT; 
  print ('n={:4d}   t={:10f}\r'.format(n,t),)   
  R1_n=np.copy(R1_np1);               # Replace solution 
  R2_n=np.copy(R2_np1);               # Replace solution 
  p_np1=(R1_np1+R2_np1)/2;
  u_np1=(R1_np1-R2_np1)/(2*ro*co);
  if (t>2.0):
     rms_m += p_np1[-1]*p_np1[-1];
     nrms  += 1;

  # Save time and solution for plotting
  if (n % plot_skip) == 0 or n == nt-1:
      plot_t = np.append( plot_t, t)        
      plot_f = np.append( plot_f, fon)        
      plot_c = np.append( plot_c, R1_np1[0]/10.)        
      plot_m = np.append( plot_m, p_np1[-1])        
      plot_u = np.vstack((plot_u, u_np1)) 
      plot_p = np.vstack((plot_p, p_np1)) 

print ('------------------------------------------------------------------')
print ('RMS amplitude of exit pressure:',np.sqrt(rms_m/float(nrms)))
print ('------------------------------------------------------------------')


#------------------------------------------------------
#  Make animated plot
#------------------------------------------------------
fig = plt.figure(figsize=(14,6))

ax1  = fig.add_subplot(121)
ax1.grid();ax1.set_ylim([-0.4, 0.6])
alx   = np.array([0.,0]);
aly   = np.array([-0.01,0.01]);
slx   = np.array([xSrc-wSrc/2.,xSrc+wSrc/2.]);
sly   = np.array([0.,0.]);
cpnt  = ax1.plot(alx, aly, '-g', lw=10 )[0]
aline = ax1.plot(alx, aly, '-g', lw=10 )
sline = ax1.plot(slx, sly, '-r', lw=10 )
pline = ax1.plot(x, x,'-b', label='p')[0]
time  = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)

ax2 = fig.add_subplot(122)
ax2.grid(); ax2.set_xlim([0, plot_t[-1]])
fline = ax2.plot(plot_t[0], plot_f[0], label='Source')[0]
cline = ax2.plot(plot_t[0], plot_c[0], label='Control')[0]
mline = ax2.plot(plot_t[0], plot_m[0], label='p(1)')[0]

def init():

    cpnt.set_ydata (np.ma.array(alx,mask=True))
    pline.set_ydata(np.ma.array(x,mask=True))
    fline = ax2.plot(plot_t[0], plot_f[0], '-r', label='Source'  );
    cline = ax2.plot(plot_t[0], plot_c[0], '-g', label='Control' );
    mline = ax2.plot(plot_t[0], plot_m[0], '-b', label='p(1)'    );
    time.set_text('')
    return  cpnt, pline, fline, cline, mline, time

def step(i):
    ax1.legend(); 
    cpnt.set_ydata(np.array([-0.01,plot_c[i]*10]));
    pline.set_ydata(plot_p[i])
    ax2.clear(); ax2.grid();
    ax2.set_xlim([0, plot_t[-1]]);ax2.set_ylim([-0.2,0.2]);
    fline = ax2.plot(plot_t[:i], plot_f[:i], '-r', label='Source'  );
    cline = ax2.plot(plot_t[:i], plot_c[:i], '-g', label='Control' );
    mline = ax2.plot(plot_t[:i], plot_m[:i], '-b', label='p(1)'    );
    ax2.legend();
    time.set_text('time = %.3fs'%(plot_t[i]))
    return cpnt, pline, fline, cline, mline, time

ani = pltani.FuncAnimation(fig, step, plot_u.shape[0], 
                           init_func=init,
                           interval=plot_int)
plt.show()

