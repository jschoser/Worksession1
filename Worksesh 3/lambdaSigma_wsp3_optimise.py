#=================================================================
# AE2220-II - Computational Modelling.
# Analysis program for work session 3 preparation
#
# Line 20: Definition of gammas for 4-stage time march
# Line 18: Definition of the lambda-sigma relation
#
#=================================================================
import numpy as np
import scipy.linalg as spl
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#------------------------------------------------------
# Input parameters
#------------------------------------------------------
nx    = 100;   # Number of mesh points (must be even)
alpha = 0.8;  # Courant number
kdt   = 0.04; # Artificial viscosity parameter * Delta t
g1    = 1;    # LSRK : gamma 1
# g2    = 1/3;    # LSRK : gamma 2
g3    = 0.5;    # LSRK : gamma 3
g4    = 1;    # LSRK : gamma 4

#------------------------------------------------------
# Function for the lambda-sigma relation
#------------------------------------------------------
def lamSig(ldt, g2):
   # sigma = 1 + ldt;               # Euler explicit time march
   #sigma = 1/(1- ldt);            # Euler implicit time march
   #sigma = (1+ldt/2)/(1- ldt/2)   # Trapezoidal time march
   sigma = 1 + g4 * ldt + g3 * g4 * ldt ** 2 + g2 * g3 * g4 * ldt ** 3 + g1 * g2 * g3 * g4 * ldt ** 4
   return sigma

#------------------------------------------------------
# Define the semi-discrete matrix A * Dt
# for the linear advection operator
#------------------------------------------------------

def eval_g2_perf(vars):
   g2 = vars[0]
   AaDt = np.zeros((nx, nx))
   for i in range(0, nx):
      if i == 0:      # Left periodic boundary
         AaDt[i,nx-1] = -1;
         AaDt[i,i]    =  0;
         AaDt[i,i+1]  =  1;
      elif i == nx-1: # Right periodic boundary
         AaDt[i,i-1]  = -1;
         AaDt[i,i]    =  0;
         AaDt[i,0]    =  1;
      else :          # Interior
         AaDt[i,i-1]  = -1;
         AaDt[i,i]    =  0;
         AaDt[i,i+1]  =  1;
   AaDt *= -alpha/2.;

   #------------------------------------------------------
   # Define the semi-discrete matrix A * Dt
   # for linear diffusion * (Delta x)^2
   # (This provides an artificial viscosity which
   #  vanishes with decreasing Delta x.
   #------------------------------------------------------
   AdDt = np.zeros((nx, nx))
   for i in range(0, nx):
      if i == 0:      # Left periodic boundary
         AdDt[i,nx-1] =  1;
         AdDt[i,i]    = -2;
         AdDt[i,i+1]  =  1;
      elif i == nx-1: # Right periodic boundary
         AdDt[i,i-1]  =  1;
         AdDt[i,i]    = -2;
         AdDt[i,0]    =  1;
      else :          # Interior
         AdDt[i,i-1]  =  1;
         AdDt[i,i]    = -2;
         AdDt[i,i+1]  =  1;
   AdDt *= kdt;

   #------------------------------------------------------------
   # Define the total semi-discrete matrix A*DT, then
   # compute the semi-discrete eigenvalues lambda *DT
   # from the expression in the notes for circulant matrices
   #------------------------------------------------------------
   ADt = AaDt + AdDt;
   beta=np.zeros(nx);
   ldt=np.zeros(nx,'complex')
   for m in range(nx):
     beta[m] = 2*np.pi*m/nx;
     if beta[m] > np.pi: beta[m] = 2*np.pi - beta[m]; # negative beta modes
     for j in range(0, nx):
       ldt[m] = ldt[m] + ADt[0,j]*np.exp(1j*2.*np.pi*j*m/nx);

   #------------------------------------------------------------
   # Compute the eigenvalues of C using the lambda-sigma relation,
   # then determine the amplitude and relative phase of each mode
   #------------------------------------------------------------
   sigma  = lamSig(ldt, g2);
   magSig = np.abs(sigma);
   relPse = np.ones(nx);
   for m in range(1,nx):
      relPse[m] = -np.angle(sigma[m])/(alpha*beta[m]);
      if (m>nx/2): relPse[m] = -relPse[m] # negative beta modes
   return sum(np.abs(1 - np.array(relPse[:13])))


inital_guess = [1/3]
result = scipy.optimize.fmin(eval_g2_perf, inital_guess, ftol=1E-5, xtol=1E-5)
print(result)

