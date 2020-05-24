#================================================================= 
#
# AE2220-II: Computational Modelling 
# Additional Code for work session 1
#
#=================================================================
# This file includes some additional functions to assist 
# with the problem definition. 
# 
# There is no need to inspect or edit this file to complete
# the work session
#
# For the interested, the flow speeds in the chamber are assumed 
# to be low enough so that the velocity is mostly divergence free. 
# In this case the energy equation (describing e.g. temperature)
# becomes decoupled from the momentum equations. 
# It is still mixed hyperbolic-elliptic, however, so solving
# for temperature to determine the reaction rate would in principle 
# require at least one more PDE in our system, as well as giving up 
# the marching procedure. This goes beyond the level of complexity 
# appropriate for a work session, so we approximate the net effect on the 
# source term in the mass fraction equation using an empirical model.
# 
#=================================================================
import math

#=================================================================
# Returns the stream function at x,y
#=================================================================
def getPsi(x,y):
  psi = -2.0*x - 2.0*math.atan((x)/(y+1.0))     -  6.0*math.atan((x)/(y-15.0)) \
               - 4.0*math.atan((x-2.5)/(y+1.0)) +  2.0*math.atan((x-2.5)/(y-15.0)) \
               - 8.0*math.atan((x+2.5)/(y+1.0)) +  4.0*math.atan((x+2.5)/(y-15.0))
  return psi


#=================================================================
# Returns velocities at x,y
# Depending on the intensity of the velocity field, it 
# may be necessary to adjust k to avoid -ve Z.
#=================================================================
def getUV(x,y):
  dx1=x;     dy1=y+1.0;  r1=math.sqrt(dx1*dx1+dy1*dy1); t1=math.atan2(dx1,dy1);
  dx2=x;     dy2=y-15.0; r2=math.sqrt(dx2*dx2+dy2*dy2); t2=math.atan2(dx2,dy2);
  dx3=x-2.5; dy3=y+1.0;  r3=math.sqrt(dx3*dx3+dy3*dy3); t3=math.atan2(dx3,dy3);
  dx4=x-2.5; dy4=y-15.0; r4=math.sqrt(dx4*dx4+dy4*dy4); t4=math.atan2(dx4,dy4);
  dx5=x+2.5; dy5=y+1.0;  r5=math.sqrt(dx5*dx5+dy5*dy5); t5=math.atan2(dx5,dy5);
  dx6=x+2.5; dy6=y-15.0; r6=math.sqrt(dx6*dx6+dy6*dy6); t6=math.atan2(dx6,dy6);
  u = +math.sin(t1)*2.0/r1 + math.sin(t2)*6.0/r2 \
      +math.sin(t3)*4.0/r3 - math.sin(t4)*2.0/r4 \
      +math.sin(t5)*8.0/r5 - math.sin(t6)*4.0/r6 
  v = +math.cos(t1)*2.0/r1 + math.cos(t2)*6.0/r2 \
      +math.cos(t3)*4.0/r3 - math.cos(t4)*2.0/r4 \
      +math.cos(t5)*8.0/r5 - math.cos(t6)*4.0/r6 + 2.0
  return (u,v)


#=================================================================
# Returns the reaction term based on 
# the local flow state and mass fraction
#=================================================================
def getF(u,v,Z):
  magv=math.sqrt(u*u+v*v)
  FVal = 0.
  if ((magv<6.) and (Z>0.)):
     FVal = - (6.-magv)*0.3*Z;
  return min(0.,FVal)


#=================================================================
# Evaluates the performance of the flame
#=================================================================
def getPerf(imax,nmax,x,y,dx,dy,F):
  perf = 0;
  for n in range(0, nmax-1):
    for i in range(0, imax-1):
       floc=-1000*0.25*(F[i,n]+F[i+1,n]+F[i,n+1]+F[i+1,n+1]);
       xloc=0.25*(x[i,n]+x[i+1,n]+x[i,n+1]+x[i+1,n+1]);
       perf += dx*dy*math.exp(-10.0*(floc-1.0)*(floc-1.0))*math.exp(-5.0*xloc*xloc);

  return perf
