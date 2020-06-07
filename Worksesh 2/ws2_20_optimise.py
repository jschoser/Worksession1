##================================================================= 
#
# AE2220-II: Computational Modelling 
# Main program for work session 2
#
# Line 92:  Definition of f for manufactured solution
# Line 110: Definition of Ke[i,j]
#
#
#=================================================================
# This code provides a base for computing the Laplace equation
# with a finite-element method based on triangles 
# with linear shape functions.
#=================================================================
import math
import numpy as np
import matplotlib.pyplot as plt
import TriFEMLibGF
from scipy.optimize import fmin

0
def find_dev(vars):
  #=========================================================
  # Input parameters
  #=========================================================
  n=4                # Mesh refinement factor
  a=vars[0]              # ice thickness amplitude
  b=vars[1]               # Displacement of minimum from zero

  #=========================================================
  # Fixed parameters
  #=========================================================
  xmQ,ymQ =-10, 10.5     # Position of satellite Q
  xmS,ymS = 10, 9.39     # Position of satellite S
  urQ=202.08943488261804 # satellite potential at location Q
  urS=147.08534204993452 # satellite potential at location S
  yIce=2.0;              # Upper ice boundary

  #=========================================================
  # Create the mesh
  #=========================================================
  mesh = TriFEMLibGF.TriMesh();
  mesh.loadMesh(n, yIce, a, b)
  # print ("Mesh: nVert=",mesh.nVert,"nElem=",mesh.nElem);
  #mesh.plotMesh();# quit(); 1

  #=========================================================
  # Create a finite-element space.
  # This object maps the degrees of freedom in an element
  # to the degrees of freedom of the global vector.
  #=========================================================
  fes = TriFEMLibGF.LinTriFESpace(mesh)

  #=========================================================
  # Prepare the global left-hand matrix, right-hand vector
  # and solution vector
  #=========================================================
  sysDim = fes.sysDim
  LHM    = np.zeros((sysDim,sysDim));
  RHV    = np.zeros(sysDim);
  solVec = np.zeros(sysDim);

  #=========================================================
  # Assemble the global left-hand matrix and
  # right-hand vector by looping over the elements
  print ("Assembling system of dimension",sysDim)
  #=========================================================

  for elemIndex in range(mesh.nElem):

    #----------------------------------------------------------------
    # Create a FiniteElement object for
    # the element with index elemIndex
    #----------------------------------------------------------------
    elem = TriFEMLibGF.LinTriElement(mesh,elemIndex)

    #----------------------------------------------------------------
    # Initialise the element vector and matrix to zero.
    # In this case we have only one unknown varible in the PDE (u),
    # So the element vector dimension is the same as
    # the number of shape functions (psi_i)  in the element.
    #----------------------------------------------------------------
    evDim   = elem.nFun
    elemVec = np.zeros((evDim))
    elemMat = np.zeros((evDim,evDim))

    #----------------------------------------------------------------
    # Evaluate the shape function integrals in the vector and matrix
    # by looping over integration points (integration by quadrature)
    # int A = sum_ip (w_ip*A_ip) where A is the function to be
    # integrated and w_ip is the weight of an integration point
    #----------------------------------------------------------------
    for ip in range(elem.nIP):

      # Retrieve the coordinates and weight of the integration point
      xIP  = elem.ipCoords[ip,0]
      yIP  = elem.ipCoords[ip,1]
      ipWeight = elem.ipWeights[ip];

      yWat = TriFEMLibGF.iceWat(xIP, a, b)

      # Compute the local value of the source term, f
      if yIP<=yWat :
        fIP = 0
      elif yIP<=yIce :
        fIP = -200
      else :
        fIP = 0.


      # Retrieve the gradients evaluated at this integration point
      # - psi[i] is the value of the function psi_i at this ip.
      # - gradPsi[i] is a vector contraining the x and y
      #   gradients of the function psi_i at this ip
      #   e.g.
      #     gradPsi[2][0] is the x gradient of shape 2 at point xIP,yIP
      #     gradPsi[2][1] is the y gradient of shape 2 at point xIP,yIP
      psi     = elem.getShapes(xIP,yIP)
      gradPsi = elem.getShapeGradients(xIP,yIP)


      # Add this ip's contribution to the integrals in the
      # element vector and matrix
      for i in range(evDim):
        elemVec[i] += ipWeight*psi[i]*fIP;   # Right-hand side of weak form
        for j in range(evDim):
          # ***** Change the line below for the desired left-hand side
          # elemMat[i,j] += 1.
          elemMat[i, j] += - ipWeight * (gradPsi[i][0] * gradPsi[j][0] + gradPsi[i][1] * gradPsi[j][1])


    #----------------------------------------------------------------
    # Add the completed element matrix and vector to the system
    #----------------------------------------------------------------
    fes.addElemMat(elemIndex, elemMat, LHM )
    fes.addElemVec(elemIndex, elemVec, RHV )

  #=========================================================
  print ("Applying boundary conditions")
  #=========================================================
  # Lower boundary conditions
  #=========================================================
  coord = np.asarray(fes.lowerCoords)[:,0]
  for i in range(fes.nLower):
     row = int(fes.lowerDof[i]);
     LHM[row,:]   = 0.
     LHM[row,row] = 1.
     RHV[row]     = 0.


  #=========================================================
  print ("Solving the system")
  #=========================================================
  #fes.printMatVec(LHM,RHV,"afterConstraints")
  solVec = np.linalg.solve(LHM, RHV)


  #=========================================================
  # Find and output potential and the difference from the
  # reference values at the measurement points
  #=========================================================
  umQ  =fes.getLeftBndU(solVec,   ymQ)
  umS  =fes.getRightBndU(solVec,  ymS)
  return math.sqrt( (umQ-urQ)*(umQ-urQ) + (umS-urS)*(umS-urS) )


initial =  [0.73872975, 2.93201468]
print()
result = fmin(find_dev, initial, xtol = 0.1, ftol = 0.1)
print(result)
