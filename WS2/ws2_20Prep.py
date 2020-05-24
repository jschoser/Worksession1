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
import TriFEMLibGPrep

#=========================================================
# Input parameters
#=========================================================
n=8            # Mesh refinement factor
a=1.           # Manufactured solution a
b=1.           # Manufactured solution b


#=========================================================
# Create the mesh 
#=========================================================
mesh = TriFEMLibGPrep.TriMesh();
aaa = mesh.loadMesh(n)
print ('Mesh: nVert=',mesh.nVert)
print('nElem=',mesh.nElem)
#mesh.plotMesh();# quit(); 1

#=========================================================
# Create a finite-element space.
# This object maps the degrees of freedom in an element
# to the degrees of freedom of the global vector.
#=========================================================
fes = TriFEMLibGPrep.LinTriFESpace(mesh)

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
print ('Assembling system of dimension:',sysDim)
#=========================================================

for elemIndex in range(mesh.nElem):

  #----------------------------------------------------------------
  # Create a FiniteElement object for 
  # the element with index elemIndex
  #----------------------------------------------------------------
  elem = TriFEMLibGPrep.LinTriElement(mesh,elemIndex)

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


    # Compute the local value of the source term, f
    # ***** For the manufactured solution, add the appropriate value below
    fIP = -np.sin(a * xIP) * np.sin(b * yIP) * (a ** 2 + b ** 2)


    # Retrieve the gradients evaluated at this integration point
    # - psi[i] is the value of the function psi_i at this ip.
    # - gradPsi[i][j] is a vector contraining the x and y
    #   gradients of the function psi_i at this ip
    # e.g.
    #  gradPsi[2][0] is the gradient of shape 2 [2] along x [0] at point xIP,yIP
    #  gradPsi[2][1] is the y gradient of shape 2 [2] along y [1] at point xIP,yIP
    psi     = elem.getShapes(xIP,yIP)
    gradPsi = elem.getShapeGradients(xIP,yIP)
	

    # Add this ip's contribution to the integrals in the
    # element vector and matrix
    for i in range(evDim):
      elemVec[i] += ipWeight*psi[i]*fIP;   # Right-hand side of weak form
      for j in range(evDim):
        # ***** Change the line below for the desired left-hand side
        # elemMat[i,j] += -ipWeight*psi[i]*psi[j]
        elemMat[i, j] += -ipWeight * (gradPsi[i][0] * gradPsi[j][0] + gradPsi[i][1] * gradPsi[j][1])

  
  #----------------------------------------------------------------
  # Add the completed element matrix and vector to the system
  #----------------------------------------------------------------
  fes.addElemMat(elemIndex, elemMat, LHM )
  fes.addElemVec(elemIndex, elemVec, RHV ) 
		
#=========================================================
print ('Applying manufactured boundary conditions')
#=========================================================


#=========================================================
# Lower boundary conditions
#=========================================================
coord = np.asarray(fes.lowerCoords)[:,0]
for i in range(fes.nLower):
   xy  = fes.lowerCoords[i]; 
   row = int(fes.lowerDof[i]);
   LHM[row,:]   = 0.
   LHM[row,row] = 1.
   RHV[row]     = math.sin(a*xy[0])*math.sin(b*xy[1])


#=========================================================
# Upper boundary conditions
#=========================================================
coord = np.asarray(fes.upperCoords)[:,0]
for i in range(fes.nUpper):
   xy  = fes.upperCoords[i]; 
   row = int(fes.upperDof[i]);
   LHM[row,:]   = 0.
   LHM[row,row] = 1.
   RHV[row]     = math.sin(a*xy[0])*math.sin(b*xy[1])



#=========================================================
# Left boundary conditions
#=========================================================
coord = np.asarray(fes.leftCoords)[:,0]
for i in range(fes.nLeft):
   xy  = fes.leftCoords[i]; 
   row = int(fes.leftDof[i]);
   LHM[row,:]   = 0.
   LHM[row,row] = 1.
   RHV[row]     = math.sin(a*xy[0])*math.sin(b*xy[1])


#=========================================================
# Right boundary conditions
#=========================================================
coord = np.asarray(fes.rightCoords)[:,0]
for i in range(fes.nRight):
   xy  = fes.rightCoords[i]; 
   row = int(fes.rightDof[i]);
   LHM[row,:]   = 0.
   LHM[row,row] = 1.
   RHV[row]     = math.sin(a*xy[0])*math.sin(b*xy[1])


			
#=========================================================
print ('Solving the system')
#=========================================================
#fes.printMatVec(LHM,RHV,"afterConstraints")
solVec = np.linalg.solve(LHM, RHV)


#=========================================================
# Compute the error by comparing the exact solution
# to the computed solution at the vertices
#=========================================================
sumsq = 0.;
uexact = np.zeros(fes.sysDim);
for i in range(mesh.nVert):
  xy        = mesh.getVertCoords(i);
  uexact[i] = math.sin(a*xy[0])*math.sin(b*xy[1]);
  sumsq    += (solVec[i]-uexact[i])*(solVec[i]-uexact[i])

print ('RMS Error =',math.sqrt(sumsq/mesh.nVert));

#=========================================================
# Plot the results
#=========================================================
print ('Plotting results')
fes.plotRawSoln(solVec, 'Solution')
fes.plotRawSoln(uexact-solVec,"Error")
