# Date: 09/09/2025
# Author: Thomas Lavigne
# Reference: Giuseppe Sciumè
# Laboratory: I2M
# 
# 
# --------------------------------------- #
# 	Create the geometries and boundaries  #
# --------------------------------------- #

def Axysimmetric_rectangle_mesh(length_x, radius_y, Nx, size_min=0.1, size_max=1, **kwargs):
	""" 
Generate a 2D axisymmetric rectangular mesh using Gmsh.

This function builds a rectangle of size ``length_x × radius_y`` for
axisymmetric simulations, applies mesh size settings, labels boundaries,
creates physical groups, and writes the resulting mesh in Gmsh format.
Optionally, it can also log Gmsh output to a file.

Parameters
----------
length_x : float
    Length of the domain in the x-direction.
radius_y : float
    Height (radius) of the axisymmetric domain in the y-direction.
Nx : int
    Number of divisions along the x-axis, used to compute the characteristic
    mesh length ``lc = length_x / Nx``.
size_min : float, optional
    Minimum mesh size factor applied to ``lc`` (default is 0.1).
size_max : float, optional
    Maximum mesh size factor applied to ``lc`` (default is 1.0).
**kwargs :
    Optional Gmsh configuration flags. Supported keys include:
    
    - ``log`` : bool  
      If True, writes Gmsh logs to ``geometry/gmsh_output.log``.
    - ``Terminal`` : int  
      Controls Gmsh terminal display (General.Terminal).
    - ``Optimize`` : int  
      Enables or disables mesh optimization.
    - ``Netgen`` : int  
      Activates Netgen optimization.
    - ``FileVersion`` : float  
      Sets the mesh file version.
    - ``MeshSizeExtendFromBoundary`` : int  
      Controls mesh size propagation from boundaries.
    - ``MeshSizeFromPoints`` : int  
      Enables mesh size interpolation from points.
    - ``MeshSizeFromCurvature`` : int  
      Enables curvature-based mesh refinement.

Physical Groups Created
-----------------------
The function automatically identifies edges by their center of mass and
creates the following physical groups:

- ``Left``   (tag = 4) : x = 0
- ``Top``    (tag = 3) : y = radius_y
- ``Right``  (tag = 2) : x = length_x
- ``Bottom`` (tag = 1) : y = 0
- ``domain`` (tag = 0) : the 2D surface

Output Files
------------
The following files are written into the ``geometry/`` directory:

- ``2D_axysim_rectangle.msh`` : The mesh with physical tags.
- ``2D_axysim_rectangle.geo_unrolled`` : (if log=True) flattened geometry file.
- ``gmsh_output.log`` : (if log=True) Gmsh log messages.

Returns
-------
None
    The function writes geometry/mesh files to disk and finalizes Gmsh.

Notes
-----
Requires Gmsh with Python API installed:
``pip install gmsh``
	"""
	#----------------------------------------------------------------------
	# Libraries
	#----------------------------------------------------------------------
	# 
	import gmsh
	import numpy
	import sys
	import os
	# 
	# Create the model
	gmsh.initialize()
	gmsh.clear()
	gmsh.model.add("2D_Axysim_rectangle")
	gmsh.model.occ.synchronize()
	# 
	# Initialize log
	log = kwargs.get('log',None)
	# 
	if log:
		log_dir = "geometry"
		log_path = os.path.join(log_dir, "gmsh_output.log")
		# 
		# Crée le dossier s'il n'existe pas
		os.makedirs(log_dir, exist_ok=True)
		# 
		# Ouvre le fichier en mode écriture ("w" écrase si déjà existant)
		log_file = open(log_path, "w")
		# 
		gmsh.logger.start()
	# 
	#----------------------------------------------------------------------
	# Geometrical parameters
	#----------------------------------------------------------------------
	# Dimension of the problem
	gdim = 2
	# Characteristic length
	lc   = length_x / Nx
	#----------------------------------------------------------------------
	# 
	#----------------------------------------------------------------------
	# Set options
	#----------------------------------------------------------------------
	# 
	gmsh.option.setNumber("General.Terminal",1)
	gmsh.option.setNumber("Mesh.MeshSizeMin", size_min*lc)
	gmsh.option.setNumber("Mesh.MeshSizeMax", size_max*lc)
	# 
	terminal					 = kwargs.get('Terminal', None)
	optimise					 = kwargs.get('Optimize', None)
	netgen						 = kwargs.get('Netgen', None)
	fileversion					 = kwargs.get('FileVersion', None)
	meshsizeextendfromboundary   = kwargs.get('MeshSizeExtendFromBoundary', None)
	meshsizefrompoints			 = kwargs.get('MeshSizeFromPoints', None)
	meshsizefromcurvature		 = kwargs.get('MeshSizeFromCurvature', None)
	# 
	if terminal:
		gmsh.option.setNumber("General.Terminal",terminal)
	if optimise:
		gmsh.option.setNumber("Mesh.Optimize", optimise)
	if netgen:
		gmsh.option.setNumber("Mesh.OptimizeNetgen", netgen)
	if fileversion:
		gmsh.option.setNumber("Mesh.MshFileVersion", fileversion)
	if meshsizeextendfromboundary:
		gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", meshsizeextendfromboundary)
	if meshsizefrompoints:
		gmsh.option.setNumber("Mesh.MeshSizeFromPoints", meshsizefrompoints)
	if meshsizefromcurvature:
		gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", meshsizefromcurvature)
	# synchronise the model
	gmsh.model.occ.synchronize()
	# 
	#----------------------------------------------------------------------
	# Compute the geometry
	#----------------------------------------------------------------------
	# 
	# gmsh.model.occ.addRectangle(x, y, z, dx, dy, tag=-1, roundedRadius=0.)
	s1 = gmsh.model.occ.addRectangle(0, 0, 0, length_x, radius_y, tag=-1)
	# 
	gmsh.model.occ.synchronize()
	#----------------------------------------------------------------------
	# Create physical group for mesh generation and tagging
	#----------------------------------------------------------------------
	# 
	# Get the entities of the model for stable tagging
	lines, surfaces, volumes = [gmsh.model.getEntities(d) for d in [1, 2, 3]]
	# 
	left, top, right, bottom = [], [], [], []
	tag_left, tag_top, tag_right, tag_bottom = 4, 3, 2, 1
	# 
	domain = []
	tag_domain = 0
	# 
	for line in lines:
		center_of_mass = gmsh.model.occ.getCenterOfMass(line[0], line[1])
		if numpy.isclose(center_of_mass[0],0):
			left.append(line[1])
		elif numpy.isclose(center_of_mass[1],radius_y):
			top.append(line[1])
		elif numpy.isclose(center_of_mass[0],length_x):
			right.append(line[1])
		elif numpy.isclose(center_of_mass[1],0):
			bottom.append(line[1])
	# 
	gmsh.model.addPhysicalGroup(gdim-1, left, tag_left)
	gmsh.model.setPhysicalName(gdim-1, tag_left, 'Left')
	# 
	gmsh.model.addPhysicalGroup(gdim-1, top, tag_top)
	gmsh.model.setPhysicalName(gdim-1, tag_top, 'Top')
	# 
	gmsh.model.addPhysicalGroup(gdim-1, right, tag_right)
	gmsh.model.setPhysicalName(gdim-1, tag_right, 'Right')
	# 
	gmsh.model.addPhysicalGroup(gdim-1, bottom, tag_bottom)
	gmsh.model.setPhysicalName(gdim-1, tag_bottom, 'Bottom')
	# 
	for surface in surfaces:
		domain.append(surface[1])
	# 
	gmsh.model.addPhysicalGroup(gdim, domain, tag_domain)
	gmsh.model.setPhysicalName(gdim, tag_domain, 'domain')
	# 
	#----------------------------------------------------------------------
	# Export the geometry with the tags for control
	#----------------------------------------------------------------------
	# 
	gmsh.model.occ.synchronize()
	# 
	gmsh.model.mesh.generate(gdim)
	gmsh.write("geometry/2D_axysim_rectangle.msh")
	# 
	if log:
		gmsh.write('geometry/2D_axysim_rectangle.geo_unrolled')
		# Récupérer les messages internes de Gmsh
		logs = gmsh.logger.get()
		for line in logs:
			log_file.write(line + "\n")
		log_file.close()
		# Stop logger (nettoyage)
		gmsh.logger.stop()
	else:
		pass
	# 
	gmsh.finalize()
	return None
# 
#____________________________________________________________#
# 					End of functions
#____________________________________________________________#
# 
# 
if __name__ == "__main__":
    print("Loading of the user-defined geometry functions successfully completed.")
    # EoF