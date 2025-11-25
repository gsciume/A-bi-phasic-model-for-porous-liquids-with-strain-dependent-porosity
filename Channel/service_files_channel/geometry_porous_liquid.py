# Date: 09/09/2025
# Author: Thomas Lavigne
# Reference: Giuseppe Sciumè
# Laboratory: I2M
# 
# 
# --------------------------------------- #
# 	Create the geometries and boundaries  #
# --------------------------------------- #
def T_mesh(length_x, height_x,length_y,height_y, size_min=0.1, size_max=1, **kwargs):
	""" 
	2D T-Mesh Generation Utility

	Generates a 2D geometry resembling a 'T' shape by fusing two rectangles,
	creates a mesh based on user-defined size parameters, and assigns physical
	groups (tags) to the boundaries and the domain using the Gmsh API.

	Parameters
	----------
	length_x : float
	    The width of the main horizontal rectangle.
	height_x : float
	    The height of the main horizontal rectangle.
	length_y : float
	    The width of the vertical top part of the 'T'.
	height_y : float
	    The height of the vertical top part of the 'T'.
	size_min : float, optional
	    The minimum mesh element size (Mesh.MeshSizeMin in Gmsh). Defaults to 0.1.
	size_max : float, optional
	    The maximum mesh element size (Mesh.MeshSizeMax in Gmsh). Defaults to 1.0.
	**kwargs : dict, optional
	    Optional Gmsh configuration parameters (see Notes).

	Returns
	-------
	None

	Notes
	-----
	Optional Keyword Arguments:
	- `Terminal` (int): Sets General.Terminal (1 to display Gmsh output).
	- `Optimize` (int): Sets Mesh.Optimize (1 for standard mesh optimization).
	- `Netgen` (int): Sets Mesh.OptimizeNetgen (1 to use Netgen optimization).
	- `FileVersion` (float): Sets Mesh.MshFileVersion (e.g., 2.2 or 4.1).
	- `MeshSizeExtendFromBoundary` (int): Sets Mesh.MeshSizeExtendFromBoundary (1 to extend boundary-defined sizes).
	- `MeshSizeFromPoints` (int): Sets Mesh.MeshSizeFromPoints (1 to use sizes defined at points).
	- `MeshSizeFromCurvature` (int): Sets Mesh.MeshSizeFromCurvature (1 to refine mesh based on curvature).
	- `log` (bool): If True, enables Gmsh logging and saves a log and the 
	  `.geo_unrolled` file in the `geometry/` directory.

	Outputs
	-------
	- `geometry/2D_T_mesh.msh`: The generated computational mesh file.
	- `geometry/2D_T_mesh.geo_unrolled`: The Gmsh geometry script (if `log=True`).
	- `geometry/gmsh_output.log`: A log of internal Gmsh messages (if `log=True`).

	Physical Groups (Boundary Tags):
	- 'domain' (Tag 0, Dim 2): The entire T-shaped area (Volume/Domain).
	- 'Bottom' (Tag 1, Dim 1): The bottom boundary (y=0).
	- 'Right' (Tag 2, Dim 1): The right-side boundaries.
	- 'Top' (Tag 3, Dim 1): The top boundary of the vertical stem.
	- 'Left' (Tag 4, Dim 1): The left-side boundaries (x=0).
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
	gmsh.model.add("2D_T_mesh")
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
	#----------------------------------------------------------------------
	# 
	#----------------------------------------------------------------------
	# Set options
	#----------------------------------------------------------------------
	# 
	gmsh.option.setNumber("General.Terminal",1)
	gmsh.option.setNumber("Mesh.MeshSizeMin", size_min)
	gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
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
	print("Min mesh size allowed ", gmsh.option.getNumber("Mesh.MeshSizeMin"))
	print("Max mesh size allowed ", gmsh.option.getNumber("Mesh.MeshSizeMax"))
	# 
	#----------------------------------------------------------------------
	# Compute the geometry
	#----------------------------------------------------------------------
	# 
	# gmsh.model.occ.addRectangle(x, y, z, dx, dy, tag=-1, roundedRadius=0.)
	r1 = gmsh.model.occ.addRectangle(0, 0, 0, length_x, height_x, tag=-1)
	r2 = gmsh.model.occ.addRectangle(length_x/2-length_y/2, height_x, 0, length_y, height_y, tag=-1)
	domain, _ = gmsh.model.occ.fuse([(2, r1)], [(2, r2)])
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
		elif center_of_mass[1]>=height_x:
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
	gmsh.write("geometry/2D_T_mesh.msh")
	# 
	if log:
		gmsh.write('geometry/2D_T_mesh.geo_unrolled')
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