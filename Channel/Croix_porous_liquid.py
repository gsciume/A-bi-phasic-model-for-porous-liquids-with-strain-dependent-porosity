# Date: 09/09/2025
# Author: Thomas Lavigne
# Reference: Giuseppe Sciumè
# Laboratory: I2M
# 
# 
# --------------------------------------- #
# 			Import libraries			  #
# --------------------------------------- #
import sys
import os
import shutil
import dolfinx
import mpi4py
import ufl
import basix
import numpy
from dolfinx.fem.petsc import NonlinearProblem
import time
# 
if mpi4py.MPI.COMM_WORLD == 0:
	print("Dolfinx version is:",dolfinx.__version__)
# 
# Compute the overall computation time
# Set time counter
begin_t = time.time()
# 
# --------------------------------------- #
# 	   Import user-defined functions	  #
# --------------------------------------- #
# Add path to the service files
sys.path.append('./service_files_channel')
# 
import geometry_porous_liquid
import constitutive_laws
import functions
# 
# --------------------------------------- #
# 	   			Clean workspace	 		  #
# --------------------------------------- #
# path to the ouptut
folder = "./outputs_croix/"
# 
# remove old output if exists
if mpi4py.MPI.COMM_WORLD.rank == 0:
	if os.path.exists(folder) and os.path.isdir(folder):
	    shutil.rmtree(folder)
	    print(f"Removed folder: {folder}")
	else:
	    print(f"Folder does not exist: {folder}")
# 
# 
# --------------------------------------- #
# 	   			 Log level				  #
# --------------------------------------- #
# 
log      = False
# 
log_mesh = False
# 
create_new_mesh = True
# 
def evaluate_point(mesh, function, contributing_cells, point, output_list, index):
	"""
	Suitable Evaluations functions for Parallel computation
	Inputs: mesh, function to evaluate, contributing cells to the point, point, output list to store the value, index in the list
	outputs_fluid_terzaghi: the evaluated function value is added at the index location in output list
	"""
	function_eval = None
	if len(contributing_cells) > 0:
		function_eval = function.eval(point, contributing_cells[:1])
	function_eval = mesh.comm.gather(function_eval, root=0)
	# Choose first pressure that is found from the different processors
	if mpi4py.MPI.COMM_WORLD.rank == 0:
		for element in function_eval:
			if element is not None:
				output_list[index]=element[0]
				break
	return None
# 
# --------------------------------------- #
# 	   		 Time parameters			  #
# --------------------------------------- #
# 
# Final time [s]
t_fin = 5
# Time step [s]
dt    = 1e-2
# 
num_steps = int(t_fin/dt)
# 
# time steps to evaluate the pressure in space:
n0, n1, n2,n3,n4 = 10,70,115,300,450
# --------------------------------------- #
# 	   	Geometrical parameters			  #
# --------------------------------------- #
# 
L1,H1,l2,H2 = 1e-1, 2e-2,1e-2,5e-2
# 
# --------------------------------------- #
# 	   		Generate Mesh				  #
# --------------------------------------- #
# Create gmsh tagged mesh
gmsh_name = "geometry/2D_T_mesh.msh"
# 
if (os.path.exists(gmsh_name)) and (create_new_mesh == False):
	pass
else:
	if mpi4py.MPI.COMM_WORLD.rank == 0:
		geometry_porous_liquid.T_mesh(L1,H1,l2,H2, size_min=1e-4, size_max=1e-3, log=True, optimise=True, netgen=True)
# 
# Read the mesh and tags
gmsh_model_rank = 0
gdim            = 2
# 
domain, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(gmsh_name, mpi4py.MPI.COMM_WORLD, gmsh_model_rank, gdim=gdim)
# 
# # 
if log_mesh:
	xdmftag = dolfinx.io.XDMFFile(mpi4py.MPI.COMM_WORLD, 'outputs_croix/debug_tags_mesh.xdmf', "w")
	xdmftag.write_mesh(domain)
	xdmftag.write_meshtags(facet_tags, domain.geometry)
	xdmftag.write_meshtags(cell_tags,  domain.geometry)
	xdmftag.close()
# 
mesh_stats = functions.get_element_size_stats(domain)
volume  = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1.0 * ufl.Measure('dx', domain=domain)))
volume_ = domain.comm.allreduce(volume, op=mpi4py.MPI.SUM)
if mpi4py.MPI.COMM_WORLD.rank == 0:
	print("--- Loaded Mesh Stats ---")
	print(f"Total Volume: {volume_}")
	print(f"Mean Element Size (h_bar): {mesh_stats['mean_h']:.6f}")
	print(f"Standard Deviation (sigma_h): {mesh_stats['std_dev_h']:.6f}")
# 
# 
# --------------------------------------- #
# 	   		Evaluation points			  #
# --------------------------------------- #
# 
# Identify contributing cells to our points of interest for post processing
num_points = 15
# Physical points we want an evaluation in
x_check          = numpy.linspace(0,L1,num_points)
y_check 			 = numpy.linspace(0,H1,num_points)
points_for_space_y_L = numpy.zeros((num_points,3))
points_for_space_y_R = numpy.zeros((num_points,3))
points_for_space_x   = numpy.zeros((num_points,3))
for ii in range(num_points):
	points_for_space_x[ii,1] = 0.
	points_for_space_x[ii,0] = x_check[ii]
	points_for_space_y_L[ii,1] = y_check[ii]
	points_for_space_y_L[ii,0] = L1/10
	points_for_space_y_R[ii,1] = y_check[ii]
	points_for_space_y_R[ii,0] = 9*L1/10
# Create the bounding box tree and identify contributing cells
tree             = dolfinx.geometry.bb_tree(domain, domain.geometry.dim)
points           = numpy.concatenate((points_for_space_y_L,points_for_space_y_R,points_for_space_x))
cell_candidates  = dolfinx.geometry.compute_collisions_points(tree, points)
colliding_cells  = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points)
# 
# --------------------------------------- #
# 	   		Material parameters			  #
# --------------------------------------- #
# 
# Viscosities [Pa.s]
scafflod_like_viscosity      = dolfinx.default_scalar_type(50) 
interstitial_fluid_viscosity = dolfinx.default_scalar_type(1e-3)
# 
# 
# Thresholds for the constitutive law 
# cell_pressure(cell fraction)
# adapted from Byrne and Preziosi (2003)
equilibrium_fraction = dolfinx.default_scalar_type(0.65)
decohesion_fraction  = dolfinx.default_scalar_type(0.50)
interaction_fraction = dolfinx.default_scalar_type(0.00)
# 
# Decohesion pressure
decohesion_pressure                = dolfinx.default_scalar_type(-500)
# 
# Permeability of the interstitium [m²] (plays significant role on v identified)
permea_eps_square = True
if permea_eps_square:
	permeability = dolfinx.default_scalar_type((1e-9)/((1-equilibrium_fraction)*(1-equilibrium_fraction)))
else:
	permeability = dolfinx.default_scalar_type(1e-9)
# 
# --------------------------------------- #
# 	   		 Boundary values			  #
# --------------------------------------- #
# 
# Pressures [Pa]
p_right = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0.))
p_left  = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(1000.))
# v_left  = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(1e-6))
# 
# --------------------------------------- #
# 	   			Operators				  #
# --------------------------------------- #
# Coordinates, for axisymmetry
# Normals
normal = ufl.FacetNormal(domain)
# Specify the desired quadrature degree
q_deg  = 4
# attribute the cell_tags to the integral element
dx     = ufl.Measure('dx', metadata={"quadrature_degree":q_deg}, subdomain_data=cell_tags, domain=domain)
# attribute the facet_tags to the surfacic integral element
ds     = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
# 
# --------------------------------------- #
# 	  	 		FunctionSpace			  #
# --------------------------------------- #
# Mixed Space (R,R,R2) -> (pl,pb,u)
P2_v      = basix.ufl.element("P", domain.topology.cell_name(), degree=2, shape=(domain.topology.dim,))
P1_v      = basix.ufl.element("P", domain.topology.cell_name(), degree=1, shape=(domain.topology.dim,))
P1        = basix.ufl.element("P", domain.topology.cell_name(), degree=1)
# 
# Function spaces
P2v_space     = dolfinx.fem.functionspace(domain, P2_v)
P1v_space     = dolfinx.fem.functionspace(domain, P1_v)
P1_space      = dolfinx.fem.functionspace(domain, P1)
# Mixed-space
MS            = dolfinx.fem.functionspace(mesh=domain, element=basix.ufl.mixed_element([P1,P1,P2_v]))
#
# --------------------------------------- #
# 	   			Functions				  #
# --------------------------------------- #
# Define the solution at the current and previous time steps
solution          = dolfinx.fem.Function(MS)
previous_solution = dolfinx.fem.Function(MS)
# 
# Export P2 -> P1
velocity          = dolfinx.fem.Function(P1v_space)
# 
# Post processing
porosity         = dolfinx.fem.Function(P1_space)
cell_pressure    = dolfinx.fem.Function(P1_space)
# 
# Mapping in the Mixed Space: FunctionSpace, Mapping_in_the_mixed_space = MS.sub(xx).collapse()
Pressure_, Pressure_to_MS   = MS.sub(0).collapse()
Fraction_, Fraction_to_MS   = MS.sub(1).collapse()
Velocity_, Velocity_to_MS   = MS.sub(2).collapse()
# 
# Create the test functions
q_p, q_e, q_v = ufl.TestFunctions(MS)
# 
# --------------------------------------- #
# 	   		   Expressions			      #
# --------------------------------------- #
porosity_expr      = dolfinx.fem.Expression(1-previous_solution.sub(1) , P1_space.element.interpolation_points())
cell_pressure_expr = dolfinx.fem.Expression(constitutive_laws.Byrne_law(previous_solution.sub(1),equilibrium_fraction,interaction_fraction,decohesion_fraction,decohesion_pressure,log=True) , P1_space.element.interpolation_points())
# --------------------------------------- #
# 	   		Initial conditions			  #
# --------------------------------------- #
# 
solution.x.array[Fraction_to_MS]=dolfinx.default_scalar_type(equilibrium_fraction)
solution.x.scatter_forward()
previous_solution.x.array[Fraction_to_MS]=dolfinx.default_scalar_type(equilibrium_fraction)
previous_solution.x.scatter_forward()
# 
# --------------------------------------- #
# 	   		Boundary conditions			  #
# --------------------------------------- #
bcs    = []
fdim   = domain.topology.dim - 1
# 
# bottom, right, top, left = 1, 2, 3, 4
# 
# bottom
facets = facet_tags.find(1)
# (vy = 0, tx = 0, grad(pl).n = 0)
# & no slip
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(0), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(0)))
# 
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1)))
# right
facets = facet_tags.find(2)
# (vx = 0, vy = 0, pl = 0)
# 
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1)))
# 
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(0), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(p_right, dofs, MS.sub(0)))
# 
# top
facets = facet_tags.find(3)
# No-slip condition
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(0), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(0)))
# 
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1))) 
# 
# left vy=0
facets = facet_tags.find(4)
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1)))
# 
# --------------------------------------- #
# 	   			Weak Form				  #
# --------------------------------------- #
# 
fluid_pressure,  cell_fraction,  cell_velocity  = ufl.split(solution)
fluid_pressure_n,  cell_fraction_n,  cell_velocity_n = ufl.split(previous_solution)
#  
# Weak form
# mass epsilon c
F = (
	(1/dt)*(cell_fraction-cell_fraction_n)*q_e*dx 
	+ ufl.nabla_div(cell_fraction*cell_velocity)*q_e*dx
	)
# mass epsilon m
F+= (
	(1/dt)*(cell_fraction_n-cell_fraction)*q_p*dx 
	+ ufl.nabla_div((1-cell_fraction)*cell_velocity)*q_p*dx 
	)
if permea_eps_square == False:
	F+= ((permeability)/interstitial_fluid_viscosity)*ufl.dot(ufl.grad(fluid_pressure), ufl.grad(q_p))*dx
else:
	F+= ((permeability*(1-cell_fraction)**2)/interstitial_fluid_viscosity)*ufl.dot(ufl.grad(fluid_pressure), ufl.grad(q_p))*dx
	
# test velocity
F+=(
	- fluid_pressure*ufl.nabla_div(q_v)*dx
	-  (constitutive_laws.Byrne_law(cell_fraction,equilibrium_fraction,interaction_fraction,decohesion_fraction,decohesion_pressure,log=True))*ufl.nabla_div(q_v)*dx 
	+ scafflod_like_viscosity*ufl.inner(ufl.grad(cell_velocity), ufl.grad(q_v))*dx 
	+ scafflod_like_viscosity*ufl.inner(ufl.grad(cell_velocity).T, ufl.grad(q_v))*dx 
	+ p_left*ufl.dot(normal, q_v)*ds(4) 
	)
# 
# --------------------------------------- #
# 	   				Solver			 	  #
# --------------------------------------- #
# Non linear problem definition
dsolution = ufl.TrialFunction(MS)
J         = ufl.derivative(F, solution, dsolution)
Problem   = NonlinearProblem(F, solution, bcs = bcs, J = J)
solver    = functions.set_non_linear_solver_parameters(domain, Problem, 1e-10, 1e-12, "incremental", 10, log_newton=log)
# 
# --------------------------------------- #
# 	   	  Solving & Post-treatment		  #
# --------------------------------------- #
# 
pressuref_space0       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressuref_space1       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressuref_space2       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressuref_space3       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressuref_space4       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressures_space0       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressures_space1       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressures_space2       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressures_space3       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressures_space4       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
L_velocity_space0       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
L_velocity_space1       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
L_velocity_space2       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
L_velocity_space3       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
L_velocity_space4       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
R_velocity_space0       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
R_velocity_space1       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
R_velocity_space2       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
R_velocity_space3       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
R_velocity_space4       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
# 
# 
# Open file for export
xdmf = dolfinx.io.XDMFFile(domain.comm, "outputs_croix/Output.xdmf", "w")
xdmf.write_mesh(domain)
# vtk outputs
pressure_vtk       = dolfinx.io.VTKFile(domain.comm, "outputs_croix/VTK/pressure.pvd", "w")
pressure_vtk.write_mesh(domain)
fraction_vtk       = dolfinx.io.VTKFile(domain.comm, "outputs_croix/VTK/fraction.pvd", "w")
fraction_vtk.write_mesh(domain)
velocity_vtk       = dolfinx.io.VTKFile(domain.comm, "outputs_croix/VTK/velocity.pvd", "w")
velocity_vtk.write_mesh(domain)
porosity_vtk       = dolfinx.io.VTKFile(domain.comm, "outputs_croix/VTK/porosity.pvd", "w")
porosity_vtk.write_mesh(domain)
cell_pressure_vtk  = dolfinx.io.VTKFile(domain.comm, "outputs_croix/VTK/cell_pressure.pvd", "w")
cell_pressure_vtk.write_mesh(domain)
# initialise time
t = 0
__pressure, __fraction, __v  = previous_solution.split()
__pressure.name              = "Fluid Pressure"
__fraction.name              = "Cell Fraction"
__pressure2 = __pressure.collapse()
__fraction2 = __fraction.collapse()
__pressure2.name              = "Fluid Pressure"
__fraction2.name              = "Cell Fraction"
velocity.interpolate(previous_solution.sub(2))
velocity.x.scatter_forward()
velocity.name                = "Cell Velocity"
porosity.interpolate(porosity_expr)
porosity.x.scatter_forward()
porosity.name                = "Porosity"
cell_pressure.interpolate(cell_pressure_expr)
cell_pressure.x.scatter_forward()
cell_pressure.name                = "cell pressure"
# Export in the xdmf file 
xdmf.write_function(__pressure,t)
xdmf.write_function(__fraction,t)
xdmf.write_function(velocity,t)
xdmf.write_function(porosity,t)
xdmf.write_function(cell_pressure,t)
pressure_vtk.write_function(__pressure2,t)
fraction_vtk.write_function(__fraction2,t)
velocity_vtk.write_function(velocity,t)
porosity_vtk.write_function(porosity,t)
cell_pressure_vtk.write_function(cell_pressure,t)
# 
for iteration in range(num_steps):
	t+=dt 
	if mpi4py.MPI.COMM_WORLD.rank == 0:
		print('Step: ',iteration+1,'/',num_steps, 'time:', t, ' s')
	try:
		num_its, converged = solver.solve(solution)
	except:
		if mpi4py.MPI.COMM_WORLD.rank == 0:
			print("*************") 
			print("Solver failed")
			print("*************") 
		break
	# Ensure pushing the solution to all processes
	solution.x.scatter_forward()
	# Update previous solution
	previous_solution.x.array[:] = solution.x.array[:]
	previous_solution.x.scatter_forward()
	# 
	__pressure, __fraction, __v  = previous_solution.split()
	__pressure.name              = "Fluid Pressure"
	__fraction.name              = "Cell Fraction"
	__pressure2 = __pressure.collapse()
	__fraction2 = __fraction.collapse()
	__pressure2.name              = "Fluid Pressure"
	__fraction2.name              = "Cell Fraction"
	# 
	velocity.interpolate(__v)
	velocity.x.scatter_forward()
	velocity.name                = "Cell Velocity"
	# 
	porosity.interpolate(porosity_expr)
	porosity.x.scatter_forward()
	porosity.name                = "Porosity"
	cell_pressure.interpolate(cell_pressure_expr)
	cell_pressure.x.scatter_forward()
	cell_pressure.name                = "cell pressure"
	if iteration%1 == 0:
		# Export in the xdmf file 
		xdmf.write_function(__pressure,t)
		xdmf.write_function(__fraction,t)
		xdmf.write_function(velocity,t)
		xdmf.write_function(porosity,t)
		xdmf.write_function(cell_pressure,t)
		pressure_vtk.write_function(__pressure2,t)
		fraction_vtk.write_function(__fraction2,t)
		velocity_vtk.write_function(velocity,t)
		porosity_vtk.write_function(porosity,t)
		cell_pressure_vtk.write_function(cell_pressure,t)
		# in space
	if iteration == n0:
		for ii in range(num_points):
			evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressuref_space0, ii)
			evaluate_point(domain, cell_pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressures_space0, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(num_points+ii), points[num_points+ii], R_velocity_space0, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(ii), points[ii], L_velocity_space0, ii)
	elif iteration == n1:
		for ii in range(num_points):
			evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressuref_space1, ii)
			evaluate_point(domain, cell_pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressures_space1, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(num_points+ii), points[num_points+ii], R_velocity_space1, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(ii), points[ii], L_velocity_space1, ii)
	elif iteration == n2:
		for ii in range(num_points):
			evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressuref_space2, ii)
			evaluate_point(domain, cell_pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressures_space2, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(num_points+ii), points[num_points+ii], R_velocity_space2, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(ii), points[ii], L_velocity_space2, ii)
	elif iteration == n3:
		for ii in range(num_points):
			evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressuref_space3, ii)
			evaluate_point(domain, cell_pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressures_space3, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(num_points+ii), points[num_points+ii], R_velocity_space3, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(ii), points[ii], L_velocity_space3, ii)
	elif iteration == n4:
		for ii in range(num_points):
			evaluate_point(domain, __pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressuref_space4, ii)
			evaluate_point(domain, cell_pressure, colliding_cells.links(2*num_points+ii), points[2*num_points+ii], pressures_space4, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(num_points+ii), points[num_points+ii], R_velocity_space4, ii)
			evaluate_point(domain, __v.sub(0), colliding_cells.links(ii), points[ii], L_velocity_space4, ii)
xdmf.close()
pressure_vtk.close()
fraction_vtk.close()
velocity_vtk.close()
porosity_vtk.close()
cell_pressure_vtk.close()
# 
# Evaluate final time
end_t = time.time()
t_hours = int((end_t-begin_t)//3600)
tmin = int(((end_t-begin_t)%3600)//60)
tsec = int(((end_t-begin_t)%3600)%60)
if mpi4py.MPI.COMM_WORLD.rank == 0:
	print(f"FEM operated with {num_steps} iterations, in {t_hours} h {tmin} min {tsec} sec")
# 
# EoF
if mpi4py.MPI.COMM_WORLD.rank == 0:
	functions.export_to_csv(
		[x_check,pressuref_space0,pressures_space0,pressuref_space1,pressures_space1,pressuref_space2,pressures_space2,pressuref_space3,pressures_space3,pressuref_space4,pressures_space4],
		"pressure_space.csv",
		["x","pressuref_space0","pressures_space0","pressuref_space1","pressures_space1","pressuref_space2","pressures_space2","pressuref_space3","pressures_space3","pressuref_space4","pressures_space4"]
		)
	functions.export_to_csv(
		[y_check,L_velocity_space0,R_velocity_space0,L_velocity_space1,R_velocity_space1,L_velocity_space2,R_velocity_space2,L_velocity_space3,R_velocity_space3,L_velocity_space4,R_velocity_space4],
		"velocity_space.csv",
		["y","L_velocity_space0","R_velocity_space0","L_velocity_space1","R_velocity_space1","L_velocity_space2","R_velocity_space2","L_velocity_space3","R_velocity_space3","L_velocity_space4","R_velocity_space4"]
		)
	def transpose_csv(input_file,output_file):
		import csv
		with open(input_file, newline='') as f:
		    reader = csv.reader(f)
		    rows = list(reader)

		# Séparer le header et les données
		header = rows[0]       # première ligne
		data = rows[1:]        # toutes les autres lignes

		# Transposer seulement les données
		transposed_data = list(map(list, zip(*data)))

		# Réécrire le CSV avec le header inchangé
		with open(output_file, 'w', newline='') as f:
		    writer = csv.writer(f)
		    writer.writerow(header)           # écrire le header original
		    writer.writerows(transposed_data) # écrire les données transposées
		print(f"Transposition terminée pour latex: {output_file}")

	input_file = "pressure_space.csv"
	output_file = "pressure_space_t.csv"
	transpose_csv(input_file,output_file)

	input_file = "velocity_space.csv"
	output_file = "velocity_space_t.csv"
	transpose_csv(input_file,output_file)