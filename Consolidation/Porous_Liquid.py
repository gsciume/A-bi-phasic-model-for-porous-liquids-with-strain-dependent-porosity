# Date: 20/11/2025
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
import dolfinx
import mpi4py
import ufl
import basix
import numpy
import time
import pandas
from dolfinx.fem.petsc import NonlinearProblem
# 
if mpi4py.MPI.COMM_WORLD.rank == 0:
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
sys.path.append('./service_files_consolidation')
# 
import geometry_porous_liquid
import constitutive_laws
import functions
# 
# --------------------------------------- #
# 	    More user-defined functions	      #
# --------------------------------------- #
# 
def terzaghi_p(x):
	"""
	Compute the Exact Terzaghi solution for L2 error estimation
	Inputs: coordinates of the mesh
	outputs_fluid_terzaghi: Fluid pressure 
	"""
	# applied pressure and characteristic size of the domain
	p0,L = p_left_terzaghi, length_cylinder
	# initialize pression
	pression = 0
	# number of iterations
	kmax=1e3
	# Consider coordinate (L-x) to respect Terzaghi standards and compute the series
	for k in range(1,int(kmax)):
		pression += 4/numpy.pi*(-1)**(k-1)/(2*k-1)*numpy.cos((2*k-1)*0.5*numpy.pi*((L-x[0])/L))*numpy.exp(-(2*k-1)**2*0.25*numpy.pi**2*cv*t/L**2)
		pl = p0*pression
	return pl
# 
def terzaghi_v(x):
	"""
	Compute the Exact Terzaghi solution for L2 error estimation
	Inputs: coordinates of the mesh
	Outputs: Fluid pressure 
	"""
	# applied pressure and characteristic size of the domain
	p0,L = p_left_terzaghi, length_cylinder
	# initialize displacement
	speed = 0
	# number of iterations
	kmax=1e3
	# Consider coordinate (L-x) to respect Terzaghi standards and compute the series
	for k in range(1,int(kmax)):
		speed += 2/(L*M)*(-1)**(k+1)*cv*numpy.sin((2*k-1)*0.5*numpy.pi*((L-x[0])/L))*numpy.exp(-(2*k-1)**2*0.25*numpy.pi**2*cv*t/L**2)
		speeds = p0*speed
	return speeds
# 
def L2_error_p(mesh, P1, __p):
    P1space = dolfinx.fem.functionspace(mesh, P1)
    p_theo  = dolfinx.fem.Function(P1space)
    p_theo.interpolate(terzaghi_p)
    L2_errorp = dolfinx.fem.form(ufl.inner(__p - p_theo, __p - p_theo) * dx)
    L2_vol  = dolfinx.fem.form(1 * dx)

    num_local = dolfinx.fem.assemble_scalar(L2_errorp)
    den_local = dolfinx.fem.assemble_scalar(L2_vol)

    num = mesh.comm.allreduce(num_local, op=mpi4py.MPI.SUM)
    den = mesh.comm.allreduce(den_local, op=mpi4py.MPI.SUM)
    return numpy.sqrt(num / den)
# 
def L2_error_v(mesh, P1, __v):
    P1space = dolfinx.fem.functionspace(mesh, P1)
    v_theo  = dolfinx.fem.Function(P1space)
    v_theo.interpolate(terzaghi_v)
    L2_errorv = dolfinx.fem.form(ufl.inner(__v.sub(0) - v_theo, __v.sub(0) - v_theo) * dx)
    L2_vol  = dolfinx.fem.form(1 * dx)

    num_local = dolfinx.fem.assemble_scalar(L2_errorv)
    den_local = dolfinx.fem.assemble_scalar(L2_vol)

    num = mesh.comm.allreduce(num_local, op=mpi4py.MPI.SUM)
    den = mesh.comm.allreduce(den_local, op=mpi4py.MPI.SUM)
    return numpy.sqrt(num / den)
# 
def evaluate_point(mesh, function, contributing_cells, point, output_list, index):
    """
    Parallel-safe version of point evaluation, compatible with the original interface.
    
    Inputs:
        mesh              : dolfinx mesh
        function          : dolfinx.fem.Function to evaluate
        contributing_cells: list/array of local cell indices that might contain the point
        point             : coordinates of the point (1D array of length mesh.geometry.dim)
        output_list       : pre-allocated Python list or array to store the value
        index             : index in output_list to store the result
    Actions:
        Evaluates `function` at `point` if the current processor owns the cell,
        gathers the result to rank 0, and sets `output_list[index]`.
    """
    val_local = None

    # Only evaluate if the point is contained in a local cell
    if len(contributing_cells) > 0:
        try:
            val_local = function.eval(point, [int(contributing_cells[0])])
            val_local = float(val_local[0])  # Convert to scalar if function is scalar
        except Exception:
            val_local = None

    # Gather values from all ranks
    gathered_vals = mesh.comm.gather(val_local, root=0)

    # Rank 0 chooses the first non-None value and stores it in output_list
    if mesh.comm.rank == 0:
        for v in gathered_vals:
            if v is not None:
                output_list[index] = v
                break

    return None
# 
# --------------------------------------- #
# 	   		 Time parameters			  #
# --------------------------------------- #
# 
# Final time [s]
t_fin = 25.
# Time step [s]
dt    = 0.005
# dt    = 0.5
# Number of required steps [-]
num_steps = int(t_fin/dt)
# 
# time steps to evaluate the pressure in space:
n0, n1, n2 = 700,2000,4000
# n0, n1, n2 = 7,20,40
#  
# --------------------------------------- #
# 	   	Geometrical parameters			  #
# --------------------------------------- #
# 
# Radius of the cylinder [m]
radius_cylinder = 0.01
# length of the cylinder [m]
length_cylinder = 0.1
# 
# --------------------------------------- #
# 	   		Generate Mesh				  #
# --------------------------------------- #
# Spacial discretising [-]
Nx = 40
# Create gmsh tagged mesh
gmsh_name = "geometry/2D_axysim_rectangle.msh"
create_new_mesh = False
# Need to regenerate the mesh or not
if (os.path.exists(gmsh_name)) and (create_new_mesh == False):
	pass
else:
	if mpi4py.MPI.COMM_WORLD.rank == 0:
		geometry_porous_liquid.Axysimmetric_rectangle_mesh(length_cylinder, radius_cylinder, Nx, size_min=0.1, size_max=1, log=True, optimise=True, netgen=True)
# 
# Read the mesh and the mesh tags
gmsh_model_rank = 0
gdim            = 2
# 
domain, cell_tags, facet_tags = dolfinx.io.gmshio.read_from_msh(gmsh_name, mpi4py.MPI.COMM_WORLD, gmsh_model_rank, gdim=gdim)
# 
# To ensure proper reading, one can refer to this xdmf including the boundary tags not working in // need to explicit xdmf_tag = ...
xdmftag = dolfinx.io.XDMFFile(mpi4py.MPI.COMM_WORLD, 'outputs_fluid_terzaghi/debug_tags_mesh.xdmf', "w")
xdmftag.write_mesh(domain)
xdmftag.write_meshtags(facet_tags, domain.geometry)
xdmftag.write_meshtags(cell_tags,  domain.geometry)
xdmftag.close()
# 
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
x_check          = numpy.linspace(0,length_cylinder,num_points)
points_for_time  = numpy.array([[length_cylinder, 0., 0.], [length_cylinder/2, 0., 0.]])
points_for_space = numpy.zeros((num_points,3))
for ii in range(num_points):
	points_for_space[ii,1] = 0.
	points_for_space[ii,0] = x_check[ii]
# Create the bounding box tree and identify contributing cells
tree             = dolfinx.geometry.bb_tree(domain, domain.geometry.dim)
points           = numpy.concatenate((points_for_time,points_for_space))
cell_candidates  = dolfinx.geometry.compute_collisions_points(tree, points)
colliding_cells  = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points)
cells_x_L        = colliding_cells.links(0)
cells_x_L_over_2 = colliding_cells.links(1)
cells_x_0        = colliding_cells.links(2)
# 
# --------------------------------------- #
# 	   		Material parameters			  #
# --------------------------------------- #
# 
# Viscosities [Pa.s]
scafflod_like_viscosity      = dolfinx.default_scalar_type(50) 
interstitial_fluid_viscosity = dolfinx.default_scalar_type(1e-3)
# 
# Permeability of the interstitium [m²]
permeability      = dolfinx.default_scalar_type(1e-9*(1-0.65)**2) # 0.65 to equal t if permea_eps_square == False, set k_tilde = kf*(1-eps_eq^s)^2
permea_eps_square = False
# 
# Thresholds for the constitutive law 
# cell_pressure(cell fraction)
# adapted from Byrne and Preziosi (2003)
# Volume fractions of reference [-]
equilibrium_fraction = dolfinx.default_scalar_type(0.85)
decohesion_fraction  = dolfinx.default_scalar_type(0.60)
interaction_fraction = dolfinx.default_scalar_type(0.00)
# 
# decohesion pressure [Pa]
decohesion_pressure  = dolfinx.default_scalar_type(-500)
# 
# Compute the Byrne law coefficients
alpha, beta          = constitutive_laws.Byrne_coefficients(equilibrium_fraction,interaction_fraction,decohesion_fraction,decohesion_pressure)
# Estimate the compressibility of the cells
compressibility_cell = alpha * equilibrium_fraction**2/((1-equilibrium_fraction)**beta)
# 
M = equilibrium_fraction*compressibility_cell
if mpi4py.MPI.COMM_WORLD.rank == 0:
	print(f'linearized longitudinal Modulus = {M}')
	print(f'cell compressibility Modulus = {compressibility_cell}')
# 
# cv = permeability*M/interstitial_fluid_viscosity
if permea_eps_square == False:
	cv = permeability*M/interstitial_fluid_viscosity
else:
	cv = (permeability*(1-equilibrium_fraction)**2)*M/interstitial_fluid_viscosity
# 
# --------------------------------------- #
# 	   		 Boundary values			  #
# --------------------------------------- #
# Pressures [Pa]
p_right = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0.))
# p_left  = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(100.))
p_left  = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(10.))
# For Terzaghi
p_left_terzaghi = 10
# 
# --------------------------------------- #
# 	   			Operators				  #
# --------------------------------------- #
# Coordinates, for axisymmetry
x = ufl.SpatialCoordinate(domain)
# radius + small shift to avoid division by 0
r = abs(x[1]) #+1e-10
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
solution.x.array[Pressure_to_MS]=dolfinx.default_scalar_type(p_left)
solution.x.scatter_forward()
previous_solution.x.array[Fraction_to_MS]=dolfinx.default_scalar_type(equilibrium_fraction)
previous_solution.x.array[Pressure_to_MS]=dolfinx.default_scalar_type(p_left)
previous_solution.x.scatter_forward()
# 
# --------------------------------------- #
# 	   		Boundary conditions			  #
# --------------------------------------- #
no_slip = False
bcs    = []
fdim   = domain.topology.dim - 1
# 
# bottom, right, top, left = 1, 2, 3, 4
# 
# bottom
facets = facet_tags.find(1)
# (vy = 0, tx = 0, grad(pl).n = 0)
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1)))
# right
facets = facet_tags.find(2)
# (vx = 0, vy = 0, pl = 0)
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(0), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(0)))
# 
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1)))
# 
# top
facets = facet_tags.find(3)
# No-slip condition
if no_slip:
	dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(0), fdim, facets)
	bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(0)))
# 
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(2).sub(1), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(2).sub(1)))
# left
facets = facet_tags.find(4)
# (tx = p0, ty = 0, grad(pl).n = 0)
dofs   = dolfinx.fem.locate_dofs_topological(MS.sub(0), fdim, facets)
bcs.append(dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0), dofs, MS.sub(0)))
# 
# --------------------------------------- #
# 	   			Weak Form				  #
# --------------------------------------- #
# 
fluid_pressure,  cell_fraction,  cell_velocity  = ufl.split(solution)
fluid_pressure_n,  cell_fraction_n,  cell_velocity_n = ufl.split(previous_solution)
# 
# We compute the mean speed similarly to cranck nickolson method
# mean_cell_fraction = 0.5 * (cell_fraction + cell_fraction_n)
# 
# Weak form
# mass epsilon c
F = (
	(1/dt)*(cell_fraction-cell_fraction_n)*q_e*r*dx 
	# + functions.div_cyl_x(mean_cell_fraction*cell_velocity,x)*q_e*r*dx
	+ functions.div_cyl_x(cell_fraction*cell_velocity,x)*q_e*r*dx
	)
# mass epsilon m
F+= (
	(1/dt)*(cell_fraction_n-cell_fraction)*q_p*r*dx 
	+ functions.div_cyl_x((1-cell_fraction)*cell_velocity,x)*q_p*r*dx 
	)
if permea_eps_square == False:
	F+= ((permeability)/interstitial_fluid_viscosity)*ufl.dot(ufl.grad(fluid_pressure), ufl.grad(q_p))*r*dx
else:
	F+= ((permeability*(1-cell_fraction)**2)/interstitial_fluid_viscosity)*ufl.dot(ufl.grad(fluid_pressure), ufl.grad(q_p))*r*dx
	
# test velocity
F+=(
	- fluid_pressure*functions.div_cyl_x(q_v,x)*r*dx
	- (constitutive_laws.Byrne_law(cell_fraction,equilibrium_fraction,interaction_fraction,decohesion_fraction,decohesion_pressure,log=True))*functions.div_cyl_x(q_v,x)*r*dx 
	# - (constitutive_laws.Linearized_Byrne(cell_fraction,equilibrium_fraction,interaction_fraction,decohesion_fraction,decohesion_pressure))*functions.div_cyl_x(q_v,x)*r*dx 
	+ scafflod_like_viscosity*ufl.inner(functions.grad_cyl_x(cell_velocity,x), functions.grad_cyl_x(q_v,x))*r*dx 
	+ scafflod_like_viscosity*ufl.inner(functions.grad_cyl_x(cell_velocity,x).T, functions.grad_cyl_x(q_v,x))*r*dx 
	+ p_left*ufl.dot(normal, q_v)*r*ds(4) 
	)
# 
# --------------------------------------- #
# 	   				Solver			 	  #
# --------------------------------------- #
# Non linear problem definition
dsolution = ufl.TrialFunction(MS)
J         = ufl.derivative(F, solution, dsolution)
Problem   = NonlinearProblem(F, solution, bcs = bcs, J = J)
solver    = functions.set_non_linear_solver_parameters(domain, Problem, 1e-10, 1e-12, "incremental", 50, log_newton=False)
# 
# --------------------------------------- #
# 	   	  Solving & Post-treatment		  #
# --------------------------------------- #
# 
# vtk outputs_fluid_terzaghi
pressure_vtk       = dolfinx.io.VTKFile(domain.comm, "outputs_fluid_terzaghi/VTK/pressure.pvd", "w")
pressure_vtk.write_mesh(domain)
fraction_vtk       = dolfinx.io.VTKFile(domain.comm, "outputs_fluid_terzaghi/VTK/fraction.pvd", "w")
fraction_vtk.write_mesh(domain)
velocity_vtk       = dolfinx.io.VTKFile(domain.comm, "outputs_fluid_terzaghi/VTK/velocity.pvd", "w")
velocity_vtk.write_mesh(domain)
porosity_vtk       = dolfinx.io.VTKFile(domain.comm, "outputs_fluid_terzaghi/VTK/porosity.pvd", "w")
porosity_vtk.write_mesh(domain)
cell_pressure_vtk  = dolfinx.io.VTKFile(domain.comm, "outputs_fluid_terzaghi/VTK/cell_pressure.pvd", "w")
cell_pressure_vtk.write_mesh(domain)
# ------------------------------------------------------------------------ #
# 
# Create output lists in time and space for the IF pressure
pressure_x_L          = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
pressure_x_L_over_2   = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
porosity_all_L        = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
porosity_all_L_over_2 = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
fraction_all          = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
velocity_0            = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
velocity_L_over_2     = numpy.zeros(num_steps+1, dtype=dolfinx.default_scalar_type)
pressure_space0       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressure_space1       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
pressure_space2       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
velocity_space0       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
velocity_space1       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
velocity_space2       = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
# 
# Error 
L2_p = numpy.zeros(num_steps, dtype=dolfinx.default_scalar_type)
L2_v = numpy.zeros(num_steps, dtype=dolfinx.default_scalar_type)
# in space
Signed_p0 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
Signed_v0 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
Signed_p1 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
Signed_v1 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
Signed_p2 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
Signed_v2 = numpy.zeros(num_points, dtype=dolfinx.default_scalar_type)
# 
# ------------------------------------------------------------------------ #
# 
# initialise time
t = 0
# Export initial solution at t=0s
__pressure, __fraction, __v  = previous_solution.split()
__pressure.name              = "Fluid Pressure"
__fraction.name              = "Cell Fraction"
velocity.interpolate(previous_solution.sub(2))
velocity.x.scatter_forward()
velocity.name                = "Cell Velocity"
porosity.interpolate(porosity_expr)
porosity.x.scatter_forward()
porosity.name                = "Porosity"
cell_pressure.interpolate(cell_pressure_expr)
cell_pressure.x.scatter_forward()
cell_pressure.name                = "cell pressure"
# 
# Initial step in lists
evaluate_point(domain, __pressure, cells_x_L, points[0], pressure_x_L, 0)
evaluate_point(domain, __pressure, cells_x_L_over_2, points[1], pressure_x_L_over_2, 0)
evaluate_point(domain, porosity, cells_x_L, points[0], porosity_all_L, 0)
evaluate_point(domain, porosity, cells_x_L_over_2, points[1], porosity_all_L_over_2, 0)
evaluate_point(domain, __fraction, cells_x_L_over_2, points[1], fraction_all, 0)
# Export in the vtk files
pressure_vtk.write_function(__pressure.collapse(),t)
fraction_vtk.write_function(__fraction.collapse(),t)
velocity_vtk.write_function(velocity,t)
porosity_vtk.write_function(porosity,t)
cell_pressure_vtk.write_function(cell_pressure,t)
# 
for iteration in range(num_steps):
	# increment time
	t+=dt 
	# 
	if mpi4py.MPI.COMM_WORLD.rank == 0:
		print('Step: ',iteration+1,'/',num_steps, 'time:', t, ' s')
	# Solve
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
	# Export
	__pressure, __fraction, __v  = previous_solution.split()
	__pressure.name              = "Fluid Pressure"
	__fraction.name              = "Cell Fraction"
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
		# Export in the vtk files
		pressure_vtk.write_function(__pressure.collapse(),t)
		fraction_vtk.write_function(__fraction.collapse(),t)
		velocity_vtk.write_function(velocity,t)
		porosity_vtk.write_function(porosity,t)
		cell_pressure_vtk.write_function(cell_pressure,t)
	# Compute L2 norm for pressure
	error_L2p     = L2_error_p(domain,P1,__pressure)
	L2_p[iteration] = error_L2p
	error_L2v     = L2_error_v(domain,P1,velocity)
	L2_v[iteration] = error_L2v
	if mpi4py.MPI.COMM_WORLD.rank == 0:
		print(f"L2-error p {error_L2p:.2e} [Pa]")
		print(f"L2-error v {error_L2v:.2e} [m/s]")
	# Evaluate the functions
	# in time
	evaluate_point(domain, __pressure, cells_x_L, points[0], pressure_x_L, iteration+1)
	evaluate_point(domain, __pressure, cells_x_L_over_2, points[1], pressure_x_L_over_2, iteration+1)
	evaluate_point(domain, porosity, cells_x_L, points[0], porosity_all_L, iteration+1)
	evaluate_point(domain, porosity, cells_x_L_over_2, points[1], porosity_all_L_over_2, iteration+1)
	evaluate_point(domain, __fraction, cells_x_L, points[0], fraction_all, iteration+1)
	evaluate_point(domain, velocity.sub(0), cells_x_0, points[2], velocity_0, iteration+1)
	evaluate_point(domain, velocity.sub(0), cells_x_L_over_2, points[1], velocity_L_over_2, iteration+1)
	# in space
	if iteration == n0:
		for ii in range(num_points):
			evaluate_point(domain, __pressure, colliding_cells.links(ii+2), points[ii+2], pressure_space0, ii)
			evaluate_point(domain, velocity.sub(0), colliding_cells.links(ii+2), points[ii+2], velocity_space0, ii)
		t0 = t
	elif iteration == n1:
		for ii in range(num_points):
			evaluate_point(domain, __pressure, colliding_cells.links(ii+2), points[ii+2], pressure_space1, ii)
			evaluate_point(domain, velocity.sub(0), colliding_cells.links(ii+2), points[ii+2], velocity_space1, ii)
		t1 = t
	elif iteration == n2:
		for ii in range(num_points):
			evaluate_point(domain, __pressure, colliding_cells.links(ii+2), points[ii+2], pressure_space2, ii)
			evaluate_point(domain, velocity.sub(0), colliding_cells.links(ii+2), points[ii+2], velocity_space2, ii)
		t2 = t
	# 
# 
# --------------------------------------- #
# 	   	  Post-processing Terzaghi		  #
# --------------------------------------- #
# print some error statistics
if mpi4py.MPI.COMM_WORLD.rank == 0:
	print(f"L2 error p, min {numpy.min(L2_p):.2e} Pa, mean {numpy.mean(L2_p):.2e} Pa, max {numpy.max(L2_p):.2e} Pa, std {numpy.std(L2_p):.2e} Pa")
	print(f"L2 error v, min {numpy.min(L2_v):.2e} m/s, mean {numpy.mean(L2_v):.2e} m/s, max {numpy.max(L2_v):.2e} m/s, std {numpy.std(L2_v):.2e} m/s")
	print("Normalised to max")
	print(f"L2 error p, min {numpy.min([100*x/numpy.max(pressure_x_L) for x in L2_p]):.2e} %, mean {numpy.mean([100*x/numpy.max(pressure_x_L) for x in L2_p]):.2e} %, max {numpy.max([100*x/numpy.max(pressure_x_L) for x in L2_p]):.2e} %, std {numpy.std([100*x/numpy.max(pressure_x_L) for x in L2_p]):.2e} %")
	print(f"L2 error v, min {numpy.min([100*x/numpy.max(velocity_0) for x in L2_v]):.2e} %, mean {numpy.mean([100*x/numpy.max(velocity_0) for x in L2_v]):.2e} %, max {numpy.max([100*x/numpy.max(velocity_0) for x in L2_v]):.2e} %, std {numpy.std([100*x/numpy.max(velocity_0) for x in L2_v]):.2e} %")
# Evaluate final time
end_t = time.time()
t_hours = int((end_t-begin_t)//3600)
tmin = int(((end_t-begin_t)%3600)//60)
tsec = int(((end_t-begin_t)%3600)%60)
if mpi4py.MPI.COMM_WORLD.rank == 0:
	print(f"FEM operated with {num_steps} iterations, in {t_hours} h {tmin} min {tsec} sec")
# 
if mpi4py.MPI.COMM_WORLD.rank == 0:
	###################################################
	################ Analytical solutions #############
	###################################################
	# 
	# 
	tstart = 0
	y=0
	t=numpy.linspace(0,t_fin,num_steps+1)
	pressure4 = numpy.zeros(num_steps+1)
	kmax=1e3
	for i in range(num_steps+1):
		pressure4[i] = functions.terzaghi(p_left_terzaghi,length_cylinder,cv,y,t[i],tstart,int(kmax))
	# 
	y=length_cylinder/2
	t=numpy.linspace(0,t_fin,num_steps+1)
	pressure5 = numpy.zeros(num_steps+1)
	for i in range(num_steps+1):
		pressure5[i] = functions.terzaghi(p_left_terzaghi,length_cylinder,cv,y,t[i],tstart,int(kmax))
	# 
	pressure0 = numpy.zeros(num_points)
	for i in range(num_points):
		pressure0[i] = functions.terzaghi(p_left_terzaghi,length_cylinder,cv,x_check[i],t0,tstart,int(kmax))
	# 
	pressure1 = numpy.zeros(num_points)
	for i in range(num_points):
		pressure1[i] = functions.terzaghi(p_left_terzaghi,length_cylinder,cv,x_check[i],t1,tstart,int(kmax))
	# 
	pressure2 = numpy.zeros(num_points)
	for i in range(num_points):
		pressure2[i] = functions.terzaghi(p_left_terzaghi,length_cylinder,cv,x_check[i],t2,tstart,int(kmax))
	# 
	velocity0 = numpy.zeros(num_points)
	for i in range(num_points):
		velocity0[i] = functions.terzaghi_v(p_left_terzaghi,length_cylinder,cv,x_check[i],t0,tstart,int(kmax),M)
	# 
	velocity1 = numpy.zeros(num_points)
	for i in range(num_points):
		velocity1[i] = functions.terzaghi_v(p_left_terzaghi,length_cylinder,cv,x_check[i],t1,tstart,int(kmax),M)
	# 
	velocity2 = numpy.zeros(num_points)
	for i in range(num_points):
		velocity2[i] = functions.terzaghi_v(p_left_terzaghi,length_cylinder,cv,x_check[i],t2,tstart,int(kmax),M)
	# 
	Signed_p0 = [(pressure_space0[index]-pressure0[index]) for index in range(num_points)]
	Signed_p1 = [(pressure_space1[index]-pressure1[index]) for index in range(num_points)]
	Signed_p2 = [(pressure_space2[index]-pressure2[index]) for index in range(num_points)]
	Signed_v0 = [(velocity_space0[index]-velocity0[index]) for index in range(num_points)]
	Signed_v1 = [(velocity_space1[index]-velocity1[index]) for index in range(num_points)]
	Signed_v2 = [(velocity_space2[index]-velocity2[index]) for index in range(num_points)]
	# 
	functions.export_to_csv(
		[x_check,pressure_space0,pressure_space1,pressure_space2,velocity_space0,velocity_space1,velocity_space2,Signed_p0,Signed_v0,Signed_p1,Signed_v1,Signed_p2,Signed_v2],
		"outputs_fluid_terzaghi/f_Results_x_"+str(permeability)+".csv",
		["x","pressure_space0","pressure_space1","pressure_space2","velocity_space0","velocity_space1","velocity_space2","Signed_p0","Signed_v0","Signed_p1","Signed_v1","Signed_p2","Signed_v2"]
		)
	functions.export_to_csv(
		[t,porosity_all_L,porosity_all_L_over_2,fraction_all,velocity_0,velocity_L_over_2,pressure_x_L,pressure_x_L_over_2,L2_p,L2_v, [x/numpy.max(pressure_x_L) for x in L2_p], [x/numpy.max(velocity_0) for x in L2_v]],
		"outputs_fluid_terzaghi/f_Results_t_"+str(permeability)+".csv",
		["t","porosity_all_L","porosity_all_L_over_2","fraction_all","velocity_0","velocity_L_over_2","pressure_x_L","pressure_x_L_over_2","L2_p","L2_v","L2_p_normmax","L2_v_normmax"]
		)
	functions.export_to_csv(
		[x_check,pressure0,pressure1,pressure2,velocity0,velocity1,velocity2],
		"outputs_fluid_terzaghi/f_Terzaghi_x_"+str(permeability)+".csv",
		["x","pressure0","pressure1","pressure2","velocity0","velocity1","velocity2"]
		)
	functions.export_to_csv(
		[t,pressure4,pressure5],
		"outputs_fluid_terzaghi/f_Terzaghi_t_"+str(permeability)+".csv",
		["t","pressure4","pressure5"]
		)
	# 
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

	input_file = "outputs_fluid_terzaghi/f_Results_t_"+str(permeability)+".csv"
	output_file = "outputs_fluid_terzaghi/ft_Results_t_"+str(permeability)+".csv"
	transpose_csv(input_file,output_file)

	input_file = "outputs_fluid_terzaghi/f_Results_x_"+str(permeability)+".csv"
	output_file = "outputs_fluid_terzaghi/ft_Results_x_"+str(permeability)+".csv"
	transpose_csv(input_file,output_file)

	input_file = "outputs_fluid_terzaghi/f_Terzaghi_x_"+str(permeability)+".csv"
	output_file = "outputs_fluid_terzaghi/ft_Terzaghi_x_"+str(permeability)+".csv"
	transpose_csv(input_file,output_file)

	input_file = "outputs_fluid_terzaghi/f_Terzaghi_t_"+str(permeability)+".csv"
	output_file = "outputs_fluid_terzaghi/ft_Terzaghi_t_"+str(permeability)+".csv"
	transpose_csv(input_file,output_file)


#EoF