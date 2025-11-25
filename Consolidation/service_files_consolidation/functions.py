# This document holds the functions defined
# by the user for the ease of computation
# such as the solver settings
# 
# Author: Thomas Lavigne
# Date: 18/09/2024
#------------------------------------------------------------#
#                    Axisymmetric operators                  #
#------------------------------------------------------------#
# 
def grad_cyl_x(u,spatial_coordinates):
	"""
    Compute the gradient of a vector field in axisymmetric (cylindrical) coordinates (x, r, θ).

    Parameters
    ----------
    u : list or tuple of UFL expressions
        The vector field components [u_x, u_r] defined in the (x, r)-plane.
        - u[0] = axial component (along x)
        - u[1] = radial component (along r)

    spatial_coordinates : ufl.SpatialCoordinate
        The spatial coordinates of the mesh, typically obtained with
        `spatial_coordinates = ufl.SpatialCoordinate(mesh)`.

    Returns
    -------
    ufl.Tensor
        A 3x3 tensor representing the gradient ∇u in cylindrical coordinates
        with the assumption of axisymmetry (no dependence on θ).
        The θ-component is reconstructed using the standard cylindrical formula.
	"""
	import ufl
	r = abs(spatial_coordinates[1])
	return ufl.as_tensor([[u[0].dx(0), u[0].dx(1), 0.], [u[1].dx(0), u[1].dx(1), 0.], [0., 0., u[1]/r]])
# 
def div_cyl_x(u,spatial_coordinates):
	"""
    Compute the divergence of a vector field in axisymmetric (cylindrical) coordinates (x, r, θ).

    Parameters
    ----------
    u : list or tuple of UFL expressions
        The vector field components [u_x, u_r] defined in the (x, r)-plane.
        - u[0] = axial component (along x)
        - u[1] = radial component (along r)

    spatial_coordinates : ufl.SpatialCoordinate
        The spatial coordinates of the mesh, typically obtained with
        `spatial_coordinates = ufl.SpatialCoordinate(mesh)`.

    Returns
    -------
    ufl.Expression
        A scalar expression representing ∇·u in cylindrical coordinates
        under the axisymmetric assumption (no dependence on θ).
	"""
	r = abs(spatial_coordinates[1])
	return u[1]/r + u[0].dx(0) + u[1].dx(1)
#------------------------------------------------------------#
#                    Computation tools                       #
#------------------------------------------------------------#
# 
def set_non_linear_solver_parameters(mesh, Problem, atol, rtol, convergence_criterion, max_it, log_newton=True):
	"""
	Configures and returns a non-linear Newton solver for a given mesh and problem.

	Parameters:
	-----------
	mesh : dolfinx.Mesh
	    The computational mesh of the domain. This object provides the communication context for the solver.

	Problem : dolfinx.fem.NonlinearProblem
	    The non-linear problem to be solved. This contains the variational formulation for the system.

	atol : float
	    The absolute tolerance for the solver. Determines when the solution is considered sufficiently converged 
	    regardless of the relative tolerance.

	rtol : float
	    The relative tolerance for the solver. Convergence is determined based on relative changes between 
	    iterations.

	convergence_criterion : str
	    The criterion for convergence, usually set to 'incremental' or 'residual' depending on the desired 
	    stopping criterion.

	max_it : int
	    The maximum number of allowed iterations for the Newton solver before stopping.

	log_newton : bool, optional
	    If set to True (default), Newton convergence logging is enabled to display information during the 
	    solution process. This helps track the solver's progress.

	Returns:
	--------
	solver : dolfinx.nls.petsc.NewtonSolver
	    Configured Newton solver with the specified parameters. The solver is ready to solve the non-linear problem.

	Notes:
	------
	- Uses PETSc Krylov solver for preconditioning, specifically LU factorization with the "mumps" solver type.
	- The Newton solver is initialized with absolute and relative tolerances, as well as a maximum number of iterations.
	- Convergence behavior can be logged using the `log_newton` flag, providing insights into the iteration progress.
	- The Krylov solver (KSP) is configured to use direct factorization (LU) to handle the linear systems at each 
	  Newton step, ensuring robust convergence for difficult problems.

	Exceptions:
	-----------
	If any error occurs during solver configuration, the solver may not initialize correctly, so ensure that the 
	`mesh`, `Problem`, and other arguments are valid.
	"""
	from dolfinx.nls.petsc import NewtonSolver
	import petsc4py
	# 
	if log_newton:
		from dolfinx import log
		log.set_log_level(log.LogLevel.INFO)
	# 
	# set up the non-linear solver
	solver                       = NewtonSolver(mesh.comm, Problem)
	# Absolute tolerance
	solver.atol                  = atol
	# relative tolerance
	solver.rtol                  = rtol
	# Convergence criterion
	solver.convergence_criterion = convergence_criterion
	# Maximum iterations
	solver.max_it                = max_it
	# 
	# 
	ksp  = solver.krylov_solver
	opts = petsc4py.PETSc.Options()
	option_prefix = ksp.getOptionsPrefix()
	opts[f"{option_prefix}ksp_type"] = "preonly"
	opts[f"{option_prefix}pc_type"]  = "lu"
	opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
	ksp.setFromOptions()
	return solver
# 
def export_to_csv(data, filename, header=None):
	"""
	Exports data to a CSV file with an optional header.

	Parameters:
	----------
	data : list of lists
	    The data to be written to the CSV file. Each inner list represents a row.

	filename : str
	    The name of the CSV file to which the data will be written. This file will
	    be created if it doesn't exist, or overwritten if it does.

	header : list, optional
	    A list representing the header row. If provided, this will be written at the
	    top of the CSV file before any data rows.

	Exceptions:
	-----------
	If an error occurs during file writing, an error message will be printed
	with details of the exception.

	Notes:
	------
	- The file is written in write mode ('w'), so any existing file with the same
	  name will be overwritten.
	- The 'newline' parameter is set to an empty string to avoid extra blank lines
	  between rows on Windows systems.
	"""
	import csv
	try:
		with open(filename, 'w', newline='') as file:
			writer = csv.writer(file)
			if header:
				writer.writerow(header)
			writer.writerows(data)
			print(f"Data exported to {filename} successfully")
	except Exception as e:
		print(f"An error occurred while exporting data to {filename}: {e}")
# 
def RMSE(x,xref):
	"""
	Computes the Root Mean Square Error (RMSE) between two numpy arrays.

	Parameters:
	-----------
	x : numpy.ndarray
	    The array to be evaluated. This can represent predicted values, measurements, or any other data.

	xref : numpy.ndarray
	    The reference array of the same size as `x`, representing the true or expected values.

	Returns:
	--------
	rmse_value : float
	    The computed RMSE, which quantifies the average magnitude of the error between `x` and `xref`.

	Notes:
	------
	- RMSE is a commonly used metric in regression and error analysis. It provides a measure of how close predictions 
	  or estimates are to the true values.
	- The formula for RMSE is:
	    RMSE = sqrt((1/N) * Σ (x - xref)²)
	  where N is the number of elements in the arrays, and the sum is taken over all elements.
	- RMSE has the same units as the input arrays and is sensitive to large errors, making it useful for identifying 
	  significant deviations.
	"""
	import numpy as np
	return np.sqrt(np.mean((x-xref)**2))
# 
#------------------------------------------------------------#
#                    User-defined functions                  #
#------------------------------------------------------------#
# 
def terzaghi(p0,L,cv,y,t,tstart,kmax):
	"""
	Computes the pore pressure at a given position and time using Terzaghi's one-dimensional consolidation theory.

	Parameters:
	-----------
	p0 : float
	    The initial applied pressure (typically at the surface).
	    
	L : float
	    The length (thickness) of the sample.
	    
	cv : float
	    The coefficient of consolidation, which characterizes the rate of pore pressure dissipation.
	    
	y : float
	    The position within the sample where the pressure is being calculated (0 <= y <= L).
	    
	t : float
	    The time at which the pressure is being evaluated.
	    
	tstart : float
	    The start time of the consolidation process.
	    
	kmax : int
	    The maximum number of terms to include in the summation (affects accuracy).
	    
	Returns:
	--------
	pl : float
	    The pore pressure at position `y` and time `t` based on Terzaghi's consolidation theory.

	Notes:
	------
	- The accuracy of the result depends on the value of `kmax`. A higher `kmax` gives a more accurate solution but requires more computation.
	- This function uses a Fourier series to approximate the solution, which is typical for one-dimensional consolidation problems.
	- The function assumes that the consolidation process begins at `tstart`. For `t < tstart`, the pressure is considered zero.
	"""
	import numpy
	pression=0
	for k in range(1,kmax):
		pression += 4/numpy.pi*(-1)**(k-1)/(2*k-1)*numpy.cos((2*k-1)*0.5*numpy.pi*((L-y)/L))*numpy.exp(-(2*k-1)**2*0.25*numpy.pi**2*cv*(t-tstart)/L**2)
	pl = p0*pression
	return pl
# 
def terzaghi_v(p0,L,cv,x,t,tstart,kmax,M):
	"""
	Computes the scaffold velocity at a given position and time using Terzaghi's one-dimensional consolidation theory.

	Parameters:
	-----------
	p0 : float
	    The initial applied pressure (typically at the surface).
	    
	L : float
	    The length (thickness) of the sample.
	    
	cv : float
	    The coefficient of consolidation, which characterizes the rate of pore pressure dissipation.
	    
	x : float
	    The position within the sample where the pressure is being calculated (0 <= y <= L).
	    
	t : float
	    The time at which the pressure is being evaluated.
	    
	tstart : float
	    The start time of the consolidation process.
	    
	kmax : int
	    The maximum number of terms to include in the summation (affects accuracy).
	    
	Returns:
	--------
	pl : float
	    The pore pressure at position `x` and time `t` based on Terzaghi's consolidation theory.

	Notes:
	------
	- The accuracy of the result depends on the value of `kmax`. A higher `kmax` gives a more accurate solution but requires more computation.
	- This function uses a Fourier series to approximate the solution, which is typical for one-dimensional consolidation problems.
	- The function assumes that the consolidation process begins at `tstart`. For `t < tstart`, the pressure is considered zero.
	"""
	import numpy
	pression=0
	for k in range(1,kmax):
		pression += (2*cv)/(M*L)*(-1)**(k+1)*numpy.sin((2*k-1)*0.5*numpy.pi*((L-x)/L))*numpy.exp(-(2*k-1)**2*0.25*numpy.pi**2*cv*(t-tstart)/L**2)
	pl = p0*pression
	return pl
# 
def get_element_size_stats(mesh_obj) -> dict:
	"""
	Calculates the volume-weighted mean and standard deviation of the
	cell diameter (element size 'h') of a dolfinx mesh.

	The mean (h_bar) is calculated as:
	    h_bar = (integral_Omega h * dV) / (integral_Omega 1 * dV)

	The standard deviation (sigma_h) is calculated as:
	    sigma_h = sqrt( (integral_Omega (h - h_bar)^2 * dV) / (integral_Omega 1 * dV) )

	Args:
	    mesh_obj: The dolfinx.mesh.Mesh object.

	Returns:
	    A dictionary containing the mean ('mean_h') and standard 
	    deviation ('std_dev_h') of the element size.
	"""
	import dolfinx.mesh as mesh
	import dolfinx.fem as fem
	from mpi4py import MPI
	import ufl
	import numpy as np

	# 1. Define the element size h as the UFL cell diameter
	h = ufl.CellDiameter(mesh_obj)

	# Define the measure for integration (volume in 3D, area in 2D)
	dx = ufl.Measure("dx", domain=mesh_obj)

	# 2. Calculate the total measure (Volume/Area) of the domain Omega

	# Create a UFL form for the integral of 1 over Omega
	volume_form = fem.form(1.0 * dx)

	# Assemble the integral
	volume_integral = fem.assemble_scalar(volume_form)

	# Reduce the total volume across all processors
	total_volume = mesh_obj.comm.allreduce(volume_integral, op=MPI.SUM)

	if total_volume < np.finfo(float).eps:
		if MPI.COMM_WORLD.rank == 0:
		    print("Error: Total volume of the mesh is zero.")
		return {'mean_h': 0.0, 'std_dev_h': 0.0}

	# 3. Calculate the Mean Element Size (h_bar)

	# UFL form for the integral of h over Omega
	mean_h_integral_form = fem.form(h * dx)

	# Assemble and reduce
	integral_h = fem.assemble_scalar(mean_h_integral_form)
	total_integral_h = mesh_obj.comm.allreduce(integral_h, op=MPI.SUM)

	mean_h = total_integral_h / total_volume

	# 4. Calculate the Standard Deviation (sigma_h)

	# UFL form for the integral of (h - mean_h)^2 over Omega (The variance)
	variance_h_form = fem.form((h - mean_h)**2 * dx)

	# Assemble and reduce
	integral_variance_h = fem.assemble_scalar(variance_h_form)
	total_integral_variance_h = mesh_obj.comm.allreduce(integral_variance_h, op=MPI.SUM)

	# Variance is the total integral of (h - mean_h)^2 divided by the total volume
	variance_h = total_integral_variance_h / total_volume

	# Standard deviation is the square root of the variance
	std_dev_h = np.sqrt(variance_h)

	return {
	    'mean_h': mean_h,
	    'std_dev_h': std_dev_h
	}
# 
#____________________________________________________________#
# 					End of functions
#____________________________________________________________#
# 
# 
if __name__ == "__main__":
    print("Loading of the user-defined functions successfully completed.")
    # EoF