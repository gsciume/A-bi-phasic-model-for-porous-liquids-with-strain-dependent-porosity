# This document holds the functions defined
# by the user for the ease of computation
# such as the solver settings
# 
# Author: Thomas Lavigne
# Date: 18/09/2024
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
# 
#------------------------------------------------------------#
#                    User-defined functions                  #
#------------------------------------------------------------#
if __name__ == "__main__":
    print("Loading of the user-defined functions successfully completed.")
    # EoF