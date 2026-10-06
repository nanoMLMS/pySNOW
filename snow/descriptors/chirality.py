# a module for chirality measures.
# as of now, the Hausdorff chirality measure (hcm) is implemented
# Thanks to dr. Giacomo Becatti for the implementation 

import numpy as np
import datetime
from scipy import optimize
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist



from snow.misc.rototranslation import transform_points


def calculate_hausdorff_distance(set_a, set_b):
    """
    Compute Hausdorff distance between two sets of points.

    The Hausdorff distance is the maximum distance from any point
    in one set to the nearest point in the other set.

    Parameters
    ----------
    set_a : np.ndarray
        Array of shape (n, 3) containing points in first set.
    set_b : np.ndarray
        Array of shape (m, 3) containing points in second set.

    Returns
    -------
    float
        Hausdorff distance between set_a and set_b.
    """
    # Build KD-trees for efficient nearest neighbor queries
    tree_b = KDTree(set_b)
    tree_a = KDTree(set_a)

    # Query distances from set_a to nearest points in set_b
    distances_a_to_b, _ = tree_b.query(set_a)

    # Query distances from set_b to nearest points in set_a
    distances_b_to_a, _ = tree_a.query(set_b)

    # Calculate Hausdorff distance
    forward_hausdorff = np.max(distances_a_to_b)
    backward_hausdorff = np.max(distances_b_to_a)

    return max(forward_hausdorff, backward_hausdorff)


def hausdorff_chirality(coords, method="bfgs", n_points=4, pop_size=None,
                        tol=None, verbose=False, **kwargs):
    """
    Compute the Hausdorff chirality measure (HCM) for a set of atomic coordinates.

    The HCM quantifies chirality by finding the optimal rigid transformation
    (rotation + translation) that minimizes the Hausdorff distance between a
    set of points and its mirror image.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (n_atoms, 3) containing atomic coordinates.
    method : str, default "bfgs"
        Optimization method: "bfgs", "differential", or "mixed".
    n_points : int, default 4
        Number of points per axis for grid sampling in bfgs method
        (total n_points^3 initial configurations).
    pop_size : int, optional
        Population size for differential evolution. Required if method is
        'differential' or 'mixed'. Also accepts 'population_size' via kwargs.
    tol : float, optional
        Tolerance for differential evolution. Required if method is
        'differential' or 'mixed'. Also accepts 'tolerance' via kwargs.
    verbose : bool, default False
        If True, print additional information during calculation.
    **kwargs : dict, optional
        Additional arguments. Supports 'population_size' and 'tolerance'
        as aliases for 'pop_size' and 'tol'.

    Returns
    -------
    float
        Normalized Hausdorff chirality measure. Values closer to 0 indicate
        lower chirality (more achiral) for the best found registration.
    """
    # Handle aliases from kwargs for backward compatibility
    if pop_size is None and 'population_size' in kwargs:
        pop_size = kwargs['population_size']
    if tol is None and 'tolerance' in kwargs:
        tol = kwargs['tolerance']

    # Convert to array if needed
    positions = np.asarray(coords)

    # Center the nanoparticle at origin first
    center = np.mean(positions, axis=0)
    centered_positions = positions - center

    # Create mirror image (flipping x-coordinate)
    mirror_positions = centered_positions.copy()
    mirror_positions[:, 0] *= -1

    # Pre-calculate diameter for normalization
    diameter = pdist(positions).max()

    # Define optimization function
    def objective_function(params):
        # Scale translations relative to diameter
        dx, dy, dz = (
            params[0] * diameter,
            params[1] * diameter,
            params[2] * diameter,
        )
        rx, ry, rz = params[3], params[4], params[5]

        # Transform mirror image
        transformed_mirror = transform_points(
            mirror_positions, dx, dy, dz, rx, ry, rz
        )

        # Calculate Hausdorff distance
        hausdorff_dist = calculate_hausdorff_distance(
            centered_positions, transformed_mirror
        )

        # Return normalized distance
        return hausdorff_dist / diameter

    best_value = float("inf")
    method_lower = str(method).lower()

    # Minimize the objective function
    if method_lower == "bfgs":
        initial_points = []
        n_pts = int(n_points)
        for rx in np.linspace(0, 2 * np.pi, n_pts):
            for ry in np.linspace(0, 2 * np.pi, n_pts):
                for rz in np.linspace(0, 2 * np.pi, n_pts):
                    initial_points.append([0, 0, 0, rx, ry, rz])

        for i, x0 in enumerate(initial_points):
            if verbose:
                print(f"Configuration {i}/{len(initial_points)}")
                print("-" * 50)
                print("Beginning evaluation through BFGS")
                x = datetime.datetime.now()
                print(f"Date: {x.day}-{x.month}-{x.year} \t Time:{x.time()}")

            result = optimize.minimize(
                objective_function,
                x0=x0,
                method="BFGS",
                options={"gtol": 1e-6, "maxiter": 200},
            )

            if verbose:
                print("-" * 50)
                x = datetime.datetime.now()
                print(f"BFGS for point {i}/{len(initial_points)} completed")
                print(f"Date: {x.day}-{x.month}-{x.year} \t Time:{x.time()}")
                print("\n\n")
                print("Dumping results from the optimization:")
                print(result)
                print("-" * 50)
                print("\nCurrent determination of the HCM")
                print(f"\nOld best value: {best_value}")
                print(f"New computed value: {result.fun}")
                print(f"Change in value: {result.fun - best_value:.2e}")
                print("\n\n")
                print("-" * 50)

            if result.fun < best_value:
                best_value = result.fun

        return best_value

    if method_lower == "differential":
        if pop_size is None:
            raise ValueError(
                "pop_size (or population_size) is required when using method 'differential'"
            )
        if tol is None:
            raise ValueError(
                "tol (or tolerance) is required when using method 'differential'"
            )

        bounds = (
            (-0.5, 0.5),  # dx/diameter
            (-0.5, 0.5),  # dy/diameter
            (-0.5, 0.5),  # dz/diameter
            (0, np.pi),  # rx
            (0, np.pi),  # ry
            (0, np.pi),  # rz
        )
        global_result = optimize.differential_evolution(
            objective_function,
            bounds=bounds,
            popsize=pop_size,
            tol=tol,
            mutation=(0.5, 1.0),
            recombination=0.7,
            updating="deferred",
            workers=1, # Changed from -1 to 1 to avoid multiprocessing issues
        )

        return global_result.fun

    if method_lower == "mixed":
        if pop_size is None:
            raise ValueError(
                "pop_size (or population_size) is required when using method 'mixed'"
            )
        if tol is None:
            raise ValueError(
                "tol (or tolerance) is required when using method 'mixed'"
            )

        bounds = (
            (-0.5, 0.5),  # dx/diameter
            (-0.5, 0.5),  # dy/diameter
            (-0.5, 0.5),  # dz/diameter
            (0, np.pi),  # rx
            (0, np.pi),  # ry
            (0, np.pi),  # rz
        )
        global_result = optimize.differential_evolution(
            objective_function,
            bounds=bounds,
            popsize=pop_size,
            tol=tol,
            mutation=(0.5, 1.0),
            recombination=0.7,
            updating="deferred",
            workers=1,
        )

        starting_points = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, np.pi / 2, 0, 0], # 90° around x
            [0, 0, 0, 0, np.pi / 2, 0], # 90° around y
            [0, 0, 0, 0, 0, np.pi / 2], # 90° around z
            [0, 0, 0, np.pi / 4, np.pi / 4, np.pi / 4], # 45 aroaund all axes
        ]

        # Add more starting points with finer rotational sampling
        #TODO: Rotation only from 0 to PI assuming simmetry. The simmetry has been confirmed on a sample but is not rigorously verified
        for i in range(10):
            global_result = optimize.differential_evolution(
                objective_function,
                bounds=bounds,
                popsize=pop_size,
                tol=tol,
                mutation=(0.5, 1.0),
                recombination=0.7,
                updating="deferred",
                workers=1,
            )
            starting_points.append(global_result.x)

        for x0 in starting_points:
            # Use L-BFGS-B first for rough optimization (faster)
            rough_result = optimize.minimize(
                objective_function,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-6, "gtol": 1e-6, "maxiter": 200},
            )

            # Refine with SLSQP for precision
            result = optimize.minimize(
                objective_function,
                x0=rough_result.x,
                method="SLSQP",
                bounds=bounds,
                options={"ftol": 1e-10, "maxiter": 500},
            )


            # Try Nelder-Mead as a final refinement (without bounds)
            # This can sometimes find better solutions in the local neighborhood
            nm_result = optimize.minimize(
                objective_function,
                x0=result.x,
                method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 200},
            )

            # Use the best of SLSQP and Nelder-Mead
            final_result = (
                nm_result if nm_result.fun < result.fun else result
            )

            if final_result.fun < best_value:
                best_value = final_result.fun

        return best_value

    raise ValueError(
        f"Unknown method '{method}'. Supported methods: 'bfgs', 'differential', 'mixed'."
    )
