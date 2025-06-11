import numpy as np

import numpy as np
import scipy.linalg
import warnings

def quadratic_partial_conjugate(hessian, indices, cond_limit=1e12):
    """
    Computes a robust partial convex conjugate of a quadratic function,
    preserving the original matrix structure and handling ill-conditioned matrices.

    This function is designed for numerical stability by:
    1. Validating the input matrix for symmetry.
    2. Using a Cholesky decomposition-based solver, which is highly efficient
       and stable for symmetric positive-definite matrices.
    3. Checking the matrix condition number and warning if it's high.
    4. Providing a robust fallback to the pseudo-inverse for matrices that
       are not positive-definite (i.e., singular or semidefinite).
    5. Ensuring the output is numerically symmetric.

    Args:
        hessian (np.ndarray): The symmetric positive semidefinite Hessian matrix.
        indices (list or np.ndarray): A list of indices of the variables with
                                      respect to which the conjugate is computed.
        cond_limit (float): The condition number limit above which a warning is
                            issued about potential numerical instability.

    Returns:
        np.ndarray: The n x n Hessian matrix of the resulting partial convex
                    conjugate, with the original index order preserved.
                    
    Raises:
        ValueError: If the input matrix is not square or not symmetric.
    """
    # --- 1. Input Validation ---
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Input hessian must be a square 2D matrix.")
    if not np.allclose(hessian, hessian.T):
        raise ValueError("Input hessian must be symmetric.")
    
    n = hessian.shape[0]
    if not indices:
        return hessian.copy() # Return a copy if no indices are conjugated

    # --- 2. Partitioning ---
    indices = np.asarray(indices)
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True
    remaining_indices = np.where(~mask)[0]

    H_aa = hessian[np.ix_(remaining_indices, remaining_indices)]
    H_ab = hessian[np.ix_(remaining_indices, indices)]
    H_bb = hessian[np.ix_(indices, indices)]
    
    # Handle the edge case where all variables are conjugated
    if H_aa.shape[0] == 0:
        try:
            return np.linalg.inv(H_bb)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(H_bb)

    # --- 3. Robust Computation using Symmetry ---
    try:
        # Check condition number to warn about ill-conditioned subproblems
        cond_H_bb = np.linalg.cond(H_bb)
        if cond_H_bb > cond_limit:
            warnings.warn(
                f"The submatrix for conjugation is ill-conditioned (cond={cond_H_bb:.2e}). "
                "Results may be inaccurate.",
                UserWarning
            )

        # Use Cholesky factorization: the best method for symmetric, pos-def matrices
        L, low = scipy.linalg.cho_factor(H_bb, lower=True)

        # Efficiently compute the new blocks using the factorization
        # new_H_ab = -H_ab @ inv(H_bb) is solved via H_bb @ X = -H_ab
        new_H_ab = -scipy.linalg.cho_solve((L, low), H_ab.T).T
        
        # new_H_bb = inv(H_bb) is solved via H_bb @ X = I
        identity_bb = np.identity(H_bb.shape[0], dtype=H_bb.dtype)
        new_H_bb = scipy.linalg.cho_solve((L, low), identity_bb)

    except np.linalg.LinAlgError:
        # --- 4. Fallback for non-positive-definite or singular matrices ---
        warnings.warn(
            "Cholesky decomposition failed. The submatrix may not be positive-definite. "
            "Falling back to the pseudo-inverse, which may be slow.",
            UserWarning
        )
        H_bb_pinv = np.linalg.pinv(H_bb)
        new_H_bb = H_bb_pinv
        new_H_ab = -H_ab @ H_bb_pinv

    # --- 5. Assemble Final Hessian, Enforcing Symmetry ---
    # Schur complement: H_aa - H_ab @ H_bb_inv @ H_ba
    # Reuse new_H_ab:   H_aa + new_H_ab @ H_ba (since new_H_ab = -H_ab @ H_bb_inv)
    H_ba = H_ab.T
    new_H_aa = H_aa + new_H_ab @ H_ba

    # The new off-diagonal block is the transpose of the other
    new_H_ba = new_H_ab.T

    # Assemble the final matrix
    H_star = np.empty_like(hessian, dtype=float)
    H_star[np.ix_(remaining_indices, remaining_indices)] = new_H_aa
    H_star[np.ix_(indices, indices)] = new_H_bb
    H_star[np.ix_(remaining_indices, indices)] = new_H_ab
    H_star[np.ix_(indices, remaining_indices)] = new_H_ba

    # Final enforcement of symmetry to clean up potential minor floating point errors
    return (H_star + H_star.T) / 2.0