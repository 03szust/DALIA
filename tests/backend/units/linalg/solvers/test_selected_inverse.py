# tests/backend/units/linalg/solvers/test_selected_inverse.py

import pytest
import numpy as np

from dalia.backend.linalg.solvers import DenseSolver, SparseSolver, CuDSS
from dalia.backend.config import memory_regime, cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from .conftest import INTERNAL_DEVICE_TYPES, OVERWRITE

class TestSolve:

    @pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("overwrite", OVERWRITE)
    def test_si_dense(self, matrix_factory, device_type, overwrite):
        """Test solving a dense linear system Ax = b."""
        M = matrix_factory("DenseMatrix", shape=(3, 3), hw_target=device_type)
        E = matrix_factory("DenseMatrix", shape=(3, 3), data=np.eye(3), hw_target=device_type)
        A = M.T @ M + E  # Make it symmetric positive definite
        solver = DenseSolver(A)
        solver.factorize()
        x = solver.selected_inverse(overwrite)
        # Verify the solution is correct
        print(np.linalg.inv(A.toarray()))
        print(x.toarray())
        assert np.allclose(np.linalg.inv(A.toarray()), x.toarray())

    @pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
    @pytest.mark.parametrize("overwrite", OVERWRITE)
    def test_si_sparse(self, matrix_factory, device_type, overwrite):
        """Test solving a sparse linear system Ax = b."""
        M = matrix_factory("SparseMatrix", shape=(3, 3), hw_target=device_type)
        E = matrix_factory("SparseMatrix", shape=(3, 3), data=np.eye(3), hw_target=device_type)
        A = M.T @ M + E  # Make it symmetric positive definite
        solver = SparseSolver(A)
        solver.factorize()
        x = solver.selected_inverse(overwrite)
        # Verify the solution is correct
        assert np.allclose(np.linalg.inv(A.toarray()), x.toarray())

    @pytest.mark.parametrize("overwrite", OVERWRITE)
    def test_si_cudss(self, matrix_factory, overwrite):
        """Test solving a linear system using CuDSS."""
        if cupy_version is None:
            pytest.skip("CuPy is not installed")
        if nvmath_version is None:
            pytest.skip("NVMATH is not installed")
        M = matrix_factory("SparseMatrix", shape=(3, 3), hw_target="accelerator")
        E = matrix_factory("SparseMatrix", shape=(3, 3), data=np.eye(3), hw_target="accelerator")
        A = M.T @ M + E  # Make it symmetric positive definite
        solver = CuDSS(A)
        solver.analyze()
        solver.factorize()
        x = solver.selected_inverse(overwrite)
        # Verify the solution is correct
        assert np.allclose(np.linalg.inv(A.toarray()), x.toarray())