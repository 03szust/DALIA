# tests/backend/units/linalg/solvers/test_logdet.py

import pytest
import numpy as np

from dalia.backend.linalg.solvers import DenseSolver, SparseSolver, CuDSS
from dalia.backend.config import memory_regime, cupy_version, nvmath_version

if cupy_version is not None:
    import cupy as cp

from .conftest import INTERNAL_DEVICE_TYPES

class TestSolve:

    @pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
    def test_ld_dense(self, matrix_factory, device_type):
        """Test solving a dense linear system Ax = b."""
        M = matrix_factory("DenseMatrix", shape=(3, 3), hw_target=device_type)
        E = matrix_factory("DenseMatrix", shape=(3, 3), data=np.eye(3), hw_target=device_type)
        A = M.T @ M + E  # Make it symmetric positive definite
        solver = DenseSolver(A)
        solver.factorize()
        x = solver.logdet()
        # Verify the solution is correct
        assert np.allclose(np.linalg.slogdet(A.toarray())[1], x)

    @pytest.mark.parametrize("device_type", INTERNAL_DEVICE_TYPES)
    def test_ld_sparse(self, matrix_factory, device_type):
        """Test solving a sparse linear system Ax = b."""
        M = matrix_factory("SparseMatrix", shape=(3, 3), hw_target=device_type)
        E = matrix_factory("SparseMatrix", shape=(3, 3), data=np.eye(3), hw_target=device_type)
        A = M.T @ M + E  # Make it symmetric positive definite
        solver = SparseSolver(A)
        solver.factorize()
        x = solver.logdet()
        # Verify the solution is correct
        assert np.allclose(np.linalg.slogdet(A.toarray())[1], x)

    @pytest.mark.skip()
    def test_ld_cudss(self, matrix_factory):
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
        x = solver.logdet()
        print(type(x))
        # Verify the solution is correct
        assert np.allclose(np.linalg.slogdet(A.toarray())[1], x)