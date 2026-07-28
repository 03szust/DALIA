
from dalia.backend.config import mpi_version
import pytest
from pathlib import Path

class TestMPI:

    @pytest.mark.skipif(mpi_version is None, reason="No mpi4py installed")
    def test_send_recv(self, run_script):
        run_script("mpi", "helper_send_recv.py")

    @pytest.mark.skipif(mpi_version is None, reason="No mpi4py installed")
    def test_bcast(self, run_script):
        run_script("mpi", "helper_bcast.py")

    @pytest.mark.skipif(mpi_version is None, reason="No mpi4py installed")
    def test_reduce(self, run_script):
        run_script("mpi", "helper_reduce.py")

    @pytest.mark.skipif(mpi_version is None, reason="No mpi4py installed")
    def test_allreduce(self, run_script):
        run_script("mpi", "helper_allreduce.py")

    @pytest.mark.skipif(mpi_version is None, reason="No mpi4py installed")
    def test_reduce_scatter(self, run_script):
        run_script("mpi", "helper_reduce_scatter.py")

    @pytest.mark.skipif(mpi_version is None, reason="No mpi4py installed")
    def test_gather(self, run_script):
        run_script("mpi", "helper_gather.py")

    @pytest.mark.skipif(mpi_version is None, reason="No mpi4py installed")
    def test_allgather(self, run_script):
        run_script("mpi", "helper_allgather.py")

    @pytest.mark.skipif(mpi_version is None, reason="No mpi4py installed")
    def test_allgatherv(self, run_script):
        run_script("mpi", "helper_allgatherv.py")
    
    