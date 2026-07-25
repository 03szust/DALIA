
import numpy as np

from dalia.backend.config import  cupy_version, mpi_version, nccl_version, mpi_cuda_aware

if cupy_version is not None:
    import cupy as cp

if mpi_version is not None:
    from mpi4py import MPI

if nccl_version is not None:
    import nccl
    import cupy.cuda.nccl as cupy_nccl

# TODO: are these two functions really needed?
def get_host(matrix):
    matrix.to_host()
    return matrix

def get_accelerator(matrix):
    matrix.to_accelerator()
    return matrix

def get_active_comm(
        comm,
        n_parallelizable_evaluations: int,
        tag: str,
    ):
        """Return a CommunicatorType made out of all the processes that can be active
        given the number of parallelizable evaluations."""
        if comm is not None:
            rank = comm.Get_rank()
            size = comm.size
            group_size = size // n_parallelizable_evaluations

            if size > n_parallelizable_evaluations and rank >= (
                group_size * n_parallelizable_evaluations
            ):
                # Remainder processes are excluded because they cannot be assigned to any group
                print(
                    f"Rank: {rank} won't party tonight at '{tag}' level because you need a multiple of {n_parallelizable_evaluations} processes in the calling comm_group"
                )
                color = MPI.UNDEFINED
            else:
                color = 0

            active_comm = comm.Split(color, rank)
        else:
            active_comm = comm

        if color == MPI.UNDEFINED:
            exit()

        return active_comm