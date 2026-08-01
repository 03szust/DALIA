
import numpy as np

from dalia.backend.config import  cupy_version, mpi_version, nccl_version, mpi_cuda_aware

if cupy_version is not None:
    import cupy as cp

    if nccl_version is not None:
        import nccl
        import cupy.cuda.nccl as cupy_nccl

if mpi_version is not None:
    from mpi4py import MPI



def synchronize_host(comm):
    """Synchronize all host processes."""
    if mpi_version is not None:
        comm.Barrier()

def synchronize_accelerator():
    """Synchronize all accelerator processes."""
    if cupy_version is not None:
        cp.cuda.runtime.deviceSynchronize()

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

def get_mpi_operation(
        op
    ):
    if op == "sum":
        op = MPI.SUM
    elif op =="max":
        op = MPI.MAX
    elif op =="min":
        op = MPI.MIN
    elif op =="product":
        op = MPI.PROD
    elif op =="land":
        op = MPI.LAND
    elif op =="band":
        op = MPI.BAND
    elif op =="avg":
        raise NotImplementedError("average is not a mpi operation")            
    else:
        raise ValueError("unknown operation")

def get_nccl_operation(
        op
    ):
    if op == "sum":
        op = cupy_nccl.NCCL_SUM
    elif op =="max":
        op = cupy_nccl.NCCL_MAX
    elif op =="min":
        op = cupy_nccl.NCCL_MIN
    elif op =="product":
        op = cupy_nccl.NCCL_PROD
    elif op =="avg":
        op = cupy_nccl.NCCL_AVG
    elif op =="land":
        raise NotImplementedError("logical and is not a nccl operation")
    elif op =="band":
        raise NotImplementedError("bitwise and is not a nccl operation")             
    else:
        raise ValueError("unknown operation")

def get_accelerator_count():
    if cupy_version is not None:
        if cp.is_available():
            num_cuda = cp.cuda.runtime.getDeviceCount()
            accelerators = []
            for i in range(0, num_cuda):
                accelerators += [i]
            return num_cuda, accelerators
        return 0, None

def comm_decider(comm_type, obj):
    if comm_type == "all":
        return True, True
    
    elif comm_type == "mpi":
        return True, False
    
    elif comm_type == "nccl":
        return False, True
    
    elif comm_type == "auto":
        # TODO: automatic choice of best library depending on hardware location and availability
        # currently hard coded to use MPI
        return True, False

def get_nccl_dtype(dtype):
    if dtype == cp.float32:
        return cupy_nccl.NCCL_FLOAT32
    elif dtype == cp.float64:
        return cupy_nccl.NCCL_FLOAT64
    elif dtype == cp.complex64:
        return cupy_nccl.NCCL_COMPLEX64
    elif dtype == cp.complex128:
        return cupy_nccl.NCCL_COMPLEX128
    elif dtype == cp.int32:
        return cupy_nccl.NCCL_INT32
    elif dtype == cp.int64:
        return cupy_nccl.NCCL_INT64
    else:
        raise ValueError(f"Unsupported data type {dtype} for NCCL communication")