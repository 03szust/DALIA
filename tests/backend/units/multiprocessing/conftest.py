

import pytest

import subprocess
import sys
from pathlib import Path

@pytest.fixture
def run_script():

    def _run_script(m_type, script, nproc=4):
            m_type = "helpers_" + m_type
            subprocess.run(
            [
                "mpiexec",
                "-n", str(nproc),
                sys.executable,
                
                str(Path(__file__).parent / m_type / script),
            ],
            check=True,
        )
            
    return _run_script