import fire
import os
from .attacks.nettack import NettackSession
from .attacks.metattack import MetattackSession
from .attacks.sparse_smoothing import SparseSmoothingSession

def resume_from_job(sessionClass):
    return eval(sessionClass).resume_from_job

if __name__ == "__main__":
    fire.Fire()