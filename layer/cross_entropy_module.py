from .computing import cross_entropy_module as computing
from . import interface_module
import tensor

class CrossEntropy (interface_module.BackwardStartable):
    def __init__(self):
        self.dx = 