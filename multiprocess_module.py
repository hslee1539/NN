import tensor
import NN as nn
import multiprocessing

class Process:
    def __init__(self, process_num):
        self.process_num = process_num
    
    def run(self, iters):
        for () in range(self.process_num - 1):
            multiprocessing.Process(target=self._run, args=(iters,))
        self._run(iters)

    def _run(self, iters):
        returnValue = 0
        for item in iters:
            for value in item:
                returnValue = value
        return returnValue