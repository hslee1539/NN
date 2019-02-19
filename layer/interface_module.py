
class Backwardable:
    """역전파 인터페이스입니다."""
    def backward(self, dx):
        raise NotImplementedError

    def backward_line(self,dx):
        raise NotImplementedError

class BackwardStartable:
    """역전파를 시작하는 인터페이스입니다."""
    def startBackward(self, t):
        raise NotImplementedError
    
    def startBackward_line(self, t):
        raise NotImplementedError

class ForwardStartable:
    """순전파를 시작하는 인터페이스입니다."""
    def startForward(self, x):
        raise NotImplementedError
    def startForward_line(self, x):
        raise NotImplementedError

class Forwardable:
    """순전파 인터페이스입니다."""
    def forward(self, x):
        raise NotImplementedError

    def forward_line(self, x):
        raise NotImplementedError
    
class Initable:
    def init(self):
        raise NotImplementedError

class Updatable:
    def update(self, optimizer):
        raise NotImplementedError

class Learnable:
    def learn(self, t):
        raise NotImplementedError


