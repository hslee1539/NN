class Forwardable:
    """순전파 인터페이스입니다."""
    def initForward(self, x):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
    
    def setX(self, x):
        raise NotImplementedError

class PartialForwardable(Forwardable):
    def initPartial(self, max_index):
        raise NotImplementedError
    
    def partialForward(self, index):
        """동기화가 필요한 시점에서 yield합니다."""
        raise NotImplementedError

class Backwardable:
    """역전파 인터페이스입니다."""
    def initBackward(self, dout, t = None):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def setT(self, t):
        raise NotImplementedError

class PartialBackwardable(Backwardable, PartialForwardable):
    def partialBackward(self, index):
        raise NotImplementedError

class Initable:
    def init(self):
        raise NotImplementedError

class Updatable:
    def update(self, optimizer):
        raise NotImplementedError

class PartialUpdatable(PartialBackwardable):
    def partialUpdate(self, optimizer, index):
        """동기화가 필요한 시점에서 yield합니다."""
        raise NotImplementedError


