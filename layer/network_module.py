from . import interface_module
import tensor




class Network(interface_module.Updatable, interface_module.PartialUpdatable):
    """하나 이상의 레이어들 묶은 네트워크 레이어 입니다."""
    def __init__(self):
        # 사용된 모든 레이어
        self.layers = []
        self.last_layer = None
        self.backwardable_layers = []
        # 옵티마이저에 의한 업데이트가 필요한 레이어들
        # 역순으로 함.
        self.updatable_layers = []

        self.max_index = 0
        self.batch_size = 0

    def append(self, layer):
        self.layers.append(layer)
        for base in type(layer).__bases__:
            if((base == interface_module.Updatable)):
                self.updatable_layers.insert(0, layer)
            if((base == interface_module.Backwardable) | (base == interface_module.PartialBackwardable) | (base == interface_module.PartialUpdatable)):
                if(self.last_layer != None):
                    self.backwardable_layers.insert(0, self.last_layer)
                self.last_layer = layer
        return self

    def append_list(self, *layers):
        for layer in layers:
            self.append(layer)
        return self

    def setX(self, x):
        self.layers[0].setX(x)
        return self
    
    def setT(self, t):
        self.last_layer.setT(t)
        return self

    def initForward(self, x):
        for layer in self.layers:
            x = layer.initForward(x)
        return x

    def forward(self):
        for layer in self.layers:
            out = layer.forward()
        return out
    
    def initBackward(self, dout, t = None):
        dout = self.last_layer.initBackward(dout, t)
        for layer in self.backwardable_layers:
            dout = layer.initBackward(dout, None)
        return dout

    def backward(self):
        dx = self.last_layer.backward()
        for layer in self.backwardable_layers:
            dx = layer.backward()
        return dx
    
    def initPartial(self, max_index):
        self.max_index = max_index
        self.process_state_map = [0] * max_index
        for layer in self.layers:
            layer.initPartial(max_index)

    def partialForward(self, index):
        for i, layer in enumerate(self.layers):
            for left_value in layer.partialForward(index):
                yield i + left_value
     
    def partialBackward(self, index): 
        for subLayer in self.last_layer.partialBackward(index):
            yield subLayer + len(self.backwardable_layers)
        for i, layer in enumerate(self.backwardable_layers):
            for left_value in layer.partialBackward(index):
                yield i + left_value

    def update(self, optimizer):
        for layer in self.updatable_layers:
            layer.update(optimizer)
        return None

    def partialUpdate(self, optimizer, index):
        for i, layer in enumerate(self.updatable_layers):
            for left_value in layer.partialUpdate(optimizer, index):
                yield i + left_value


            

    
