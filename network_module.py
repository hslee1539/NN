import tensor
import NN as nn

class Network(nn.layer.Network):
    """NN.layer.Network에 추가 기능을 더한 Network입니다."""
    def __init__(self, *input):
        """입력 뉴런 구조를 정의하여 인공신경망 네트워크를 만듭니다."""
        nn.layer.Network.__init__(self)
        self.input = list(input)

    def append_affine(self, *output):
        """출력 뉴런 구조를 정의하여 affine레이어를 뒤에 추가합니다."""
        tmp = self.input.pop()
        while(len(self.input) != 0):
            tmp *= self.input.pop()
        self.input.append(tmp)
        self.input.extend(output)
        w = tensor.create_gauss(self.input)
        self.input = list(output)
        b = tensor.create_zeros(self.input)
        self.input = self.input.copy()
        return self.append(nn.layer.Affine(w,b))
    
    def append_batchnormalization(self):
        """뉴런들 값을 정규화하는 batch normalization 레이어를 뒤에 추가합니다. 정규화만 하고, shift 과정은 하지 않습니다."""
        return self.append(nn.layer.Batchnormalization())
    
    def append_convolution(self, stride, pad, padding, *output):
        """stride(filter 이동 간격 크기), pad(빈 공간 크기), padding(빈 공간 값)과 출력 뉴런 구조를 정의하여 conv3d 레이어를 뒤에 추가합니다."""
        input_shape = [1,1,1]
        for c in range(len(self.input)):
            input_shape[-1 -c] = self.input[-1 -c]
        output_shape = [1,1,1]
        for c in range(len(output)):
            output_shape[-1 -c] = output[-1 -c]
        filter = tensor.create_gauss([output_shape[0], input_shape[0], input_shape[1] + 2 * pad + stride - stride * output_shape[1], input_shape[2] + 2 * pad + stride - stride * output_shape[2]])
        bias = tensor.create_gauss([output_shape[0],1,1])
        self.input = list(output)
        return self.append(nn.layer.Conv3d(filter, bias, stride, pad, padding))
    
    def append_cross_entropy(self):
        return self.append(nn.layer.CrossEntropy())
    
    def append_relu(self):
        return self.append(nn.layer.Relu())
    
    def append_shift(self):
        """가중치와 편향으로 이동시키는 레이어를 추가합니다."""
        w = tensor.create_ones(self.input)
        b = tensor.create_zeros(self.input)
        return self.append(nn.layer.Shift(w,b))
    
    def append_sigmoid(self):
        return self.append(nn.layer.Sigmoid())
    
    def append_softmax(self):
        """cross entropy를 지원하는 softmax 레이어를 뒤에 추가합니다."""
        return self.append(nn.layer.Softmax())
