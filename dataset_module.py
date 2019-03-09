import tensor
import NN as nn

def copy_row(source_array, point_array, point_start, out_array):
    col = len(out_array) // len(point_array)
    for out_index in range(len(out_array)):
        out_array[out_index] = source_array[point_array[point_start + out_index // col] * col + (out_index % col)]

def partialCopy_row(source_array, point_array, point_start, out_array, index, max_index):
    col = len(out_array) // len(point_array)
    for out_index in range(index * len(out_array) // max_index, (index + 1) * len(out_array) // max_index):
        out_array[out_index] = source_array[point_array[point_start + out_index // col] * col + (out_index % col)]

class Dataset:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self.x_batch = None
        self.y_batch = None
        self.index_all = tensor.create_arange(0, x.shape[0])
        self.batch_size = 0

    def shuffle(self, parameter_list):
        """데이터 인덱스를 섞습니다."""
        tensor.set_shuffle(self.index_all)
        return self
    
    def init(self, batch_size):
        x_batch_shape = self._x.shape.copy()
        y_batch_shape = self._y.shape.copy()
        x_batch_shape[0] = batch_size
        y_batch_shape[0] = batch_size
        self.x_batch = tensor.create_zeros(x_batch_shape)
        self.y_batch = tensor.create_zeros(y_batch_shape)
        self.batch_size = batch_size
        self.batch_count = self._x.shape[0] // self.batch_size
        return self
    
    def normalRange(self):
        """데이터를 batch 크기 만큼 분할하여 진행 수와 이 객체의 x_batch와 y_batch를 갱신해서 반환합니다. 이 함수는 제너레이터입니다."""
        for batch_index in range(self._x.shape[0] // self.batch_size):
            copy_row(self._x.array, self.index_all.array, batch_index * self.batch_size, self.x_batch.array)
            copy_row(self._y.array, self.index_all.array, batch_index * self.batch_size, self.y_batch.array)
            yield (batch_index, self.x_batch, self.y_batch)
    
    def partialRange(self, index, max_index):
        """이 클래스의 normalRange함수에서 분할 계산을 하여 갱신합니다."""
        count_max = self.batch_count - 1
        for batch_index in range(count_max + 1):
            partialCopy_row(self._x.array, self.index_all.array, batch_index * self.batch_size, self.x_batch.array, index, max_index)
            partialCopy_row(self._y.array, self.index_all.array, batch_index * self.batch_size, self.y_batch.array, index, max_index)
            yield count_max - batch_index
    


