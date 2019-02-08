class Backwardable:
    def backward(self, dx):
        pass

    def backward_line(self):
        pass

class Forwardable:
    """summary
        순전파를 할 수 있는 레이어입니다.
        """
    def forward(self, x):
        """summary
            학습을 위한 forward를 구현합니다.
            design
            순전파를 처음할 때 사용되고, x의 shape에 대한 내부 변수 초기화를 여기서 합니다.
            그 다음에 x의 shape이 같다면 forward_line함수로 변수 초기화 없이 고속으로 하게 구현합니다.
            params x
            앞 레이어의 결과물입니다
            이 인수를 수정 없이 읽기만 하는것을 원칙으로 합니다.
            return
            결과물을 반환합니다.
            위 인수의 원칙으로 결과물이 중간에 바뀌는 일은 없습니다.
            """
        pass
    def forward_line(self):
        """summary
            고속 학습을 위한 forward를 구현합니다.
            design
            내부 변수를 초기화 할 필요가 없는 점을 이용하여 고속으로 계산하게 끔 구현합니다.
            """
        pass
    
class Initable:
    def init(self):
        pass

class Updatable:
    def update(self, optimizer):
        pass

