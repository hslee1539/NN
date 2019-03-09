import sys
sys.path.append(__file__.replace('\\NN\\example\\ex2_xor2.py',''))

import NN as nn
import tensor
import time

print('start')
# 다른 프로세스가 중복 실행을 방지
if __name__ == '__main__':
    # xor 데이터
    x = tensor.Tensor([0., 0., 0., 1., 1., 0., 1., 1.,0., 0., 0., 1., 1., 0., 1., 1.], [8,2])
    y = tensor.Tensor([0.,1.,1.,0.,1.,0.,0.,1.,0.,1.,1.,0.,1.,0.,0.,1.], [8,2])
    data = nn.Dataset(x, y)


    # 입력층 뉴련이 2개인 네트워크 생성
    network = nn.Network(2)

    # 뉴런이 4개인 히든층
    network.append_affine(4).append_batchnormalization().append_shift().append_sigmoid()
    # 뉴런이 2개인 출력층
    network.append_affine(2).append_softmax()

    # 옵티마이저
    optimizer = nn.optimizer.SGD(0.01)

    # 학습을 위한 객체 생성
    learner = nn.Learner(data, network, optimizer)

    # 4개의 배치 사이즈로 내부 신경망을 초기화
    learner.init(8)

    # 처음 정확도 출력하기
    print(learner.findAccuracy())

    # 학습
    learner.learn(1000, True)

    # 결과
    print(learner.findAccuracy())



    