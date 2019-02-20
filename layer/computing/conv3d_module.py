def create_shape(x_shape, filter_shape, stride, padding):
    shape = [x_shape[0], filter_shape[0], (x_shape[2] + 2 * padding - filter_shape[2]) // stride + 1, (x_shape[3] + 2 * padding - filter_shape[3]) // stride + 1]
    return shape

def forward(x_array, x_shape, filter_array, filter_shape, bias_array, stride, padding, pad, out_array, out_shape):
    """padding : 빈 공간, pad : 빈 공간에 대체되는 수"""
    multipler1 = 0
    multipler2 = 0
    multiplerX = 0

    #반복문을 4중첩 쓰는 것 보다 풀어서 해결하는게 cpu의 제어 해저드가 들 걸림.
    for o in range(len(out_array)):
        #이곳을 반복문으로 해결하면 conv3d가 아닌 conv_N_dim으로 응용 가능
        multipler1 = out_shape[3]
        o3 = o % multipler1
        o2 = o // multipler1 % out_shape[2]
        multipler1 *= out_shape[2]
        o1 = o // multipler1 % out_shape[1]
        multipler1 *= out_shape[1]
        o0 = o // multipler1 % out_shape[0]

        out_array[o] = 0

        #반복문을 3중첩 쓰는 것 보다 풀어서 해결하는게 cpu의 제어 해저드가 들 걸림.
        for f in range(len(filter_array) // filter_shape[0]):
            #똑같이 반복문으로 해결하면 conv3d가 아닌 conv_N_dim으로 응용 가능
            multipler2 = filter_shape[3]
            f3 = f % multipler2
            f2 = f // multipler2 % filter_shape[2]
            multipler2 *= filter_shape[2]
            f1 = f // multipler2 % filter_shape[1]# 필요 없는 변수
            multipler2 *= filter_shape[1]
            #마지막은 out의 진행 상황에 따라 변함
            f0 = o1 % filter_shape[0]

            #최종 filter index
            filter_index = f + f0 * multipler2

            x3 = f3 + o3 * stride - padding
            x2 = f2 + o2 * stride - padding
            x1 = f1
            x0 = o0

            multiplerX = x_shape[3]
            x_index = x3
            x_index += x2 * multiplerX
            multiplerX *= x_shape[2]
            x_index += x1 * multiplerX
            multiplerX *= x_shape[1]
            #최종 x index
            x_index += x0 * multiplerX

            # 여기에 반대쪽의 경우를 코딩 해라.
            # x의 -1차원 부분의 index가 음수라면, padding 부분임.
            # 이 index를 아주 큰 수로 right 연산하면 양수면 0 음수면 -1이 나옴.
            # 이를 +1을 하여 음수면 0 양수면 1 즉, padding 부분이면 0이 되어 이 값을 곱하면 padding 처리가 됨.
            isPass = (x3 >> 10000) + 1
            # 이 경우, 같은 원리로, -2차원의 index가 padding 영역인지 판단함.
            isPass *= (x2 >> 10000) + 1
            # 이 경우, x_shape[-1]의 값을 넘으면 반대쪽 padding 부분임.
            isPass *= 1  + (-(x3 // x_shape[3]) >> 10000)
            # 이 경우 같은 원리로, -2차원의 경우임.
            isPass *= 1  + (-(x2 // x_shape[2]) >> 10000)

            # 인덱스 오버플로 방지
            x_index *= isPass

            # x와 filter와 곱함.
            out_array[o] += x_array[x_index] * filter_array[filter_index] * isPass
            # padding 부분은 pad 값으로 채움.
            out_array[o] += pad * (1 - isPass)
        # 바이어스와 더함.
        out_array[o] += bias_array[o1]
    return None

def backward(dx_array, dx_shape ):
    pass




