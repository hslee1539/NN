def create_shape(x_shape, filter_shape, stride, pad):
    shape = [x_shape[0], filter_shape[0], (x_shape[2] + 2 * pad - filter_shape[2]) // stride + 1, (x_shape[3] + 2 * pad - filter_shape[3]) // stride + 1]
    return shape
"""
def forward_old(x_array, x_shape, filter_array, filter_shape, bias_array, stride, pad, padding, out_array, out_shape):
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

            x3 = f3 + o3 * stride - pad
            x2 = f2 + o2 * stride - pad
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
            # x의 -1차원 부분의 index가 음수라면, pad 부분임.
            # 이 index를 아주 큰 수로 right 연산하면 양수면 0 음수면 -1이 나옴.
            # 이를 +1을 하여 음수면 0 양수면 1 즉, pad 부분이면 0이 되어 이 값을 곱하면 pad 처리가 됨.
            isPass = (x3 >> 10000) + 1
            # 이 경우, 같은 원리로, -2차원의 index가 pad 영역인지 판단함.
            isPass *= (x2 >> 10000) + 1
            # 이 경우, x_shape[-1]의 값을 넘으면 반대쪽 pad 부분임.
            isPass *= 1  + (-(x3 // x_shape[3]) >> 10000)
            # 이 경우 같은 원리로, -2차원의 경우임.
            isPass *= 1  + (-(x2 // x_shape[2]) >> 10000)

            # 인덱스 오버플로 방지
            x_index *= isPass

            # x와 filter와 곱함.
            out_array[o] += x_array[x_index] * filter_array[filter_index] * isPass
            # pad 부분은 padding 값으로 채움.
            out_array[o] += padding * (1 - isPass)
        # 바이어스와 더함.
        out_array[o] += bias_array[o1]
    return None

# 생각보다 제어 해저드가 성능을 많이 차지 안함. 아래가 더 빠름
def forward_old2(x_array, x_shape, filter_array, filter_shape, bias_array, stride, pad, padding, out_array, out_shape):
    zero = type(out_array[0])(0)
    #tmp = zero
    multipler_out2 = out_shape[3] * out_shape[2]
    multipler_out1 = multipler_out2 * out_shape[1]
    multipler_filter2 = filter_shape[3] * filter_shape[2]
    multipler_filter1 = multipler_filter2 * filter_shape[1]
    multipler_x2 = x_shape[3] * x_shape[2]
    multipler_x1 = multipler_x2 * x_shape[1]

    for o0 in range(out_shape[0]):
        out_index0 = o0 * multipler_out1
        for o1 in range(out_shape[1]):
            out_index1 = out_index0 + o1 * multipler_out2
            filter_index0 = o1 * multipler_filter1
            for o2 in range(out_shape[2]):
                out_index2 = out_index1 + o2 * out_shape[3]
                for o3 in range(out_shape[3]):
                    tmp = zero
                    for f1 in range(filter_shape[1]):
                        filter_index1 = filter_index0 + f1 * multipler_filter2
                        x_index1 = o0 * multipler_x1 + f1 * multipler_x2
                        for f2 in range(filter_shape[2]):
                            filter_index2 = filter_index1 + f2 * filter_shape[3]
                            x2 = f2 + o2 * stride - pad
                            x_index2 = x_index1 + x2 * x_shape[3]

                            isPass2 = ((x2 >> 10000) + 1 ) * (1  + (-(x2 // x_shape[2]) >> 10000))
                            
                            for f3 in range(filter_shape[3]):
                                filter_index = filter_index2 + f3
                                x3 = f3 + o3 * stride - pad
                                x_index = x_index2 + x3
                                
                                isPass = ((x3 >> 10000) + 1) * isPass2 * (1  + (-(x3 // x_shape[3]) >> 10000))

                                # 인덱스 오버플로 방지
                                x_index *= isPass
                                tmp += x_array[x_index] * filter_array[filter_index] * isPass + padding * (1 - isPass)
                    out_array[out_index2 + o3] = tmp + bias_array[o1]

# 그리고 생각보다 계산한 것을 메모리에 저장하고 다시 쓰는 것 또한 비용이 많이 듬.
def forward_old3(x_array, x_shape, filter_array, filter_shape, bias_array, stride, pad, padding, out_array, out_shape):
    zero = type(out_array[0])(0)
    tmp = zero
    multipler_out2 = out_shape[3] * out_shape[2]
    multipler_out1 = multipler_out2 * out_shape[1]
    multipler_filter2 = filter_shape[3] * filter_shape[2]
    multipler_filter1 = multipler_filter2 * filter_shape[1]
    multipler_x2 = x_shape[3] * x_shape[2]
    multipler_x1 = multipler_x2 * x_shape[1]
    for o0 in range(out_shape[0]):
        out_index0 = o0 * multipler_out1
        for o1 in range(out_shape[1]):
            out_index1 = out_index0 + o1 * multipler_out2
            filter_index0 = o1 * multipler_filter1
            for o2 in range(out_shape[2]):
                out_index2 = out_index1 + o2 * out_shape[3]
                for o3 in range(out_shape[3]):
                    tmp = zero
                    for f1 in range(filter_shape[1]):
                        filter_index1 = filter_index0 + f1 * multipler_filter2
                        x_index1 = o0 * multipler_x1 + f1 * multipler_x2
                        for f2 in range(filter_shape[2]):
                            filter_index2 = filter_index1 + f2 * filter_shape[3]
                            x2 = f2 + o2 * stride - pad
                            x_index2 = x_index1 + x2 * x_shape[3]

                            isPass2 = ((x2 >> 10000) + 1 ) * (1  + (-(x2 // x_shape[2]) >> 10000))
                            
                            for f3 in range(filter_shape[3]):
                                x3 = f3 + o3 * stride - pad
                                isPass = ((x3 >> 10000) + 1) * isPass2 * (1  + (-(x3 // x_shape[3]) >> 10000))

                                tmp += x_array[(x_index2 + x3) * isPass] * filter_array[filter_index2 + f3] * isPass + padding * (1 - isPass)
                    out_array[out_index2 + o3] = tmp + bias_array[o1]


def forward_old4(x_array, x_shape, filter_array, filter_shape, bias_array, stride, pad, padding, out_array, out_shape):
    zero = type(out_array[0])(0)
    tmp = zero
    multipler_out2 = out_shape[3] * out_shape[2]
    multipler_out1 = multipler_out2 * out_shape[1]
    multipler_filter2 = filter_shape[3] * filter_shape[2]
    multipler_filter1 = multipler_filter2 * filter_shape[1]
    multipler_x2 = x_shape[3] * x_shape[2]
    multipler_x1 = multipler_x2 * x_shape[1]
    for o0 in range(out_shape[0]):
        out_index0 = o0 * multipler_out1
        for o1 in range(out_shape[1]):
            out_index1 = out_index0 + o1 * multipler_out2
            filter_index0 = o1 * multipler_filter1
            for o2 in range(out_shape[2]):
                out_index2 = out_index1 + o2 * out_shape[3]
                x2_tmp = o2 * stride - pad
                for o3 in range(out_shape[3]):
                    tmp = zero
                    x3_tmp = o3 * stride - pad
                    for f1 in range(filter_shape[1]):
                        filter_index1 = filter_index0 + f1 * multipler_filter2
                        x_index1 = o0 * multipler_x1 + f1 * multipler_x2
                        for f2 in range(filter_shape[2]):
                            filter_index2 = filter_index1 + f2 * filter_shape[3]
                            x2 = f2 + x2_tmp
                            x_index2 = x_index1 + x2 * x_shape[3]

                            isPass2 = ((x2 >> 32) + 1 ) * (1  + (-(x2 // x_shape[2]) >> 32))
                            
                            for f3 in range(filter_shape[3]):
                                x3 = f3 + x3_tmp
                                isPass = ((x3 >> 32) + 1) * isPass2 * (1  + (-(x3 // x_shape[3]) >> 32))

                                tmp += x_array[(x_index2 + x3) * isPass] * filter_array[filter_index2 + f3] * isPass + padding * (1 - isPass)
                    out_array[out_index2 + o3] = tmp + bias_array[o1]


def forward_old5(x_array, x_shape, filter_array, filter_shape, bias_array, stride, pad, padding, out_array, out_shape):
    zero = type(out_array[0])(0)
    tmp = zero
    multipler_out2 = out_shape[3] * out_shape[2]
    multipler_out1 = multipler_out2 * out_shape[1]
    multipler_filter2 = filter_shape[3] * filter_shape[2]
    multipler_filter1 = multipler_filter2 * filter_shape[1]
    multipler_x2 = x_shape[3] * x_shape[2]
    multipler_x1 = multipler_x2 * x_shape[1]
    for o0 in range(out_shape[0]):
        out_index0 = o0 * multipler_out1
        for o1 in range(out_shape[1]):
            out_index1 = out_index0 + o1 * multipler_out2
            filter_index0 = o1 * multipler_filter1
            for o2 in range(out_shape[2]):
                out_index2 = out_index1 + o2 * out_shape[3]
                x2_tmp = o2 * stride - pad
                for o3 in range(out_shape[3]):
                    tmp = zero
                    x3_tmp = o3 * stride - pad
                    for f1 in range(filter_shape[1]):
                        filter_index1 = filter_index0 + f1 * multipler_filter2
                        x_index1 = o0 * multipler_x1 + f1 * multipler_x2
                        for f2 in range(filter_shape[2]):
                            filter_index2 = filter_index1 + f2 * filter_shape[3]
                            x2 = f2 + x2_tmp
                            x_index2 = x_index1 + x2 * x_shape[3]

                            isPass2 = (-1 < x2) & (x2 < x_shape[2])
                            for f3 in range(filter_shape[3]):
                                x3 = f3 + x3_tmp
                                if(isPass2 & (-1 < x3) & (x3 < x_shape[3])):
                                    tmp += x_array[x_index2 + x3] * filter_array[filter_index2 + f3]
                                else:
                                    tmp += padding
                    out_array[out_index2 + o3] = tmp + bias_array[o1]

def forward_old6(x_array, x_shape, filter_array, filter_shape, bias_array, stride, pad, padding, out_array, out_shape):
    zero = type(out_array[0])(0)
    multipler_out2 = out_shape[3] * out_shape[2]
    multipler_out1 = multipler_out2 * out_shape[1]
    multipler_filter2 = filter_shape[3] * filter_shape[2]
    multipler_filter1 = multipler_filter2 * filter_shape[1]
    multipler_x2 = x_shape[3] * x_shape[2]
    multipler_x1 = multipler_x2 * x_shape[1]
    for o0 in range(out_shape[0]):
        out_index0 = o0 * multipler_out1
        for o1 in range(out_shape[1]):
            out_index1 = out_index0 + o1 * multipler_out2
            filter_index0 = o1 * multipler_filter1
            for o2 in range(out_shape[2]):
                out_index2 = out_index1 + o2 * out_shape[3]
                x2_tmp = o2 * stride - pad
                for o3 in range(out_shape[3]):
                    tmp = zero
                    x3_tmp = o3 * stride - pad
                    for f1 in range(filter_shape[1]):
                        filter_index1 = filter_index0 + f1 * multipler_filter2
                        x_index1 = o0 * multipler_x1 + f1 * multipler_x2
                        for f2 in range(filter_shape[2]):
                            filter_index2 = filter_index1 + f2 * filter_shape[3]
                            x2 = f2 + x2_tmp
                            x_index2 = x_index1 + x2 * x_shape[3]

                            #isPass2 = (-1 < x2) & (x2 < x_shape[2])
                            if((-1 < x2) & (x2 < x_shape[2])):
                                for f3 in range(filter_shape[3]):
                                    x3 = f3 + x3_tmp
                                    if((-1 < x3) & (x3 < x_shape[3])):
                                        tmp += x_array[x_index2 + x3] * filter_array[filter_index2 + f3]
                                    else:
                                        tmp += padding
                            else:
                                tmp += padding * filter_shape[3]
                    out_array[out_index2 + o3] = tmp + bias_array[o1]

#위에서는 for문이 여러번있어 제어 해저드 비용이 높은 반면 중복연산 수를 최소화 했는데,
# i5-7200u cpu 기준으로 아래로 4중첩 for문을 한개로 줄이는게 미세하게 빠름
# 그리고 분할 계산 구현에 매우 좋음. (shape[0] vs array라 분할 갯수와 나눌때 이쪽이 0에 더 근접함.(0에 근접할 수록 계산량이 평등하게됨.))
# 결론은 가장 많이 연산이 될 하위 for문들만 다이어트 해주면 됨.
def forward_old7(x_array, x_shape, filter_array, filter_shape, bias_array, stride, pad, padding, out_array, out_shape):
    """"""
    zero = type(out_array[0])(0)
    multipler_out2 = out_shape[3] * out_shape[2]
    multipler_out1 = multipler_out2 * out_shape[1]
    multipler_filter2 = filter_shape[3] * filter_shape[2]
    multipler_filter1 = multipler_filter2 * filter_shape[1]
    multipler_x2 = x_shape[3] * x_shape[2]
    multipler_x1 = multipler_x2 * x_shape[1]
    for out_index in range(len(out_array)):
        o0 = out_index // multipler_out1 % out_shape[0]

        filter_index0 = (out_index // multipler_out2 % out_shape[1]) * multipler_filter1
        x2_tmp = (out_index // out_shape[3] % out_shape[2]) * stride - pad
        x3_tmp = (out_index % out_shape[3]) * stride - pad
        tmp = zero
        for f1 in range(filter_shape[1]):
            filter_index1 = filter_index0 + f1 * multipler_filter2
            x_index1 = o0 * multipler_x1 + f1 * multipler_x2
            for f2 in range(filter_shape[2]):
                filter_index2 = filter_index1 + f2 * filter_shape[3]
                x2 = f2 + x2_tmp
                x_index2 = x_index1 + x2 * x_shape[3]

                #isPass2 = (-1 < x2) & (x2 < x_shape[2])
                if((-1 < x2) & (x2 < x_shape[2])):
                    for f3 in range(filter_shape[3]):
                        x3 = f3 + x3_tmp
                        if((-1 < x3) & (x3 < x_shape[3])):
                            tmp += x_array[x_index2 + x3] * filter_array[filter_index2 + f3]
                        else:
                            tmp += padding
                else:
                    tmp += padding * filter_shape[3]
        out_array[out_index] = tmp + bias_array[out_index // multipler_out2 % out_shape[1]]
"""
#range 객체가 계속 생성하는 것을 방지.
def forward(x_array, x_shape, filter_array, filter_shape, bias_array, stride, pad, padding, out_array, out_shape):
    """"""
    zero = type(out_array[0])(0)
    multipler_out2 = out_shape[3] * out_shape[2]
    multipler_out1 = multipler_out2 * out_shape[1]
    multipler_filter2 = filter_shape[3] * filter_shape[2]
    multipler_filter1 = multipler_filter2 * filter_shape[1]
    multipler_x2 = x_shape[3] * x_shape[2]
    multipler_x1 = multipler_x2 * x_shape[1]
    f1_range = range(filter_shape[1])
    f2_range = range(filter_shape[2])
    f3_range = range(filter_shape[3])
    for out_index in range(len(out_array)):
        o0 = out_index // multipler_out1 % out_shape[0]

        filter_index0 = (out_index // multipler_out2 % out_shape[1]) * multipler_filter1
        x2_tmp = (out_index // out_shape[3] % out_shape[2]) * stride - pad
        x3_tmp = (out_index % out_shape[3]) * stride - pad
        tmp = zero
        for f1 in f1_range:
            filter_index1 = filter_index0 + f1 * multipler_filter2
            x_index1 = o0 * multipler_x1 + f1 * multipler_x2
            for f2 in f2_range:
                filter_index2 = filter_index1 + f2 * filter_shape[3]
                x2 = f2 + x2_tmp
                x_index2 = x_index1 + x2 * x_shape[3]

                #isPass2 = (-1 < x2) & (x2 < x_shape[2])
                if((-1 < x2) & (x2 < x_shape[2])):
                    for f3 in f3_range:
                        x3 = f3 + x3_tmp
                        if((-1 < x3) & (x3 < x_shape[3])):
                            tmp += x_array[x_index2 + x3] * filter_array[filter_index2 + f3]
                        else:
                            tmp += padding * filter_array[filter_index2 + f3]
                else:
                    for f3 in f3_range:
                        tmp += padding * filter_array[filter_index2 + f3]
        out_array[out_index] = tmp + bias_array[out_index // multipler_out2 % out_shape[1]]


def partialForward(x_array, x_shape, filter_array, filter_shape, bias_array, stride, pad, padding, out_array, out_shape, index, max_index):
    """out_array가 max_index와 %연산 결과 0이랑 가까울수록 분배가 잘됨."""
    zero = type(out_array[0])(0)
    multipler_out2 = out_shape[3] * out_shape[2]
    multipler_out1 = multipler_out2 * out_shape[1]
    multipler_filter2 = filter_shape[3] * filter_shape[2]
    multipler_filter1 = multipler_filter2 * filter_shape[1]
    multipler_x2 = x_shape[3] * x_shape[2]
    multipler_x1 = multipler_x2 * x_shape[1]
    f1_range = range(filter_shape[1])
    f2_range = range(filter_shape[2])
    f3_range = range(filter_shape[3])
    for out_index in range(index * len(out_array) // max_index, (index + 1) * len(out_array) // max_index):
        o0 = out_index // multipler_out1 % out_shape[0]

        filter_index0 = (out_index // multipler_out2 % out_shape[1]) * multipler_filter1
        x2_tmp = (out_index // out_shape[3] % out_shape[2]) * stride - pad
        x3_tmp = (out_index % out_shape[3]) * stride - pad
        tmp = zero
        for f1 in f1_range:
            filter_index1 = filter_index0 + f1 * multipler_filter2
            x_index1 = o0 * multipler_x1 + f1 * multipler_x2
            for f2 in f2_range:
                filter_index2 = filter_index1 + f2 * filter_shape[3]
                x2 = f2 + x2_tmp
                x_index2 = x_index1 + x2 * x_shape[3]

                #isPass2 = (-1 < x2) & (x2 < x_shape[2])
                if((-1 < x2) & (x2 < x_shape[2])):
                    for f3 in f3_range:
                        x3 = f3 + x3_tmp
                        if((-1 < x3) & (x3 < x_shape[3])):
                            tmp += x_array[x_index2 + x3] * filter_array[filter_index2 + f3]
                        else:
                            tmp += padding
                else:
                    tmp += padding * filter_shape[3]
        out_array[out_index] = tmp + bias_array[out_index // multipler_out2 % out_shape[1]]

        

def partialForward_old(x_array, x_shape, filter_array, filter_shape, bias_array, stride, pad, padding, out_array, out_shape, index, max_index):
    """pad : 빈 공간, padding : 빈 공간에 대체되는 수"""
    multipler1 = 0
    multipler2 = 0
    multiplerX = 0
    out_len = len(out_array)
    start = index * out_len // max_index
    end = (index + 1) * out_len // max_index

    #반복문을 4중첩 쓰는 것 보다 풀어서 해결하는게 cpu의 제어 해저드가 들 걸림.
    for o in range(start, end):
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

            x3 = f3 + o3 * stride - pad
            x2 = f2 + o2 * stride - pad
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
            # x의 -1차원 부분의 index가 음수라면, pad 부분임.
            # 이 index를 아주 큰 수로 right 연산하면 양수면 0 음수면 -1이 나옴.
            # 이를 +1을 하여 음수면 0 양수면 1 즉, pad 부분이면 0이 되어 이 값을 곱하면 pad 처리가 됨.
            isPass = (x3 >> 10000) + 1
            # 이 경우, 같은 원리로, -2차원의 index가 pad 영역인지 판단함.
            isPass *= (x2 >> 10000) + 1
            # 이 경우, x_shape[-1]의 값을 넘으면 반대쪽 pad 부분임.
            isPass *= 1  + (-(x3 // x_shape[3]) >> 10000)
            # 이 경우 같은 원리로, -2차원의 경우임.
            isPass *= 1  + (-(x2 // x_shape[2]) >> 10000)

            # 인덱스 오버플로 방지
            x_index *= isPass

            # x와 filter와 곱함.
            out_array[o] += x_array[x_index] * filter_array[filter_index] * isPass
            # pad 부분은 padding 값으로 채움.
            out_array[o] += padding * (1 - isPass)
        # 바이어스와 더함.
        out_array[o] += bias_array[o1]
    return None

def backward_old(x_array, dout_array, dout_shape, filter_array, filter_shape, stride, pad, padding, dfilter_array, dbias_array, dx_array, dx_shape):
    multiplerO = 0
    multiplerF = 0
    multiplerDout = 0

    
    for dout_index in range(len(dx_array)):
        dx_array[dout_index] = 0
    
    for dfilter_index in range(len(dfilter_array)):
        dfilter_array[dfilter_index] = 0
    
    for dbias_index in range(len(dbias_array)):
        dbias_array[dbias_index] = 0
    
    
    for dout_index in range(len(dout_array)):
        dout3 = dout_index % dout_shape[3]
        multiplerDout = dout_shape[3]
        dout2 = dout_index // multiplerDout % dout_shape[2]
        multiplerDout *= dout_shape[2]
        dout1 = dout_index // multiplerDout % dout_shape[1]
        multiplerDout *= dout_shape[1]
        dout0 = dout_index // multiplerDout

        for f in range(len(filter_array) // filter_shape[0]):
            filter3 = f % filter_shape[3]
            multiplerF = filter_shape[3]
            filter2 = f // multiplerF % filter_shape[2]
            multiplerF *= filter_shape[2]
            filter1 = f // multiplerF % filter_shape[1]
            multiplerF *= filter_shape[1]
            filter0 = dout1

            #최종 filter index
            filter_index = f + filter0 * multiplerF

            x3 = filter3 + dout3 * stride - pad
            x2 = filter2 + dout2 * stride - pad
            x1 = filter1
            x0 = dout0

            multiplerO = dx_shape[3]
            x_index = x3
            x_index += x2 * multiplerO
            multiplerO *= dx_shape[2]
            x_index += x1 * multiplerO
            multiplerO *= dx_shape[1]
            x_index += x0 * multiplerO

            isPass = (x3 >> 10000) + 1
            isPass *= (x2 >> 10000) + 1
            isPass *= 1  + (-(x3 // dx_shape[3]) >> 10000)
            isPass *= 1  + (-(x2 // dx_shape[2]) >> 10000)

            # 인덱스 오버플로 방지
            x_index *= isPass

            dx_array[x_index] += isPass * dout_array[dout_index] * filter_array[filter_index]
            dfilter_array[filter_index] += isPass * dout_array[dout_index] * x_array[x_index]
            # ??? 아래를 막아야 됨... X가 pad영역일때, filter * padding가 되기 때문에 이 경우의 미분은 아래가 맞지만...
            # 막아야 편미분한 것과 값이 같아짐....
            # 왜일까...?
            #dfilter_array[filter_index] += (1 - isPass) * padding * dout_array[dout_index]
        dbias_array[dout1] += dout_array[dout_index]
    return None

def backward(dout_array, dout_shape, filter_array, filter_shape, stride, pad, dx_array, dx_shape):
    zero = type(dout_array[0])(0)
    multipler_dout2 = dout_shape[3] * dout_shape[2]
    multipler_dout1 = multipler_dout2 * dout_shape[1]
    multipler_filter2 = filter_shape[3] * filter_shape[2]
    multipler_filter1 = multipler_filter2 * filter_shape[1]
    multipler_dx2 = dx_shape[3] * dx_shape[2]
    multipler_dx1 = multipler_dx2 * dx_shape[1]
    dout1_range = range(dout_shape[1])
    for dx_index in range(len(dx_array)):
        dx3 = dx_index % dx_shape[3]
        dx2 = dx_index // dx_shape[3] % dx_shape[2]
        tmp = zero

        dx3pad = dx3 + pad
        dx2pad = dx2 + pad

        dout_index0 = (dx_index // multipler_dx1) * multipler_dout1
        filter_index1 = (dx_index // multipler_dx2 % dx_shape[1]) * multipler_filter2

        dout3_range = range(max((dx3pad - filter_shape[3]) // stride, -1) + 1, min(-(-(dx3pad + 1)// stride),dout_shape[3]))
        dout2_range = range(max((dx2pad - filter_shape[2]) // stride, -1) + 1, min(-(-(dx2pad + 1)// stride),dout_shape[2]))
        

        for dout1 in dout1_range:
            dout_index1 = dout_index0 + dout1 * multipler_dout2
            filter_index0 = filter_index1 + dout1 * multipler_filter1
        
            for dout2 in dout2_range:
                dout_index2 = dout_index1 + dout2 * dout_shape[3]
                filter_index2 = filter_index0 + (dx2pad - dout2 * stride) * filter_shape[3]

                for dout3 in dout3_range:
                    tmp += filter_array[filter_index2 + dx3pad - dout3 * stride] * dout_array[dout_index2 + dout3]
        
        dx_array[dx_index] = tmp

def backward_filter(dout_array, dout_shape, x_array, x_shape, stride, pad, padding, dfilter_array, dfilter_shape):
    zero = type(dfilter_array[0])(0)
    multipler_dout2 = dout_shape[3] * dout_shape[2]
    multipler_dout1 = multipler_dout2 * dout_shape[1]
    multipler_dfilter2 = dfilter_shape[3] * dfilter_shape[2]
    multipler_dfilter1 = multipler_dfilter2 * dfilter_shape[1]
    multipler_x2 = x_shape[3] * x_shape[2]
    multipler_x1 = multipler_x2 * x_shape[1]
    dout0_range = range(dout_shape[0])
    dout2_range = range(dout_shape[2])
    dout3_range = range(dout_shape[3])

    for dfilter_index in range(len(dfilter_array)):
        x_tmp2 = (dfilter_index // dfilter_shape[3] % dfilter_shape[2]) - pad
        x_tmp3 = (dfilter_index % dfilter_shape[3]) - pad

        dout_index1 = (dfilter_index // multipler_dfilter1) * multipler_dout2
        x_index1 = (dfilter_index // multipler_dfilter2 % dfilter_shape[1]) * multipler_x2

        tmp = zero

        for dout0 in dout0_range:
            dout_index0 = dout_index1 + dout0 * multipler_dout1
            x_index0 = x_index1 + dout0 * multipler_x1
            for dout2 in dout2_range:
                dout_index2 = dout_index0 + dout2 * dout_shape[3]
                x2 = x_tmp2 + dout2 * stride
                x_index2 = x_index0 + x2 * x_shape[3]
                if((-1 < x2) & (x2 < x_shape[2])):
                    for dout3 in dout3_range:
                        x3 = x_tmp3 + dout3 * stride
                        if((-1 < x3) & (x3 < x_shape[3])):
                            tmp += x_array[x_index2 + x3] * dout_array[dout_index2 + dout3]
                        else:
                            tmp += padding * dout_array[dout_index2 + dout3]
                else:
                    for dout3 in dout3_range:
                        tmp += padding * dout_array[dout_index2 + dout3]
        dfilter_array[dfilter_index] = tmp
    return None










def partialBackward_old(dout_array, dout_shape, filter_array, filter_shape, stride, pad, dx_array, dx_shape, index, max_index):
    multipler_dx = 1
    multipler_f = 1
    multipler_dout = 1
    zero = type(dx_array[0])(0)
    dx_len = len(dx_array)
    filter_len = len(filter_array)
    f_max = filter_len // filter_shape[1]

    start = index * dx_len // max_index
    end = (index + 1) * dx_len // max_index
    filter_index = 0
    dout_index = 0

    for dx_index in range(start, end):
        dx3 = dx_index % dx_shape[3]
        multipler_dx = dx_shape[3]
        dx2 = dx_index // multipler_dx % dx_shape[2]
        multipler_dx *= dx_shape[2]
        dx1 = dx_index // multipler_dx % dx_shape[1]
        multipler_dx *= dx_shape[1]
        dx0 = dx_index // multipler_dx

        tmp = zero

        for f in range(f_max):
            f3 = f % filter_shape[3]
            filter_index = f3
            multipler_f = filter_shape[3]
            f2 = f // multipler_f % filter_shape[2]
            filter_index += f2 * multipler_f
            multipler_f *= filter_shape[2]
            f1 = dx1
            filter_index += f1 * multipler_f
            f0 = f // multipler_f
            multipler_f *= filter_shape[1]
            filter_index += f0 * multipler_f

            dout3 = (dx3 - f3 + pad) // stride
            dout2 = (dx2 - f2 + pad) // stride
            dout1 = f1
            dout0 = dx0
            
            dout_index = dout3
            multipler_dout = dout_shape[3]
            dout_index += dout2 * multipler_dout
            multipler_dout *= dout_shape[2]
            dout_index += dout1 * multipler_dout
            multipler_dout *= dout_shape[1]
            dout_index += dout0 * multipler_dout

            tmp += filter_array[filter_index] * dout_index[dout_index]
        filter_array[filter_index] = tmp

