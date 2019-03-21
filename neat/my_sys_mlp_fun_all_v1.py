
#!/usr/bin/python
import random
import serial
import numpy as np
import pickle
import os, os.path
from  math import ceil
import scipy.io

def my_sys_mlp_fun_all(inputs, inputVectorKeyMap, layerWeightMatrices, outputVectorKeyMap, act_funcMap, biasMap, responseMap, values):
    command_layer = len(inputVectorKeyMap)
    command_init_total_node = len(values)
    command_init_in_nodes = len(inputs)
    serial_in = []

    ####To send ITER_IN_VIVADO through uart===
    command_s = [command_layer, command_init_total_node, command_init_in_nodes]
    for i in range(len(command_s)):
        serial_in.append(int(command_s[i]))
    for(act_funcV, biasV, responseV)
    for (inputnodeV, WeightsV, outputnodeV, act_funcV, biasV, responseV) in \
            zip(inputVectorKeyMap, layerWeightMatrices, outputVectorKeyMap,
                act_funcMap, biasMap, responseMap):

    cols = 8
    Yo, Xo = 1, 1

    if not (inputs_from_neat  or Weights_from_neat):
        C = 200
        K = 79
        inputs = np.array([i for i in range(C)])
        inputs = inputs.reshape((*inputs.shape, 1, 1))

        C, Y, X = inputs.shape

        Weights = np.array([[f * (-1)**i for i in range(C)] for f in range(K)])
        Weights = Weights.reshape((*Weights.shape, 1, 1))

        K, _, R, S = Weights.shape

    else:
        inputs = np.array(inputs_from_neat)
        inputs = inputs.reshape((*inputs.shape, 1, 1))
        C, Y, X = inputs.shape
        Weights = np.array(Weights_from_neat)
        Weights = np.transpose(Weights)
        Weights = Weights.reshape((*Weights.shape, 1, 1))
        K, _, R, S = Weights.shape


    ##===quantization to int==============
    quantize_val = 2**10
    inputs  = (inputs * quantize_val).astype(int)
    Weights = (Weights * quantize_val).astype(int)
    ##====================================

    #####====only need when debugging in vivado testbench====
    # inputs = np.abs(inputs)
    # Weights = np.abs(Weights)
    ######===============================================

    ## =================WINDOWS_SIZE PARAMETER=============
    WINDOWS_MAX = cols
    #======================================================
    points_o_per_iter = int(ceil(WINDOWS_MAX/cols) * cols)
    #@@
    iters = int(ceil(K/points_o_per_iter))

    cutoff = C

    ifmap = [inputs.reshape(C,)]



    fmap = [Weights[i,:,0,0].reshape(C,) for i in range(K)]


    left_in_s = []
    top_in_s = []
    serial_in = []
    for it in range(iters):
        left_in = []
        top_in = []
        left_in_temp = ifmap[0]
        left_in.append(left_in_temp)
        for dk in range(cols):
            k = it * cols + dk
            if k >= K:
                top_in_temp = np.zeros((C,))
                1+2
            else:
                top_in_temp = fmap[k]
            top_in_temp = np.concatenate((np.zeros((dk,)), top_in_temp, np.zeros((cols - dk - 1,))))
            top_in.append(top_in_temp)
        serial_in = serial_in + (np.ravel(left_in).tolist() + np.ravel(top_in).tolist())
        left_in_s.append(left_in)
        top_in_s.append(top_in)

    left_pixel_per_iter = len(np.ravel(left_in).tolist())
    top_pixel_per_iter  = len(np.ravel(top_in).tolist())
    ## ===== golden answer ========
    golden = np.matmul(Weights.reshape((K,C)), inputs.reshape(C,1))
    golden = golden.reshape(K,1,1)

    1+2

    ser = serial.Serial(
        port='/dev/tty.usbmodem14431',
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )


    if ser.isOpen():
        ser.close()
    ser.open()
    ser.isOpen()


    ####To send ITER_IN_VIVADO through uart===
    temp = int(len(serial_in)) &  0xffffffff
    ser.write(temp.to_bytes(length=4, byteorder='big'))
    temp = int(C) &  0xffffffff
    ser.write(temp.to_bytes(length=4, byteorder='big'))
    ##=======================================
    for i in range(0,len(serial_in)):
        temp = int(serial_in[i]) &  0xffffffff
        ser.write(temp.to_bytes(length=4, byteorder='big'))

    result_serial = []
    for i in range(K):
        data = ser.readline()
        #    print("i: ",i, "data: ", bytes.decode(data))
        try:
            temp = int(data)
        except:
            print("My error: data =", data)
            continue
        result_serial.append(temp)
#===========to print out the difference=========
    # diff = [1 for i in range(len(golden)) if result_serial[i] != golden[i]]
    # diff_num = len(diff)
    # print("The difference b/w golden and fpga is: ", diff_num)dddddadasddssddss
#======================================
    output_from_sys_mlp.append([r/(quantize_val**2) for r in result_serial])





if __name__ == '__main__':
    inputs_from_neat = [1, 2]
    Weights_from_neat = [[2, 3]]
    output_from_sys_mlp = []
    my_sys_mlp_fun_all()