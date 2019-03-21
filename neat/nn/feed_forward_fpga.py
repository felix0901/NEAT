from neat.graphs import feed_forward_layers
from neat.six_util import itervalues
import numpy as np
import serial
import time
from neat.activations import sigmoid_activation
class FeedForwardNetworkFPGA(object):
    def __init__(self, serial_in_pre, serial_in_post, in_num_nodes, out_num_nodes, quantize=8):
        self.serial_in_pre = serial_in_pre
        self.serial_in_post = serial_in_post
        self.in_num_nodes = in_num_nodes
        self.out_num_nodes = out_num_nodes
        self.quantize = quantize
    def activate_cpu(self, inputs):
        #time_s1 = time.time()
        if self.in_num_nodes != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))
        quantize_1_time = 2**self.quantize
        quantize_2_time = 2 ** (2 * self.quantize)
        quantize_3_time = 2**(3*self.quantize)
        serial_in = np.int_(self.serial_in_pre + [int(n * quantize_1_time) for n in inputs] + self.serial_in_post)
        o_id = 0
        base_addr = 0
        command_layer = serial_in[base_addr]
        base_addr += 1
        command_init_total_node = serial_in[base_addr]
        base_addr += 1
        command_init_in_nodes  = serial_in[base_addr]
        base_addr += 1
        in_total_s = [0 for i in range(command_layer)]
        out_total_s = [0 for i in range(command_layer)]

        #====NON-LINEAR=====
            #===relu====aa
        relu_max_par = 16 * quantize_3_time
        relu_min_par = 0
            #===le_relu======
        le_relu_par = (2 / quantize_1_time)
        #=======================
        for i in range(command_layer):
            in_total_s[i] =  serial_in[base_addr]
            base_addr += 1
            out_total_s[i] = serial_in[base_addr]
            base_addr += 1
        resp_s = serial_in[base_addr:base_addr+command_init_total_node]
        base_addr += command_init_total_node
        bias_s = serial_in[base_addr:base_addr+command_init_total_node]
        base_addr += command_init_total_node
        V_s = np.zeros((command_init_total_node, 1)).astype(int)
        V_s[0:command_init_in_nodes] = serial_in[base_addr:base_addr+command_init_in_nodes].reshape((-1,1))
        base_addr += command_init_in_nodes
        #time_e = 0
        for layer_idx in range(command_layer):
            out_total, in_total = out_total_s[layer_idx], in_total_s[layer_idx]
            o_id = serial_in[base_addr:base_addr+out_total]
            base_addr += out_total
            i_id = serial_in[base_addr:base_addr+in_total]
            base_addr += in_total
            W_serial = serial_in[base_addr:base_addr+out_total*in_total]
            base_addr += out_total*in_total
            W = W_serial.reshape(out_total, in_total)
            I = V_s[i_id]
            resp = resp_s[o_id].reshape(out_total, -1)
            bias = bias_s[o_id].reshape(out_total, -1)
            #time_s = time.time()
            O = np.matmul(W, I)
            V = (resp * O + bias)
            #====No activation===
            #V_s[o_id] = V
            #==================
            #====relu====
            #V_s[o_id] = np.maximum(np.minimum(V, relu_max_par), relu_min_par) / quantize_2_time
            V_s[o_id] = np.maximum(V, relu_min_par) / quantize_2_time
            #time_e += time.time() - time_s
            #===le_relu======
            #V_s[o_id] = np.int_(np.maximum(np.minimum(V, relu_max_par), V * le_relu_par) / quantize_2_time)
        ret = [float(v/quantize_1_time) for v in V_s[o_id]]
        #time_e = 1000 * (time_e)
        #time_all = 1000 * (time.time() - time_s1)
        #print("Calculation time ", time_e, "msec")
        #print("Data processing time ", time_all-time_e, "msec")
        return ret





    def activate(self, inputs):
        if self.in_num_nodes != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))
        quantize_1_time = 2 ** self.quantize
        quantize_2_time = 2 ** (2 * self.quantize)
        quantize_3_time = 2 ** (3 * self.quantize)
        serial_in = self.serial_in_pre + [int(n* quantize_1_time) for n in inputs] + self.serial_in_post

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
        temp = int(len(serial_in)) & 0xffffffff
        ser.write(temp.to_bytes(length=4, byteorder='big'))
        ##=======================================
        for i in range(0, len(serial_in)):
            temp = int(serial_in[i]) & 0xffffffff
            ser.write(temp.to_bytes(length=4, byteorder='big'))

        result_serial = []
        for i in range(self.out_num_nodes):
            data = ser.readline()
            #    print("i: ",i, "data: ", bytes.decode(data))
            try:
                temp = int(data)
            except:
                print("My error: data =", data)
                continue
            result_serial.append(float(temp/(quantize_1_time)))
        return result_serial

    @staticmethod
    def create(genome, config, quantize = 8):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """
        idx = 0
        valueIDMap_neat2fpga = dict()
        valueIDMap_fpga2neat = dict()
        for o_id in config.genome_config.input_keys + config.genome_config.output_keys:
            valueIDMap_neat2fpga[o_id] = idx
            valueIDMap_fpga2neat[idx] = o_id
            idx += 1
        # Gather expressed connections.
        connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]
        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        layer = []
        for c in connections:
            for i in range(2):
                if c[i] not in valueIDMap_neat2fpga:
                    o_id = c[i]
                    valueIDMap_neat2fpga[o_id] = idx
                    valueIDMap_fpga2neat[idx] = o_id
                    idx += 1
        total_nodes = idx
        command_layer = len(layers)
        command_init_total_node = len(valueIDMap_neat2fpga)
        command_init_in_nodes = len(config.genome_config.input_keys)
        command_s = [command_layer, command_init_total_node, command_init_in_nodes]
        resp_s = [0] * total_nodes
        bias_s = [0] * total_nodes
        for idx in range(command_init_in_nodes,total_nodes, 1):
            o_id = valueIDMap_fpga2neat[idx]
            ng = genome.nodes[o_id]
            resp_s[idx] = int(ng.response * 2**quantize)
            bias_s[idx] = int(ng.bias * 2**(quantize + quantize + quantize))
        serial_in_pre = command_s
        serial_in_post = []
        for layer in layers:
            inputVectorThisLayer = []
            i_id = []
            outputVectorThisLayer = []
            o_id = []
            key_weight_dict = dict()
            for node in layer:
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        if inode not in inputVectorThisLayer:
                            inputVectorThisLayer.append(inode)
                            i_id.append(valueIDMap_neat2fpga[inode])
                        if onode not in outputVectorThisLayer:
                            outputVectorThisLayer.append(onode)
                            o_id.append(valueIDMap_neat2fpga[onode])
                        indx_i = inputVectorThisLayer.index(inode)
                        indx_o = outputVectorThisLayer.index(onode)
                        cg = genome.connections[conn_key]
                        key_weight_dict.update({(indx_i, indx_o): cg.weight})

                weight_matrix = np.zeros((len(outputVectorThisLayer), len(inputVectorThisLayer)))
            for key, val in key_weight_dict.items():
                indx_i, indx_o = key
                weight_matrix[indx_o][indx_i] =int(val * 2**quantize)
            serial_in_pre.append(len(inputVectorThisLayer))
            serial_in_pre.append(len(outputVectorThisLayer))
            serial_in_post = serial_in_post + o_id + i_id + list(np.ravel(weight_matrix))
        serial_in_pre = serial_in_pre + resp_s + bias_s
        return FeedForwardNetworkFPGA(serial_in_pre, serial_in_post, command_init_in_nodes, len(layer), quantize)


if __name__ == '__main__':
    serial_in_pre = [2, 4, 2, 2, 1, 3, 1, 0, 0, 256, 256, 0, 0, -8338623, -217866857]
    serial_in_post =[3,0,1, -888, 571,2,3,1,0,682,-247,326]
    in_num_nodes = 2
    out_num_nodes = 1
    quantize = 8
    inputs = [1,0]
    fpga_net = FeedForwardNetworkFPGA(serial_in_pre, serial_in_post, in_num_nodes, out_num_nodes, quantize=8)
    fpga_net.activate_cpu(inputs)



