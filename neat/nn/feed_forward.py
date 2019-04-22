from neat.graphs import feed_forward_layers
from neat.six_util import itervalues
#from neat.my_sys_mlp_fun_v2 import my_sys_mlp_fun
import numpy as np

class FeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)
        self.inputVectorKeyMap = []
        self.outputVectorKeyMap = []
        self.layerWeightMatrices = []
        self.act_funcMap = []
        self.biasMap = []
        self.responseMap = []
        self.layers = None
        self.connections = None


    def my_activate(self, inputs):

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for (inputnodeV, WeightsV, outputnodeV, act_funcV, biasV, responseV) in \
                zip(self.inputVectorKeyMap, self.layerWeightMatrices, self.outputVectorKeyMap,
                    self.act_funcMap, self.biasMap, self.responseMap):
            sys_inputs = [self.values[i] for i in inputnodeV]
            sys_outputsV_capsule = []
            my_sys_mlp_fun(sys_outputsV_capsule, sys_inputs, WeightsV)
            sys_outputsV = sys_outputsV_capsule[0]
            for o, o_val, act_func, bias, response in zip(outputnodeV, sys_outputsV, act_funcV, biasV, responseV):
                self.values[o] = act_func(bias + response * o_val)
        return [self.values[i] for i in self.output_nodes]

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            inputs = []
            Weights = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)

            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)
        ret = [self.values[i] for i in self.output_nodes]
        return ret

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]
        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = [] # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))


                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)


    def my_create_net_layer(self, genome, config):
        self.connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]
        self.layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, self.connections)
        for layer in self.layers:
            inputVectorThisLayer = []
            outputVectorThisLayer = []
            key_weight_dict = dict()
            act_funcThislayer = []
            biasThislayer = []
            responseThislayer = []
            for cg in itervalues(genome.connections):
                inode, onode = cg.key

                if onode in layer and cg.enabled:
                    if inode not in inputVectorThisLayer:
                        inputVectorThisLayer.append(inode)
                    if onode not in outputVectorThisLayer:
                        outputVectorThisLayer.append(onode)

                    indx_i = inputVectorThisLayer.index(inode)
                    indx_o = outputVectorThisLayer.index(onode)
                    key_weight_dict.update({(indx_i, indx_o): cg.weight})

                    ng = genome.nodes[onode]
                    act_funcThislayer.append(config.genome_config.activation_defs.get(ng.activation))
                    biasThislayer.append(ng.bias)
                    responseThislayer.append(ng.response)
            weight_matrix = np.zeros((len(inputVectorThisLayer), len(outputVectorThisLayer)))

            for key, val in key_weight_dict.items():
                indx_i, indx_o = key
                weight_matrix[indx_i][indx_o] = float(val)

            self.inputVectorKeyMap.append(inputVectorThisLayer)
            self.outputVectorKeyMap.append(outputVectorThisLayer)
            self.layerWeightMatrices.append(weight_matrix)
            self.act_funcMap.append(act_funcThislayer)
            self.biasMap.append(biasThislayer)
            self.responseMap.append(responseThislayer)





