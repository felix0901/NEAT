from neat.graphs import feed_forward_layers
from neat.six_util import itervalues
import os
import neat
import pickle
import numpy as np



with open("my_network.data", 'rb') as fd:
    (winner, config, xor_inputs, xor_outputs) = pickle.load(fd)


winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
winner_net.my_create_net_layer(winner, config)
for xi, xo in zip(xor_inputs, xor_outputs):
    output = winner_net.activate(xi)
    print("CPU: input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

for xi, xo in zip(xor_inputs, xor_outputs):
    output = winner_net.my_activate(xi)
    print("FPGA: input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))



