"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
##=========added to search for self-defined module==========
import sys
sys.path.insert(0, "../../../NEAT")
##==========================================================

import os
import neat
#import visualize
import pickle
import time



    #
# 2-input XOR inputs and expected outputs.
xor_inputs = [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0) , (1.0, 1.0)]
xor_outputs = [ (1.0,) , (0.0,),     (1.0,)     ,     (0.0,)]


def eval_genomes(genomes, config):
   
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        #net = neat.nn.FeedForwardNetwork.create(genome, config)
        net_fpga = neat.nn.FeedForwardNetworkFPGA.create(genome, config)
        #net.my_create_net_layer(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net_fpga.activate_cpu(xi)
            #output = net_fpga.activate_fpga(xi)
            #output = net.my_activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2



def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    #Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

#    Show output of the most fit genome against training data.
#     with open("my_network.data", 'wb') as fd:
#         pickle.dump((winner, config, stats), fd)
    # with open("my_network.data", 'rb') as fd:
    #     winner, config, stats = pickle.load(fd)


    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    net_fpga = neat.nn.FeedForwardNetworkFPGA.create(winner, config)
    #winner_net.my_create_net_layer(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        time_s = time.time()
        # output1 = net_fpga.activate(xi)
        output3 = net_fpga.activate_fpga(xi)
        #output3 = net_fpga.activate_cpu(xi)
        time_e = 1000 * (time.time() - time_s)
        # print("input {!r}, expected output FPGA {!r}, got {!r}".format(xi, xo, output1))
        print("input {!r}, expected output CPU {!r}, got {!r}".format(xi, xo, output3))
        #print("One inference time is ", time_e, "msec")
        # time_s = time.time()
        # output2 = winner_net.activate(xi)
        # time_e = 1000 * (time.time() - time_s)
        #print("input {!r}, expected output2 {!r}, got {!r}".format(xi, xo, output2))
        #print("One inference time is ", time_e, "msec")
        print(" ")

        #output = winner_net.my_activate(xi)


    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)