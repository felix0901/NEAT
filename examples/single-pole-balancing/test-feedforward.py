"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

from __future__ import print_function

import os
import pickle

from cart_pole import CartPole, discrete_actuator_force
from movie import make_movie

import neat
from neat import nn

# load the winner
with open("my_network.data", 'rb') as fd:
    winner, config, stats = pickle.load(fd)

print('Loaded genome:')
print(winner)

#winner_net = neat.nn.FeedForwardNetworkFPGA.create(winner, config)
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
#winner_net.my_create_net_layer(winner, config)
sim = CartPole()

print()
print("Initial conditions:")
print("        x = {0:.4f}".format(sim.x))
print("    x_dot = {0:.4f}".format(sim.dx))
print("    theta = {0:.4f}".format(sim.theta))
print("theta_dot = {0:.4f}".format(sim.dtheta))
print()

# Run the given simulation for up to 120 seconds.
balance_time = 0.0
# while sim.t < 120.0:
#     inputs = sim.get_scaled_state()
#     action = winner_net.activate(inputs)
#     #action = winner_net.my_activate(inputs)
#     # Apply action to the simulated cart-pole
#     force = discrete_actuator_force(action)
#     sim.step(force)
#
#     # Stop if the network fails to keep the cart within the position or angle limits.
#     # The per-run fitness is the number of time steps the network can balance the pole
#     # without exceeding these limits.
#     if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
#         break
#
#     balance_time = sim.t
#
#
# print('Pole balanced for {0:.1f} of 120.0 seconds'.format(balance_time))
#
# print()
# print("Final conditions:")
# print("        x = {0:.4f}".format(sim.x))
# print("    x_dot = {0:.4f}".format(sim.dx))
# print("    theta = {0:.4f}".format(sim.theta))
# print("theta_dot = {0:.4f}".format(sim.dtheta))
# print()
# print("Making movie...")
make_movie(winner_net, discrete_actuator_force, 10.0, "feedforward-movie.mp4")