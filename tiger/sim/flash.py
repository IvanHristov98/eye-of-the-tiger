#!/usr/bin/env python3

import nest
import numpy

import tiger.sim.sim as sim


def main():
    simulator = sim.NetRunner(3000)
    simulator.build_network()
    # simulator.init_spike_generators(retina_spikes=[])


if __name__ == "__main__":
    main()
