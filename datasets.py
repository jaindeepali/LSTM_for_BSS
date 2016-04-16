#!/usr/bin/env python
__author__ = 'Tom Schaul (tom@idsia.ch)'

from pybrain.datasets import SequentialDataSet
import numpy

class ParityDataSet(SequentialDataSet):
    """ Determine whether the bitstring up to the current point conains a pair number of 1s or not."""
    def __init__(self):
        SequentialDataSet.__init__(self, 1,1)

        # data = numpy.random.random(250)
        data = (list(range(1,10,1)) + list(range(10,1,-1)))*1000  

        self.newSequence()
        for i in range(1,10):
            self.addSample([data[i]], [data[i+1]])

        self.newSequence()
        for i in range(21,40):
            self.addSample([data[i]], [data[i+1]])

        self.newSequence()
        for i in range(34,56):
            self.addSample([data[i]], [data[i+1]])

        self.newSequence()
        for i in range(234,247):
            self.addSample([data[i]], [data[i+1]])

        # self.newSequence()
        # self.addSample([-1], [-1])
        # self.addSample([1], [1])
        # self.addSample([1], [-1])

        # self.newSequence()
        # self.addSample([1], [1])
        # self.addSample([1], [-1])

        # self.newSequence()
        # self.addSample([1], [1])
        # self.addSample([1], [-1])
        # self.addSample([1], [1])
        # self.addSample([1], [-1])
        # self.addSample([1], [1])
        # self.addSample([1], [-1])
        # self.addSample([1], [1])
        # self.addSample([1], [-1])
        # self.addSample([1], [1])
        # self.addSample([1], [-1])

        # self.newSequence()
        # self.addSample([1], [1])
        # self.addSample([1], [-1])
        # self.addSample([-1], [-1])
        # self.addSample([-1], [-1])
        # self.addSample([-1], [-1])
        # self.addSample([-1], [-1])
        # self.addSample([-1], [-1])
        # self.addSample([-1], [-1])
        # self.addSample([-1], [-1])
        # self.addSample([1], [1])
        # self.addSample([-1], [1])
        # self.addSample([-1], [1])
        # self.addSample([-1], [1])
        # self.addSample([-1], [1])
        # self.addSample([-1], [1])

        # self.newSequence()
        # self.addSample([-1], [-1])
        # self.addSample([-1], [-1])
        # self.addSample([1], [1])