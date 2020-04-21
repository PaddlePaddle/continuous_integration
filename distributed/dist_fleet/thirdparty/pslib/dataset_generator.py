#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019-12-25 20:08
# @Author  : liyang109
import paddle.fluid.incubate.data_generator as dg


class MyDataset(dg.MultiSlotDataGenerator):
    """
    user defined dataset
    """

    # process each single line
    def generate_sample(self, line):
        """
        input: a single line
        output: each parsed instance
        """

        def data_iter():
            """
            the "real" parse function
            """
            tokens = line.split(',')
            output = [("click", [int(tokens[1])]), ("feature", [])]
            for token in tokens[2:]:
                output[1][1].append(int(token))
            yield output

        return data_iter


d = MyDataset()
d.run_from_stdin()
