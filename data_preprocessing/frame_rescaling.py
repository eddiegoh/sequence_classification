# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:25:30 2018

@author: Eddie
"""


def rescale(input_list, size):
    skip = len(input_list) // size
    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]
    # Cut off the last one if needed.
    return output[:size]
