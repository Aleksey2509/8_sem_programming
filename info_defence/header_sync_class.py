#! usr/bin/python


"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr


class blk(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const"""

    def __init__(self):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Header Sync Block',   # will show up in GRC
            in_sig=[np.complex64, np.complex64],
            out_sig=[np.complex64, np.uint]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).

    def work(self, input_items, output_items):
        """example: multiply with constant"""
        output_items[0][:] = input_items[0] * self.additionFlag
        return len(output_items[0])

    def find_header (self, frame, header)
        header_len = np.size(header)
        frame_len = np.size(frame)
    
        max_corr = 0
        corr = 0
        index = -1
        for i in range(0, frame_len - header_len):
            corr = correlation(frame(i : i + header_len), header)
            if corr > max_corr:
                max_corr = corr
                index = i
    
        return index
    
    
    def correlation(self, first_arr, second_arr)
        return np.multiply(first_arr, np.conjugate(second_arr)) / (np.linalg.norm(first_arr) * np.linalg.norm(second_arr) )
