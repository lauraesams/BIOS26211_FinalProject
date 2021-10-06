from ANNarchy import *
import numpy as np

try:
    # Python 2
    xrange
except NameError:
    # Python3
    xrange=range
    

def patch_pattern(pre, post, min_value, max_value, shiftRF, rfX, rfY, offsetX, offsetY, delay=0):
    synapses = CSR()
    
    for w_post in xrange(post.geometry[0]):
        for h_post in xrange(post.geometry[1]):
    
            lowerboundX = max(w_post * shiftRF + offsetX,0)
            upperboundX = min(w_post * shiftRF + offsetX + rfX, pre.geometry[0])
            lowerboundY = max(h_post * shiftRF + offsetY,0)
            upperboundY = min(h_post * shiftRF + offsetY + rfY, pre.geometry[1])

            for d_post in xrange(post.geometry[2]):
                post_rank = post.rank_from_coordinates((w_post,h_post,d_post))
                pre_ranks = []
                weights = []

                for w_pre in xrange(lowerboundX, upperboundX):
                    for h_pre in xrange(lowerboundY, upperboundY):
                        for d_pre in xrange(pre.geometry[2]):
                            
                            if not ( (pre == post) and (w_post == w_pre) and (h_post == h_pre) and (d_post == d_pre) ):
                                pre_ranks.append(pre.rank_from_coordinates((w_pre,h_pre,d_pre)))
                                weights.append(np.random.uniform(min_value, max_value))

                synapses.add(post_rank, pre_ranks, weights, [ delay for i in range(len(pre_ranks))])
        
    return synapses
