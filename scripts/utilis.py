#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""

import numpy as np
#--------------
# auxiliary function
def KLD(p,q):
    p = p.ravel()
    q = q.ravel()
    n = len(p)
    s = 0
    for i in range(n):
        s = s + p[i]*np.log(p[i]/q[i])
    return s
#--------------