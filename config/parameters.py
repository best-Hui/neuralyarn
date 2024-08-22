import pandas as pd
from io import StringIO

def get_fiber_parameters(name=None):

    text = \
"""
            N,      Rho,    Alpha,  R_r,    R_g,    R_b,    TT_r,   TT_g,   TT_b,   R_long,     TT_long,    TT_azim
fleece,     300,    0.300,  0.240,  0.040,  0.087,  0.087,  0.452,  0.725,  0.948,  7.238,      10.000,     25.989
gabardine,  450,    0.250,  0.120,  0.185,  0.047,  0.069,  0.999,  0.330,  0.354,  2.141,      10.000,     23.548
silk,       300,    0.200,  0.000,  0.745,  0.008,  0.07,   0.62,   0.553,  0.562,  1.000,      10.000,     19.823
cotton,     600,    0.350,  0.060,  0.989,  0.959,  0.874,  0.999,  0.999,  0.999,  1.000,      27.197,     38.269
polyester,  200,    0.400,  0.200,  0.700,  0.700,  0.700,  0.600,  0.000,  0.800,  5.238,      10.000,     25.000
"""

    file = StringIO(text)
    parameters = pd.read_csv(file, index_col=0, header=0, skipinitialspace=True)

    if name is not None:
        parameters = parameters.loc[name]

    return parameters