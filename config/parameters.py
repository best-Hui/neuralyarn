import pandas as pd
from io import StringIO

# 读取一个包含布料纤维参数的数据，并根据用户提供的名称（name）返回相应的纤维参数
# 整个论文中使用的光纤参数。阴影参数基于[ksz*15]的匹配纤维，并以临时性设置几何参数
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
    # read_csv是panda库中的一个函数，用于从CSV（逗号分隔值）格式的文件或字符串中读取数据，并将其转换成一个pandasDataFrame
    parameters = pd.read_csv(file, index_col=0, header=0, skipinitialspace=True)

    # 检查用户是否提供了name参数（即你要查询的布料名称）
    if name is not None:
        parameters = parameters.loc[name]

    return parameters