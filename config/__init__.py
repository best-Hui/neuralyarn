MITSUBA_PATH = './dependencies/mitsuba3/build/python'
DRJIT_LIBLLVM_PATH = '/usr/lib/llvm-18/lib/libLLVM.so'
# VARIANT = 'llvm_ad_rgb'
VARIANT = 'cuda_ad_rgb'
DEVICE = 'cuda' if VARIANT == 'cuda_ad_rgb' else 'cpu'

# Aliases
variant = VARIANT
device = DEVICE

import sys
sys.path.append(MITSUBA_PATH)

import os
os.environ["DRJIT_LIBLLVM_PATH"] = DRJIT_LIBLLVM_PATH

