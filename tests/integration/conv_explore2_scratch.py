import sys
sys.path.insert(0,'.')
import numpy as np
import test_uf2 as t

analytical_sig = np.load('fixtures/v_3LS/analytical_signal.npy')

for M in [4,5,6,7,9,11,13,15,19,25,31,41,55,75]:
    try:
        sig_ft = t.calculate_spectrum(method='hermite', M=M)
        diff = t.L2_norm(-np.pi*sig_ft[:,0,:], analytical_sig[:,0,:])
        print(f'M={M:3d}: diff={diff:.6f}')
    except Exception as e:
        print(f'M={M:3d}: FAILED {type(e).__name__}: {e}')
