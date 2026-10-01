import sys
sys.path.insert(0,'.')
import numpy as np
import test_uf2 as t
try:
    sig_ft = t.calculate_spectrum(method='hermite', M=3)
    analytical_sig = np.load('fixtures/v_3LS/analytical_signal.npy')
    diff = t.L2_norm(-np.pi*sig_ft[:,0,:], analytical_sig[:,0,:])
    print(f'M=3: diff={diff:.6f}')
except Exception as e:
    print(f'M=3: FAILED {type(e).__name__}: {e}')
