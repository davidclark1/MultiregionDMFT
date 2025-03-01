import numpy as np
import numerics
import itertools
import sys
import os
import os.path
from os.path import join

print("RUNNING SCRIPT", flush=True)

task_idx = int(sys.argv[1])

w_re_vals = np.linspace(1., 2.5, 40)
w_im_vals = np.linspace(0., 1.5, 40)
num_runs = 10
num_superruns = 5
g_vals = np.array([1.5])

param_vals = np.array(list(itertools.product(w_re_vals, np.arange(num_superruns))))
w_re, superrun_idx = param_vals[task_idx] #, w_im = param_vals[task_idx]
print('params =', param_vals[task_idx], flush=True)

P = 2
Nt = 2500
num_slices = 6

all_slices = np.zeros((len(g_vals), len(w_im_vals), num_runs, Nt, P, num_slices))
all_S = np.zeros((len(g_vals), len(w_im_vals), num_runs, Nt, P, P))

for g_idx in range(len(g_vals)):
    print('g_idx =', g_idx, flush=True)
    g = g_vals[g_idx]
    for w_im_idx in range(len(w_im_vals)):
        print('w_im_idx =', w_im_idx, flush=True)
        w_im = w_im_vals[w_im_idx]
        for run_idx in range(num_runs):
            print('run_idx =', run_idx, flush=True)
            T, U, _ = numerics.draw_T(P, w_re, 0.01875, w_im, 0.01875)
            Delta, S, psi, H = numerics.run_ns_dmft_multi_pop(
                Nt=Nt, dt=0.05, U=U, T=T, g_vals=np.ones(P)*g, verbose=False, b=None, disable_tqdm=True)
            all_slices[g_idx, w_im_idx, run_idx] = numerics.get_slices(Delta, num_slices=num_slices)
            all_S[g_idx, w_im_idx, run_idx] = S
            np.savez("lc_results_v4/results_{}.npz".format(task_idx), all_slices=all_slices, all_S=all_S,
                g_vals=g_vals, w_re_vals=w_re_vals, w_im_vals=w_im_vals, task_idx=task_idx, param_vals=param_vals,
                w_re=w_re, Nt=Nt, P=P)
