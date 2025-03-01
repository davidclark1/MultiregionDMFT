import numpy as np
import numerics
import torch

print("RUNNING SCRIPT", flush=True)

P = 5
num_runs = 5
N = 100000
#print('N =', N, flush=True)
T_sim = 400
N_x_to_save = 100

param_sets_1 = []
param_sets_2 = []


a = np.array([2.07056209, 2.01600629, 2.03914952, 2.08963573, 2.07470232])
b = np.array([2.46090888, 2.53800354, 2.49394571, 2.49587125, 2.51642394])
g = np.array([1.5       , 1.25817094, 1.23044151, 1.204867  , 1.21775453])
param_sets_1.append((a,b,g))

a = np.array([2.06497381, 1.97552974, 1.97887313, 1.95708126, 2.03461631])
b = np.array([2.40793845, 2.56979247, 2.46955172, 2.51276156, 2.49002518])
g = np.array([2.25      , 1.11759437, 1.18710331, 1.18463783, 1.24535078])
param_sets_1.append((a,b,g))

a = np.array([1.98332969, 1.99774933, 1.91455216, 2.06561083, 1.92826258])
b = np.array([2.46633011, 2.52011526, 2.45018848, 2.45768191, 2.4636397 ])
g = np.array([2.65      , 1.29168832, 1.20166158, 1.15528298, 1.22156233])
param_sets_1.append((a,b,g))

a = np.array([2.07056209, 2.01600629, 2.03914952, 2.08963573, 2.07470232])
b = np.array([2.46090888, 2.53800354, 2.49394571, 2.49587125, 2.51642394])
g = np.array([0., 0., 0., 0., 0.])
param_sets_2.append((a,b,g))

a = np.array([2.56497381, 1.97552974, 1.97887313, 1.95708126, 2.03461631])
b = np.array([1.90793845, 2.56979247, 2.46955172, 2.51276156, 2.49002518])
g = np.array([0., 0., 0., 0., 0.])
param_sets_2.append((a,b,g))

a = np.array([2.48332969, 2.49774933, 1.91455216, 2.06561083, 1.92826258])
b = np.array([1.96633011, 2.02011526, 2.45018848, 2.45768191, 2.4636397 ])
g = np.array([0., 0., 0., 0., 0.])
param_sets_2.append((a,b,g))

if __name__ == "__main__":
    N_t = int(T_sim / 0.25)
    all_C = np.zeros((len(param_sets), num_runs, N_t-100, P))
    all_S = np.zeros((len(param_sets), num_runs, N_t, P, P))
    all_X = np.zeros((len(param_sets), num_runs, N_t, P, N_x_to_save))
    for param_set_idx in range(len(param_sets)):
        a, b, g = param_sets[param_set_idx]
        #T = numerics.make_T(a,b)
        U = np.array([np.eye(P) for _ in range(P)])
        for run_idx in range(num_runs):
            print("param_set_idx, run_idx =", param_set_idx, run_idx, flush=True)

            h = a - b
            u = np.random.choice([-1.,1.], size=5)*np.sqrt(b)
            T = numerics.make_T_from_uh(u, h)

            print("Sampling...", flush=True)
            n, m = numerics.sample_nm(T=T, U=U, N=N)
            print("Done sampling!", flush=True)
            X, S = numerics.run_sim(T_sim=T_sim, T_eval=0.25, dt=0.05,
                                    n_vecs=torch.Tensor(n).to(0),
                                    m_vecs=torch.Tensor(m).to(0),
                                    g_vals=torch.Tensor(g).to(0),
                                    verbose=True, init_std=2.5)
            print('computing C', flush=True)
            print('X.shape=', X.shape, flush=True)
            B = 5000
            C_to_avg = []
            for i in range(int(N//B)):
                X_small = torch.Tensor(X[100:, :, B*i:B*(i+1)]).to(0)
                C = torch.fft.irfft((torch.abs(torch.fft.rfft(X_small, dim=0, norm='ortho'))**2).mean(-1), dim=0).cpu().numpy()
                #X_small = X_small.cpu().numpy()
                C_to_avg.append(C)
            C_to_avg = np.array(C_to_avg)
            print("C_to_avg.shape=", C_to_avg.shape, flush=True)
            C = C_to_avg.mean(0)
            #C = torch.fft.irfft((torch.abs(torch.fft.rfft(torch.Tensor(X[100:]).to(0), dim=0))**2).mean(-1), dim=0).cpu().numpy()
            print("saving stuff", flush=True)
            print('saving C', flush=True)
            all_C[param_set_idx, run_idx] = C
            print('saving S', flush=True)
            all_S[param_set_idx, run_idx] = S
            print('saving X', flush=True)
            all_X[param_set_idx, run_idx] = X[..., :N_x_to_save]
            print('saving npz', flush=True)
            np.savez("fp_results_v5/results.npz", all_C=all_C, all_S=all_S, all_X=all_X)