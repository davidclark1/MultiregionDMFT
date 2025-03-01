import numpy as np
from scipy.special import erf
import time
import torch
from scipy.optimize import root
from tqdm import tqdm

"""
Multi-Region Neural Network DMFT Simulation

This module implements Dynamical Mean-Field Theory (DMFT) for multi-region recurrent neural networks. 
It provides tools to simulate and analyze the dynamics of neural networks with multiple interconnected 
regions, each containing structured (low-rank) and random connectivity.

Key Functions:
-------------
run_ns_dmft_multi_pop: Runs a non-stationary (time-dependent) DMFT simulation for multiple regions.
                       This is the main simulation function that evolves the DMFT equations over time
                       to track correlation functions and inter-region currents.

run_sim: Simulates the dynamics of the actual neural network (as opposed to the mean-field description).

sample_nm: Samples the connectivity vectors that define the low-rank structure of the network.

find_fps: Runs the simplified version of the DMFT numerics when no disorder is present (don't need to integrate
          two-point functions).

Key Concepts:
------------
- Regions: The network consists of P regions (called R in the paper), each with its own connectivity.
- Currents (S): Represent the low-dimensional signal transmission between regions.
- Correlation Functions (Delta): Capture the temporal structure of high-dimensional chaotic fluctuations.
- Effective Interactions (T): Tensor encoding the geometric arrangement of connectivity patterns.
- Disorder (g_vals): Standard deviations of random connectivity that can lead to chaotic dynamics.

Example Usage:
-------------
# Import and reload the module if needed
reload(numerics)

# Define network parameters
P = 5  # Number of regions
np.random.seed(43)  # For reproducibility

# Create random interaction tensor
T = np.random.randn(P, P, P) * 3.5 / np.sqrt(P)

# Set overlap between input vectors to identity matrices
U = np.array([np.eye(P) for _ in range(P)])

# Run DMFT simulation with no disorder
Delta, S, psi, H = numerics.run_ns_dmft_multi_pop(
    Nt=2500,          # Number of time steps
    dt=0.05,          # Time step size
    U=U,              # Overlap between input vectors
    T=T,              # Interaction tensor
    g_vals=np.zeros(P)  # No disorder
)
"""

phi = lambda x: erf((np.sqrt(np.pi)/2)*x)
phi_prime = lambda x: np.exp(-(np.pi/4)*x**2)

def compute_C(Delta_11, Delta_22, Delta_12): #Delta_mn = variances
    A = (Delta_11 + (2./np.pi)) * (Delta_22 + (2./np.pi))
    in_sqrt = np.clip(A - Delta_12**2, a_min=1e-20, a_max=np.inf)
    return (2./np.pi)*np.arctan(Delta_12 / (np.sqrt(in_sqrt)))

def compute_C_simple(Delta_0, Delta):
    return compute_C(Delta_0, Delta_0, Delta)

def compute_C_prime(Delta_0, Delta):
    return 2/np.sqrt((np.pi*Delta_0 + 2)**2 - (np.pi*Delta)**2)

def compute_C_antideriv(Delta_0, Delta):
    A = (Delta_0 + (2/np.pi))**2
    t1 = np.sqrt(A - Delta**2)
    t2 = Delta*np.arctan(Delta / t1)
    stuff = t1 + t2
    return (2/np.pi)*stuff

def compute_potential(Delta, Delta_0, g, A):
    Phi = compute_C_antideriv(Delta_0, Delta)
    V = -.5*Delta**2 + (g**2)*Phi + A*np.abs(Delta)
    return V


def compute_psi(Delta): #Delta = variance
    in_sqrt = np.clip(1./((np.pi/2.)*Delta + 1), a_min=1e-20, a_max=np.inf)
    return np.sqrt(in_sqrt)

def compute_psi_inv(psi):
    return (2*(1-psi**2))/(np.pi*psi**2)

class SlimMat:
    def __init__(self, N, k, *additional_indices):
        self.N = N
        self.k = k
        self.data = np.zeros((k, N, *additional_indices))
        self.shape = (N, N) + additional_indices
    def check_access(self, i1, i2, j):
        #i2 is EXCLUSIVE
        k = self.k
        if i2 - 1 > j:
            raise ValueError("i > j (i1,i2,j={},{},{})".format(i1, i2, j))
        elif j - i1 >= k:
            raise ValueError("j - i > k (i1,i2,j={},{},{})".format(i1, i2, j))
    def __getitem__(self, index):
        i, j = index[:2]
        additional_indices = index[2:]
        N, k, data = self.N, self.k, self.data
        if type(i) == int:
            i = slice(i, i+1)
        i1, i2 = i.start, i.stop
        self.check_access(i1, i2, j)
        return data.__getitem__((slice(k-(j-i1)-1, k-(j-i2)-1), j) + additional_indices)
    def __setitem__(self, index, item):
        i, j = index[:2]
        additional_indices = index[2:]
        N, k, data = self.N, self.k, self.data
        if type(i) == int:
            i = slice(i, i+1)
        i1, i2 = i.start, i.stop
        self.check_access(i1, i2, j)
        data.__setitem__((slice(k-(j-i1)-1, k-(j-i2)-1), j) + additional_indices, item)
    def __len__(self):
        return self.N
    def diag(self):
        return self.data[-1]
    def dense(self):
        k = self.k
        V = np.zeros(self.shape)
        for j in range(self.N):
            V[max(0, j-(k-1)):j+1, j] = self[max(0, j-(k-1)):j+1, j]
        return V

def diag(X):
    if type(X) == SlimMat:
        return X.diag()
    else:
        r = np.arange(len(X))
        return X[r, r]

def run_ns_dmft_multi_pop(Nt, dt, U, T, g_vals, b=None, verbose=False, S_init=None, slim=False, disable_tqdm=False):
    print("Here")
    """
    Run the numerical simulation of dynamical mean-field theory (DMFT) for a multi-region neural network.
    
    This function implements the numerical solution of the DMFT equations for a network with multiple
    regions (called P in the code, R in the paper). The network consists of P regions, each with its
    own recurrent connectivity including both random (disorder) and structured (low-rank) components.
    Regions are connected through low-rank connectivity matrices.
    
    Parameters:
    -----------
    Nt : int
        Number of time steps in the simulation.
    dt : float
        Time step size for Euler integration.
    U : ndarray, shape (P, P, P)
        Tensor representing overlap between input vectors. U[m,n,r] = <m^{mn} m^{mr}>.
        U should be symmetric for each m (U[m,n,r] = U[m,r,n]).
    T : ndarray, shape (P, P, P)
        Tensor representing effective interactions between regions.
        T[m,n,r] = <n^{mn} m^{nr}> encodes geometric arrangement of connectivity patterns.
    g_vals : ndarray, shape (P,)
        Standard deviations of random connectivity within each region.
    b : int, optional
        Buffer size for the SlimMat implementation, representing how many recent timesteps to track.
        If None, uses Nt (full matrix).
    verbose : bool, optional
        If True, prints execution time upon completion.
    S_init : ndarray, shape (P, P), optional
        Initial values for the inter-region currents. If None, initialized with random values.
    slim : bool, optional
        If True, uses SlimMat implementation to save memory by only storing recent timesteps.
    disable_tqdm : bool, optional
        If True, disables the progress bar.
    
    Returns:
    --------
    Delta : ndarray or SlimMat
        Correlation functions of preactivations. If slim=False, shape is (Nt, Nt, P).
        Otherwise, it's a SlimMat instance that can be accessed like a regular array.
    S : ndarray, shape (Nt, P, P)
        Inter-region currents over time. S[t,m,n] represents current from region n to region m at time t.
    psi : ndarray, shape (Nt, P)
        Neuronal gain in each region over time.
    H : ndarray, shape (Nt, P, P)
        Drive to the currents in the mean-field dynamics.
    """
    P = U.shape[0]  # Number of regions (called R in the paper)
    
    # Initialize correlation functions and currents
    Delta, Gamma = [SlimMat(Nt, b+2, P) if slim else np.zeros((Nt, Nt, P)) for _ in range(2)]
    S = np.zeros((Nt, P, P))  # Currents between regions
    S[0] = np.random.rand(P,P) if S_init is None else S_init
    H = np.zeros((Nt, P, P))  # Drives to currents
    psi = np.zeros((Nt, P))   # Neuronal gains
    
    # Initial conditions
    Delta[0,0] = g_vals[0]**2
    idx = np.arange(Nt)
    
    # Set buffer size to Nt if not specified
    if b is None:
        b = Nt
        
    t1 = time.time()
    
    # Main simulation loop
    for j in tqdm(range(1, Nt), desc='running numerics', disable=disable_tqdm):
        # Current time step: t' = j*dt
        st = max(0, j-b-1)  # Starting index for buffer window
        
        # Calculate neuronal gain for current time step
        psi[j] = compute_psi(Delta[j-1,j-1])
        
        # Update drives to currents
        H[j] = np.einsum('mnr,n,nr->mn', T, psi[j-1], S[j-1], optimize='optimal')
        
        # Update currents
        S[j] = (1-dt)*S[j-1] + dt*H[j-1]
        
        # Update correlation functions with Euler integration
        Delta[st:j,j] = (1-dt)*Delta[st:j,j-1] + dt*Gamma[st:j,j-1]
        Delta[j,j] = Delta[j-1,j-1]
        
        # The following two blocks represent implementation choices in the Euler integration scheme
        # The current implementation shows slightly better agreement with simulations.
        # In particular, for limit cycles in currents, the normalized two-point functions don't exceed unity.
        # Note: This discrepancy should vanish for dt->0.
        
        # Current implementation:
        C_vals = compute_C(diag(Delta)[st+1:j+1], Delta[j,j], Delta[st+1:j+1,j])
        A_vals = np.einsum('mnr,tmn,mr->tm', U, H[st+1:j+1], H[j], optimize='optimal')
        
        # Alternative implementation:
        #C_vals = compute_C(diag(Delta)[st:j], Delta[j,j], Delta[st:j,j])
        #A_vals = np.einsum('mnr,tmn,mr->tm', U, H[st:j], H[j], optimize='optimal')
        
        # Update Gamma (auxiliary variable for integration)
        for i in range(max(1, j-b), j+1):
            Gamma[i,j] = ((1-dt)*Gamma[i-1,j]
                + dt*(g_vals[None,:]**2)*C_vals[-st+i-1]
                + dt*A_vals[-st+i-1])
                
        # Update equal-time correlations
        Delta[j,j] = (1-dt)*Delta[j-1,j] + dt*Gamma[j-1,j]

    t2 = time.time()
    if verbose:
        print('time taken: {} s'.format(np.round(t2-t1, 2)), flush=True)
        
    return Delta, S, psi, H

def check_U(U):
    #U_mnr = <m^{mn} m^{mr}>
    P = U.shape[0]
    for m in range(P):
        Um = U[m]
        if np.max(np.abs(Um - Um.T)) > 1e-8:
            raise ValueError("U[{},n,r] not symmetric".format(m))
        else:
            w = np.linalg.eigvalsh(Um)
            if np.min(w) <= 0.:
                raise ValueError("U[{},n,r] is symmetric but not PD (min eigval = {})".format(m, np.min(w)))

def make_hat_vars(T, U):
    P = T.shape[0]
    T_hat = np.einsum('nr,mns->mnrs', np.eye(P), T).reshape(P**2, P**2) #nm
    U_hat = np.einsum('mr,mns->mnrs', np.eye(P), U).reshape(P**2, P**2) #mm
    return T_hat, U_hat

def sample_nm(T, U, N):
    check_U(U)
    P = T.shape[0]
    n_vecs, m_vecs = [np.zeros((P, P, N)) for _ in range(2)]
    for n in range(P):
        t = T[:, n, :]
        u = U[n, :, :]
        cov = np.zeros((2*P, 2*P))
        cov[:P,:P] = np.eye(P)
        cov[:P,P:] = t
        cov[P:,:P] = t.T
        cov[P:,P:] = u
        while np.min(np.linalg.eigvalsh(cov)) < 0:
            cov[:P,:P] = cov[:P,:P]*1.05
        if cov[0,0] > 10:
            print('warning: n_variance > 10 (= {})'.format(np.round(cov[0,0], 2)), flush=True)
        L = np.linalg.cholesky(cov)
        vecs = np.dot(L, np.random.randn(2*P, N))
        n_vecs[:,n,:] = vecs[:P]
        m_vecs[n,:,:] = vecs[P:]
    return n_vecs, m_vecs

def construct_J(n, m, g):
    P, N = m.shape[1:]
    J = np.zeros((N*P, N*P))
    for mu in range(P):
        for nu in range(P):
            i1, i2 = mu*N, (mu+1)*N
            j1, j2 = nu*N, (nu+1)*N
            if mu == nu:
                if g is not None:
                    J[i1:i2,j1:j2] += np.random.randn(N, N)*g/np.sqrt(N)
            J[i1:i2,j1:j2] += np.outer(m[mu,nu], n[mu,nu])/N
    return J

phi_torch = lambda x: torch.erf((np.sqrt(np.pi)/2)*x)
#TODO: batched version
def run_sim(T_sim, T_eval, dt, n_vecs, m_vecs, g_vals, verbose=False, x_init=None, init_std=1., use_tqdm=False):
    #n_vecs, m_vecs = (P, P, N)
    device = n_vecs.device
    P, N = n_vecs.shape[1:]
    eval_iter = int(T_eval / dt)
    Nt = int(T_sim / dt)
    N_save = int(T_sim / T_eval)
    #define stuff
    x_save = np.zeros((N_save, P, N))
    #x = init_std*torch.randn(P, N, device=device) if type(x_init) == type(None) else torch.Tensor(x_init).to(device)
    x = init_std*(m_vecs*torch.randn(P, device=device)[None,:,None]).sum(dim=1)/np.sqrt(P) if type(x_init) == type(None) else torch.Tensor(x_init).to(device)
    x_save[0] = x.cpu().numpy()
    S_save = np.zeros((N_save, P, P))
    proj_filt = None #torch.zeros(P, P, device=device)
    disorder = not np.all(g_vals.cpu().numpy() == 0.)
    print('disorder?', flush=True)
    if disorder:
        print('ahh disorder!', flush=True)
        perms = [torch.Tensor(np.random.permutation(N)).type(torch.long) for _ in range(P)]
        Chi = torch.randn(N, N, device=device)/np.sqrt(N)
        #Chi = torch.randn(P, N, N, device=device)*g_vals[:,None,None]/np.sqrt(N)
    #run
    if use_tqdm:
        import tqdm
        enumerator = tqdm(range(1, Nt), desc='running network')
    else:
        enumerator = range(1, Nt)
    print('starting loop', flush=True)
    for i in enumerator:
        if i % 100 == 0:
            print(i, Nt, flush=True)
        #print("a", flush=True)
        r = phi_torch(x)
        #print("b", flush=True)
        proj = (1./N)*torch.einsum('mni,ni->mn', n_vecs, r)
        #print("c", flush=True)
        if i == 1:
            #print("d", flush=True)
            proj_filt = proj.clone()
            #print("e", flush=True)
            S_save[0] = proj_filt.cpu().numpy()
            #print("f", flush=True)
        lr_input = torch.einsum('mni,mn->mi', m_vecs, proj)
        #print("g", flush=True)
        if disorder:
            random_input = torch.einsum('ij,mj->mi', Chi,
                torch.cat([r[ii][perms[ii]][None,:] for ii in range(P)], dim=0)*g_vals[:, None])
        else:
            random_input = 0.
        #print("h", flush=True)
        tot_input = lr_input + random_input
        #print("i", flush=True)
        x += dt*(-x + tot_input)
        #print("j", flush=True)
        proj_filt += dt*(-proj_filt + proj)
        #print("k", flush=True)
        if i % eval_iter == 0:
            #print("l", flush=True)
            x_save[i//eval_iter] = x.cpu().numpy()
            #print("m", flush=True)
            S_save[i//eval_iter] = proj_filt.cpu().numpy()
            #print("n", flush=True)
        #m_vecs = m_vecs.cpu().numpy()
        #n_vecs = n_vecs.cpu().numpy()
    return x_save, S_save

def find_fps(T, T_run, N_batch, dt=0.1, init_size=1000, return_full=False):
    P = T.shape[0]
    Nt = int(T_run / dt)
    S_series = np.zeros((Nt, N_batch, P, P))
    S_series[0] = np.random.uniform(-init_size, init_size, size=(N_batch, P, P,))
    for i in range(1, Nt):
        psi = compute_psi((S_series[i-1]**2).sum(axis=-1))
        inp = np.einsum('mnr,bn,bnr->bmn', T, psi, S_series[i-1], optimize='optimal')
        S_series[i] = S_series[i-1] + dt*(-S_series[i-1] + inp)
    S_fps = S_series[-1].copy()
    print(np.max(np.abs(S_series[-1] - S_series[-200])), flush=True)
    if return_full:
        return S_series
    else:
        return S_fps
    
def run_ns_dmft(Nt, dt, g, b=None, verbose=False, Delta_init=None):
    Delta = np.zeros((Nt, Nt))
    Gamma = np.zeros((Nt, Nt))
    Delta[0,0] = 1. if Delta_init is None else Delta_init
    if b is None:
        b = Nt
    t1 = time.time()
    for j in range(1, Nt): #t' = i*dt
        Delta[:j,j] = (1-dt)*Delta[:j,j-1] + dt*Gamma[:j,j-1]
        Delta[j,j] = Gamma[j-1,j-1]
        C_vals = compute_C(np.diag(Delta)[:j], Delta[j,j], Delta[:j,j])
        for i in range(max(1, j-b), j+1): #t = i*dt
            Gamma[i,j] = (1-dt)*Gamma[i-1,j] + dt*(g**2)*C_vals[i-1]
    t2 = time.time()
    if verbose:
        print('time taken: {} s'.format(np.round(t2-t1, 2)), flush=True)
    return Delta


C_fn = compute_C_simple
Phi_fn = compute_C_antideriv

def compute_A(Delta_inf, Delta_0, g):
    A = Delta_inf - (g**2)*C_fn(Delta_0, Delta_inf)
    return A

def compute_V(Delta_inf, Delta_0, Delta, g, A=None):
    if A is None:
        A = compute_A(Delta_inf, Delta_0, g)
    t1 = -0.5*(Delta**2 - Delta_0**2)
    t2 = (g**2)*(Phi_fn(Delta_0, Delta) - Phi_fn(Delta_0, Delta_0))
    t3 = A*(Delta - Delta_0)
    return t1+t2+t3


def compute_V_deriv(Delta_inf, Delta_0, Delta, g, A=None):
    if A is None:
        A = compute_A(Delta_inf, Delta_0, g)
    t1 = -Delta
    t2 = (g**2)*C_fn(Delta_0, Delta)
    t3 = A
    return t1+t2+t3

def compute_Delta_0_sompolinsky(g, x0=None):
    def f_opt(Delta_0):
        Delta_0 = np.abs(Delta_0)
        V = compute_V(
            Delta_inf=0.,
            Delta_0=Delta_0,
            Delta=0.,
            g=g, A=0.)
        return V
    res = root(
        fun=f_opt,
        x0=g**2 if x0 is None else x0)
    if res.success:
        Delta_0 = res.x.item().__abs__()
        return Delta_0
    else:
        return np.nan

def compute_Delta_inf(Delta_0, g, x0=None):
    def f_opt(Delta_inf):
        Delta_inf = np.abs(Delta_inf)
        V = compute_V(
            Delta_inf=Delta_inf,
            Delta_0=Delta_0,
            Delta=Delta_inf,
            g=g)
        return V
    res = root(
        fun=f_opt,
        x0=0.05 if x0 is None else x0)
    if res.success:
        Delta_inf = res.x.item().__abs__()
        return Delta_inf
    else:
        return np.nan

def make_T(a, b):
    P = len(a)
    b_sqrt = np.sqrt(b)
    c = np.outer(b_sqrt, b_sqrt) + np.diag(a - b)
    T = np.einsum('mr,mn->mnr', np.eye(P), c)
    return T

def make_T_from_uh(u, h):
    c = np.outer(u, u) + np.diag(h)
    P = len(u)
    T = np.einsum('mr,mn->mnr', np.eye(P), c)
    return T

def run_sim_classic(g, N, N_batch, T, T_burn_in, eval_int, dt, device, nonlin, mean_center):
    size_gb = np.round(((T-T_burn_in)/eval_int)*N*N_batch*32 / 8e9, 4)
    print("data size: {} GB".format(size_gb))
    J = torch.randn(N, N, device=device)*g/np.sqrt(N)
    if mean_center:
        print('mean centering')
        J -= J.mean(dim=1, keepdims=True)
    x = torch.randn(N_batch, N).to(device)*g
    X = torch.zeros(int((T-T_burn_in)/eval_int), N_batch, N)#.to(device)
    X[0] = x
    for i in range(1, int(T/dt)):
        x += dt*(-x + torch.mm(nonlin(x), J.T))
        if (i >= int(T_burn_in/dt)) and (i % int(eval_int/dt) == 0):
            X[(i//int(eval_int/dt)) - int(T_burn_in/eval_int)] = x.cpu()
    return X, J.cpu()

def draw_T(P, w_re, w_re_tol, w_im, w_im_tol):
    U = np.array([np.eye(P) for _ in range(P)])
    bad = True
    i = 0
    radius = np.sqrt(w_re**2 + w_im**2)
    while bad:
        #seed = int(np.random.rand()*(2**32 - 1))
        #np.random.seed(seed)
        T = np.random.randn(P, P, P) * radius/np.sqrt(P)
        T_hat, U_hat = make_hat_vars(T, U)
        w = np.linalg.eigvals(T_hat)
        wm = w[np.argmax(w.real)]
        if (np.abs(wm.imag)>w_im-w_im_tol
                and np.abs(wm.imag)<w_im+w_im_tol
                and wm.real>w_re-w_re_tol
                and wm.real<w_re+w_re_tol):
            bad = False
        else:
            i += 1
    return T, U, i

def get_slices(Delta, num_slices):
    r = np.arange(len(Delta))
    Delta_n = Delta / (np.sqrt(Delta[r,r][:,None,:]*Delta[r,r][None,:,:]))
    slices = []
    idxs = (np.arange(num_slices)*len(Delta)/num_slices).astype(int)
    for i in idxs:
        slic = np.array([np.roll(Delta_n[i, :, mu], -i) for mu in range(Delta_n.shape[-1])])
        slices.append(slic)
    slices = np.array(slices).T #6,2,3000
    return slices
