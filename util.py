import numpy as np
import itertools
from casadi import *
import casadi as cs
import random
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
π = np.pi


def plot_solve(X, J, x_goal, x_dims=None, color_agents=False, n_d=2, ax=None):
    """Plot the resultant trajectory on plt.gcf()"""

    if n_d not in (2, 3):
        raise ValueError()

    if not x_dims:
        x_dims = [X.shape[1]]

    if not ax:
        if n_d == 2:
            ax = plt.gca()
        else:
            ax = plt.gcf().add_subplot(projection="3d")

    N = X.shape[0]
    n = np.arange(N)
    cm = plt.cm.Set2
    # cm = plt.cm.viridis 

    X_split = split_agents(X, x_dims)
    x_goal_split = split_agents(x_goal.reshape(1, -1), x_dims)

    for i, (Xi, xg) in enumerate(zip(X_split, x_goal_split)):
        c = n
        if n_d == 2:
            if color_agents:
                c = cm.colors[i]
                ax.plot(Xi[:, 0], Xi[:, 1], c=c, lw=5)
            else:
                ax.scatter(Xi[:, 0], Xi[:, 1], c=c)
            ax.scatter(Xi[0, 0], Xi[0, 1], 80, "g", "d", label="$x_0$")
            ax.scatter(xg[0, 0], xg[0, 1], 80, "r", "x", label="$x_f$")
        else:
            if color_agents:
                # c = [cm.colors[i]] * Xi.shape[0]
                c = cm.colors[i]
            ax.plot(Xi[:, 0], Xi[:, 1], Xi[:, 2], c=c, lw=2)
            # ax.scatter(Xi[:, 0], Xi[:, 1], Xi[:, 2], c=c, lw=2)
            ax.scatter(
                Xi[0, 0], Xi[0, 1], Xi[0, 2], 
                s=50, c="w", marker="d", edgecolors="k", label="$x_0$")
            ax.scatter(
                xg[0, 0], xg[0, 1], xg[0, 2], 
                s=50, c="k", marker="x", label="$x_f$")
            ax.scatter(
                Xi[-1, 0], Xi[-1, 1], Xi[-1,2], 
                s=50, color=c, marker="o", edgecolors="k")
            
    
    plt.margins(0.1)
    plt.title(f"Final Cost: {J:.3g}")
    plt.draw()

def randomize_locs(n_pts, random=False, rel_dist=3.0, var=3.0, n_d=2):
    """Uniformly randomize locations of points in N-D while enforcing
    a minimum separation between them.
    """

    # Distance to move away from center if we're too close.
    Δ = 0.1 * n_pts
    x = var * np.random.uniform(-1, 1, (n_pts, n_d))

    if random:
        return x
    
    # Determine the pair-wise indicies for an arbitrary number of agents.
    pair_inds = np.array(list(itertools.combinations(range(n_pts), 2)))
    move_inds = np.arange(n_pts)

    # Keep moving points away from center until we satisfy radius
    while move_inds.size:
        center = np.mean(x, axis=0)
        distances = compute_pairwise_distance(x.flatten(), [n_d] * n_pts).T

        move_inds = pair_inds[distances.flatten() <= rel_dist]
        x[move_inds] += Δ * (x[move_inds] - center)

    return x


def face_goal(x0, xf):
    """Make the agents face the direction of their goal with a little noise"""

    VAR = 0.01
    dX = xf[:, :2] - x0[:, :2]
    headings = np.arctan2(*np.rot90(dX, 1))

    x0[:, -1] = headings + VAR * np.random.randn(x0.shape[0])
    xf[:, -1] = headings + VAR * np.random.randn(x0.shape[0])

    return x0, xf


def random_setup(
    n_agents, n_states, is_rotation=False, n_d=2, energy=None, do_face=False, **kwargs
):
    """Create a randomized set up of initial and final positions"""

    # We don't have to normlize for energy here
    x_i = randomize_locs(n_agents, n_d=n_d, **kwargs)

    # Rotate the initial points by some amount about the center.
    if is_rotation:
        θ = π + random.uniform(-π / 4, π / 4)
        R = Rotation.from_euler("z", θ).as_matrix()[:2, :2]
        x_f = x_i @ R - x_i.mean(axis=0)
    else:
        x_f = randomize_locs(n_agents, n_d=n_d, **kwargs)

    x0 = np.c_[x_i, np.zeros((n_agents, n_states - n_d))]
    xf = np.c_[x_f, np.zeros((n_agents, n_states - n_d))]
    
    if do_face:
        x0, xf = face_goal(x0, xf)

    x0 = x0.reshape(-1, 1)
    xf = xf.reshape(-1, 1)

    # Normalize to satisfy the desired energy of the problem.
    if energy:
        x0 = normalize_energy(x0, [n_states] * n_agents, energy, n_d)
        xf = normalize_energy(xf, [n_states] * n_agents, energy, n_d)

    return x0, xf

def setup_n_quads_V2(n_quads,r_safety):
    
    x0,v0 = set_random(n_quads,r_safety,False)
    x_0 = np.zeros((6*n_quads,))
    for agent,(pos,vel) in enumerate(zip(x0,v0)):
        x_0[agent*6:(agent+1)*6] = np.hstack((pos,vel))
    x_0 = x_0.reshape(-1,1)
    
    xf,vf = set_random(n_quads,r_safety,True)
    x_f = np.zeros((6*n_quads,))
    for agent,(pos_f,vel_f) in enumerate(zip(xf,vf)):
        x_f[agent*6:(agent+1)*6] = np.hstack((pos_f,vel_f))
    x_f = x_f.reshape(-1,1)
    return x_0, x_f

def set_random(n_quads, r_safety, target):

    for times in range(100):
        ini_x=[]
        for i in range(n_quads):
            for j in range(1000): 
                ini=np.random.rand(3)*np.array([3.5,3.5,2.5])
                RIGHT=True
            
                for k in range(len(ini_x)):
                    if target:
                        if(np.linalg.norm(ini-ini_x[k])<r_safety+0.1):
                            RIGHT=False 
                    else:
                        if(np.linalg.norm(ini-ini_x[k])<r_safety+0.05):
                            RIGHT=False 
                
                if RIGHT:
                    ini_x+=[ini]
                    break 

        if len(ini_x)==n_quads:
            print("positions retrieved")
            break 
        
        if (times+1)%10==0:
            print('Try %s times, cannot find. Keep trying.'%(times+1))

    if times == 99:
        print('please decrease the number of robots')


    ini_v=[]
    for i in range(n_quads):
        ini_v+=[np.zeros(3)] 
        
    return ini_x, ini_v


def compute_energy(x, x_dims, n_d=2):
    """Determine the sum of distances from the origin"""
    return np.linalg.norm(x[pos_mask(x_dims, n_d)].reshape(-1, n_d), axis=1).sum()


def normalize_energy(x, x_dims, energy=10.0, n_d=2):
    """Zero-center the coordinates and then ensure the sum of
    squared distances == energy
    """

    # Don't mutate x's data for this function, keep it pure.
    x = x.copy()
    n_agents = len(x_dims)
    center = x[pos_mask(x_dims, n_d)].reshape(-1, n_d).mean(0)

    x[pos_mask(x_dims, n_d)] -= np.tile(center, n_agents).reshape(-1, 1)
    x[pos_mask(x_dims, n_d)] *= energy / compute_energy(x, x_dims, n_d)
    assert x.size == sum(x_dims)

    return x


def perturb_state(x, x_dims, n_d=2, var=0.5):
    """Add a little noise to the start to knock off perfect symmetries"""

    x = x.copy()
    x[pos_mask(x_dims, n_d)] += var * np.random.randn(*x[pos_mask(x_dims, n_d)].shape)

    return x


def pos_mask(x_dims, n_d=2):
    """Return a mask that's true wherever there's a spatial position"""
    return np.array([i % x_dims[0] < n_d for i in range(sum(x_dims))])

def paper_setup_3_quads(random = False):
    
    x0 = np.array([[0.5, 1.5, 1.2, 0, 0, 0,
                    2.5, 1.5, 1.3, 0, 0, 0,
                    1.5, 1.3, 0.8, 0, 0, 0]], 
                     dtype=float).T
    xf = np.array([[2.5, 1.5, 1.5, 0, 0, 0, 
                    0.5, 1.5, 1.7, 0, 0, 0, 
                    1.5, 2.2, 1.0, 0, 0, 0]]).T
    if random == True:
        x0[pos_mask([6]*3, 3)] += 0.05*np.random.randn(9, 1)
        xf[pos_mask([6]*3, 3)] += 0.05*np.random.randn(9, 1)
    return x0, xf

def paper_setup_5_quads(random = False):
    
    x0 = np.array([[0.5, 1.5, 1.5, 0, 0, 0,
                    2.5, 1.5, 1.2, 0, 0, 0,
                    1.0, -1.3, 0.8, 0, 0, 0,
                    -2.0, 2.0, 1.9, 0, 0, 0,
                    3.0, -1.5, 1.4, 0, 0, 0]], 
                     dtype=float).T
    xf = np.array([[2.5, 1.5, 1.5, 0, 0, 0, 
                    0.5, 1.5, 1.7, 0, 0, 0, 
                    1.5, 2.2, 1.0, 0, 0, 0,
                    3.0, -1.5, 1.4, 0, 0, 0,
                    1.0, -1.3, 0.8, 0, 0, 0,
                    ]]).T
    if random == True:
        x0[pos_mask([6]*5, 3)] += 0.05*np.random.randn(15, 1)
        xf[pos_mask([6]*5, 3)] += 0.05*np.random.randn(15, 1)
    return x0, xf


def setup_n_quads(n, r_safety):
        
    right = False
    
    while not right:
        x0, xf = random_setup(n, 6, n_d=3, energy=n*2, var=n*1.5)
        # Print for debugging
        # print("x0:", x0)
        # print("xf:", xf)
        # print("min distance at x0:", compute_pairwise_distance(x0, [6] * n, 3).min())
        # print("min distance at xf:", compute_pairwise_distance(xf, [6] * n, 3).min())

        for i in range(2, len(x0), 6):
            if x0[i] <= 0.0:
                x0[i] = 2.0 + np.random.rand(1,) * 1.5

            if xf[i] <= 0.0:
                xf[i] = 1.0 + np.random.rand(1,) * 0.5
                
            
        if compute_pairwise_distance(x0, [6] * n, 3).min() > r_safety and compute_pairwise_distance(xf, [6] * n, 3).min() > r_safety:
            right = True
        else:
            print("Conditions not satisfied. Retrying...")

    return x0, xf


def objective(X, U, u_ref, xf, Q, R, Qf):
    total_stage_cost = 0
    for j in range(X.shape[1] - 1):
        for i in range(X.shape[0]):
            total_stage_cost += (X[i, j] - xf[i]) * Q[i, i] * (X[i, j] - xf[i])

    for j in range(U.shape[1]):
        for i in range(U.shape[0]):
            total_stage_cost += (U[i, j] - u_ref[i]) * R[i, i] * (U[i, j] - u_ref[i])

    # Quadratic terminal cost:
    total_terminal_cost = 0

    for i in range(X.shape[0]):
        total_terminal_cost += (X[i, -1] - xf[i]) * Qf[i, i] * (X[i, -1] - xf[i])

    return total_stage_cost + total_terminal_cost

def distance_to_goal(x,xf,n_agents,n_states):
    n_d = 3 
    return np.linalg.norm((x - xf).reshape(n_agents, n_states)[:, :n_d], axis=1)


def split_agents(Z, z_dims):
    """Partition a cartesian product state or control for individual agents"""
    return np.split(np.atleast_2d(Z), np.cumsum(z_dims[:-1]), axis=1)


def split_agents_gen(z, z_dims):
    """Generator version of ``split_agents``"""
    dim = z_dims[0]
    for i in range(len(z_dims)):
        yield z[i * dim : (i + 1) * dim]


def split_graph(Z, z_dims, graph):
    """Split up the state or control by grouping their ID's according to the graph"""
    assert len(set(z_dims)) == 1

    # Create a mapping from the graph to indicies.
    mapping = {id_: i for i, id_ in enumerate(list(graph))}

    n_z = z_dims[0]
    z_split = []
    for ids in graph.values():
        inds = [mapping[id_] for id_ in ids]
        z_split.append(
            np.concatenate([Z[:, i * n_z : (i + 1) * n_z] for i in inds], axis=1)
        )

    return z_split


def define_inter_graph_threshold(X, radius, x_dims, ids, n_dims=None):
    """Compute the interaction graph based on a simple thresholded distance
    for each pair of agents sampled over the trajectory
    """

    planning_radii = 5 * radius
    
    if n_dims:
        rel_dists = np.array([compute_pairwise_distance_nd_Sym(X, x_dims, n_dims)])
    else:    
        rel_dists = compute_pairwise_distance(X, x_dims)
    
    # print(f'determining interaction graph with the following pair-wise distance : {rel_dists}')
    # N = X.shape[0]
    # n_samples = 10
    # sample_step = max(N // n_samples, 1)
    # sample_slice = slice(0, N + 1, sample_step)

    # Put each pair of agents within each others' graphs if they are within
    # some threshold distance from each other.
    graph = {id_: [id_] for id_ in ids}
    # print(graph)
    pair_inds = np.array(list(itertools.combinations(ids, 2)))
    for i, pair in enumerate(pair_inds):
        if np.any(rel_dists[:,i] < planning_radii):
            graph[pair[0]].append(pair[1])
            graph[pair[1]].append(pair[0])

    graph = {agent_id: sorted(prob_ids) for agent_id, prob_ids in graph.items()}
    return graph

def compute_pairwise_distance_nd_Sym(X, x_dims, n_dims):
    """Analog to the above whenever some agents only use distance in the x-y plane"""
    CYLINDER_RADIUS = 0.2

    n_states = x_dims[0]
    n_agents = len(x_dims)
    distances = []
    eps = 1e-3

    for i, n_dim_i in zip(range(n_agents), n_dims):
        for j, n_dim_j in zip(range(i + 1, n_agents), n_dims[i + 1 :]):
            n_dim = min(n_dim_i, n_dim_j)

            Xi = X[i * n_states : i * n_states + n_dim, :]
            Xj = X[j * n_states : j * n_states + n_dim, :]
            dX = Xi-Xj

            if n_dim == 3:
                distances.append(sqrt(dX[0,:]**2 + dX[1,:]**2 + dX[2,:]**2+eps))
            else:
                distances.append(sqrt(dX[0,:]**2 + dX[1,:]**2 + eps)+CYLINDER_RADIUS)
    
    return distances

def compute_pairwise_distance(X, x_dims, n_d=3):
    """Compute the distance between each pair of agents"""
    assert len(set(x_dims)) == 1

    n_agents = len(x_dims)
    n_states = x_dims[0]

    if n_agents == 1:
        raise ValueError("Can't compute pairwise distance for one agent.")

    pair_inds = np.array(list(itertools.combinations(range(n_agents), 2)))
    X_agent = X.reshape(-1, n_agents, n_states).swapaxes(0, 2)
    dX = X_agent[:n_d, pair_inds[:, 0]] - X_agent[:n_d, pair_inds[:, 1]]
    
    return np.linalg.norm(dX, axis=0).T