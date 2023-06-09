import random

import torch
import numpy as np
from scipy.integrate import solve_ivp


from utils.models import TPC
from utils.models import NeuralKalmanFilter, KalmanFilter
from utils.data import generate_random_nonlinear_data


device = "cuda" if torch.cuda.is_available() else "cpu"


def to_np(x):
    return x.cpu().detach().numpy()


def run_simulation_random_data_np(activation, timepoints):
    dt = 1 / 2
    x = np.array((1., 0.))
    C_init = np.eye(2) * 2.5
    A_init = (np.array((
        (-dt / 2, 1.),
        (-1., -dt / 2)))) * 2.5

    # Generate random data (add your generate_random_nonlinear_data function)
    y_truth, A_truth = generate_random_nonlinear_data(2, 2, timepoints, x, C_init, A_init,
                                                      seed=random.randint(0, 200))

    # Infer the data with temporal predictive coding (add your TPC class)
    A = np.zeros((y_truth.shape[0], y_truth.shape[0]))
    C = np.eye(y_truth.shape[0])
    tpc = TPC(y_truth, A, C, activation=activation)
    tpc.forward()
    error = tpc.get_error()
    predicted = tpc.get_predictions()

    solution = np.zeros((y_truth.T.shape[0], 2))
    solution[:, 0] = y_truth.T[:, 0]
    solution[:, 1] = predicted.T[:, 0]

    return solution, error


def run_tracking_inf_steps(step):
    # hyper parameters
    seq_len = 1000
    inf_iters = 20
    inf_lr = 0.1
    learn_lr = 1e-5

    # create the dataset of the tracking problem
    # transition matrix A
    dt = 1e-3
    A = torch.tensor([[1., dt, 0.5 * dt ** 2],
                      [0., 1., dt],
                      [0., 0., 1.]]).to(device)

    # random emissin matrix C
    g_C = torch.Generator()
    g_C.manual_seed(200)
    C = torch.randn((3, 3), generator=g_C).to(device)
    print(C)

    # control input matrix B
    B = torch.tensor([0., 0., 1.]).to(device).reshape((3, 1))

    # initial true dynamics
    z = torch.tensor([0., 0., 0.]).to(device).reshape((3, 1))

    # control input
    def u_fun(t):
        return torch.tensor(np.exp(-0.01 * t)).reshape((1, 1)).to(device, torch.float)

    # noise covariances in KF
    Q = torch.eye(3).to(device)
    R = torch.eye(3).to(device)

    # generating dataset
    g_noise = torch.Generator()
    g_noise.manual_seed(10)
    us = []
    zs = []
    xs = []
    for i in range(seq_len):
        u = u_fun(i)
        z = torch.matmul(A, z) + torch.matmul(B, u) + torch.randn((3, 1), generator=g_noise).to(device)
        x = torch.matmul(C, z) + torch.randn((3, 1), generator=g_noise).to(device)
        us.append(u)
        zs.append(z)
        xs.append(x)

    us = torch.cat(us, dim=1)
    zs = torch.cat(zs, dim=1)
    xs = torch.cat(xs, dim=1)

    # estimating using NKF, 5 gradient steps
    nkf = NeuralKalmanFilter(A, B, C, latent_size=3, dynamic_inf=False)
    zs_nkf, _ = nkf.predict(xs, us, inf_iters=step, inf_lr=inf_lr)
    print(zs_nkf.shape)

    zs = to_np(zs)
    zs_nkf = to_np(zs_nkf)

    kf = KalmanFilter(A, B, C, Q, R, latent_size=3)
    zs_kf, _ = kf.inference(xs, us)
    zs_kf = to_np(zs_kf)

    return zs, zs_nkf, zs_kf


def run_tracking_learning(input_weights):
    # hyper parameters
    seq_len = 100
    inf_iters = 20
    inf_lr = 0.05
    learn_iters = 80
    learn_lr = 2e-5
    seeds = range(20)

    for ind, seed in enumerate(seeds):
        # create the dataset of the tracking problem
        # transition matrix A
        dt = 1e-3
        A = torch.tensor([[1., dt, 0.5 * dt ** 2],
                          [0., 1., dt],
                          [0., 0., 1.]]).to(device)

        # random emissin matrix C
        g_C = torch.Generator()
        g_C.manual_seed(1)
        C = torch.randn((3, 3), generator=g_C).to(device)

        # control input matrix B
        B = torch.tensor([0., 0., 1.]).to(device).reshape((3, 1))
        # initial true dynamics
        z = torch.tensor([0., 0., 0.]).to(device).reshape((3, 1))

        # control input
        def u_fun(t):
            return torch.tensor(np.exp(-0.01 * t)).reshape((1, 1)).to(device, torch.float)

        # noise covariances in KF
        Q = torch.eye(3).to(device)
        R = torch.eye(3).to(device)

        # generating dataset
        g_noise = torch.Generator()
        g_noise.manual_seed(seed)
        us = []
        zs = []
        xs = []
        for i in range(seq_len):
            u = u_fun(i)
            z = torch.matmul(A, z) + torch.matmul(B, u) + torch.randn((3, 1), generator=g_noise).to(device)
            x = torch.matmul(C, z) + torch.randn((3, 1), generator=g_noise).to(device)
            us.append(u)
            zs.append(z)
            xs.append(x)

        us = torch.cat(us, dim=1)
        zs = torch.cat(zs, dim=1)
        xs = torch.cat(xs, dim=1)

        # generate random A and C for initial weights
        g_A = torch.Generator()
        g_A.manual_seed(1)
        init_A = torch.randn((3, 3), generator=g_A).to(device)

        g_C = torch.Generator()
        g_C.manual_seed(2)
        init_C = torch.randn((3, 3), generator=g_C).to(device)

        # true A C
        if input_weights == "True":
            nkf = NeuralKalmanFilter(A, B, C, latent_size=3).to(device)
            zs_nkf, xs_nkf = nkf.predict(xs, us, inf_iters, inf_lr)

        # learn A C
        if input_weights == "Learn":
            AC_nkf = NeuralKalmanFilter(init_A, B, init_C, latent_size=3).to(device)
            AC_nkf.train(xs, us, inf_iters, inf_lr, learn_iters, learn_lr)
            zs_nkf, xs_nkf = AC_nkf.predict(xs, us, inf_iters, inf_lr)

        # random A C
        if input_weights == "Random":
            rAC_nkf = NeuralKalmanFilter(init_A, B, init_C, latent_size=3).to(device)
            zs_nkf, xs_nkf = rAC_nkf.predict(xs, us, inf_iters, inf_lr)

        # estimating using KF
        kf = KalmanFilter(A, B, C, Q, R, latent_size=3).to(device)
        zs_kf, xs_kf = kf.inference(xs, us)

    return zs_kf, xs_kf, zs_nkf, xs_nkf, zs, xs


def pendulum_equation(t, theta):
    g = 9.81  # gravity acceleration (m/s^2)
    L = 3  # length of pendulum (m)
    b = 0.0  # damping factor (kg/s)
    m = 0.1  # mass (kg)

    dtheta2_dt = -(b / m * theta[1]) - (g / L * np.sin(theta[0]))
    dtheta1_dt = theta[1]

    return [dtheta1_dt, dtheta2_dt]


def __run_pendulum_simulation(integration_step, et):
    # initial and end values:
    st = 0  # start time (s)
    # et = 2500.4            # end time (s)
    ts = integration_step  # time step (s)

    theta1_init = 1.8  # initial angular displacement (rad)
    theta2_init = 2.2  # initial angular velocity (rad/s)
    theta_init = [theta1_init, theta2_init]
    t_span = [st, et + st]
    t = np.arange(st, et + st, ts)
    sim_points = len(t)
    l = np.arange(0, sim_points, 1)

    theta12 = solve_ivp(pendulum_equation, t_span, theta_init, t_eval=t)

    return theta12, t


def run_pendulum_simulation(activation):
    step = dt = 0.1
    et = 2500.4  # end time (s)
    ground_truth, time = __run_pendulum_simulation(step, et)
    np.random.seed(random.randint(0, 200))
    ground_truth.y[0, :] += np.random.normal(0, 0.1, ground_truth.y[0].shape)
    ground_truth.y[1, :] += np.random.normal(0, 0.1, ground_truth.y[0].shape)

    sol = np.zeros((ground_truth.y.shape[0], ground_truth.y.shape[1]))
    sol[0, :] = ground_truth.y[1, :]
    sol[1, :] = ground_truth.y[0, :]

    theta1 = ground_truth.y[0, :]  # angular displacement
    theta2 = ground_truth.y[1, :]  # angular velocity

    # use nonlinear model
    A = np.zeros((ground_truth.y.shape[0], ground_truth.y.shape[0]))
    C = np.eye(ground_truth.y.shape[0])
    if activation == 'nonlinear':
        k1 = 8.5
        k2 = 0.9
        decay = 300
    else:
        k1 = 0.01
        k2 = 0.09
        decay = 500
    tpc = TPC(ground_truth.y, A, C, activation=activation, dt=0.1, k1=k1, k2=k2)
    tpc.forward(C_decay=decay)
    data_pred_nl = tpc.get_predictions()
    pred_sol_nl = np.zeros((ground_truth.y.shape[0], ground_truth.y.shape[1]))
    pred_sol_nl[0, :] = data_pred_nl[1, :]
    pred_sol_nl[1, :] = data_pred_nl[0, :]

    return time, ground_truth.y, pred_sol_nl, et, sol, step, data_pred_nl
