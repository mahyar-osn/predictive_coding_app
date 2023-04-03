import streamlit as st

import random
import torch

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.models import NeuralKalmanFilter, KalmanFilter
from src.utils import *

from src.np_implementation.model import TPC
from src.np_implementation.data import generate_random_nonlinear_data

# Set up the page configuration
st.set_page_config(page_title='Temporal Predictive Coding', layout='wide', initial_sidebar_state='collapsed',
                   page_icon=None)

device = "cuda" if torch.cuda.is_available() else "cpu"


if "button1_clicked" not in st.session_state:
    st.session_state.button1_clicked = False

if "button2_clicked" not in st.session_state:
    st.session_state.button2_clicked = False


def to_np(x):
    return x.cpu().detach().numpy()


# Custom CSS for dark theme
def set_custom_theme():
    st.markdown("""
    <style>
        body {
            background-color: #1f1f1f;
            color: #ffffff;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #ffffff;
        }
        .sidebar .sidebar-content {
            background-color: #333333;
            color: #ffffff;
        }
    </style>
    """, unsafe_allow_html=True)


def __moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def run_simulation_random_data_np(activation, timepoints):
    with st.spinner("Running..."):
        dt = 1 / 2
        x = np.array((1., 0.))
        C_init = np.eye(2) * 2.5
        A_init = (np.array((
            (-dt / 2, 1.),
            (-1., -dt / 2)))) * 2.5

        # Generate random data (add your generate_random_nonlinear_data function)
        y_truth, A_truth = generate_random_nonlinear_data(2, 2, timepoints, x, C_init, A_init, seed=random.randint(0, 200))

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
    with st.spinner("Running..."):
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

    if input_weights == "True":
        prompt = "Using true A & C. This should be relatively quick..."
    elif input_weights == "Learn":
        prompt = "Learning A & C. This might take a while. Please be patient..."
    elif input_weights == "Random":
        prompt = "Using random A & C. This should be relatively quick..."

    with st.spinner(prompt):
        latent_mses = np.zeros((4, len(seeds)))
        obs_mses = np.zeros((4, len(seeds)))
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


def main():
    # Apply the custom theme
    set_custom_theme()  # Introduction section
    st.title("Temporal Predictive Coding in the Brain 🧠")
    st.write("""
    Welcome to our web app! We're excited to share our recent study on temporal predictive coding in the brain, which
     focuses on understanding how our sensory system processes dynamically changing stimuli over time.
    The brain faces the challenge of inferring the state of the world from a sequence of changing stimuli, rather than
     just static information. In our study, we address key questions related to the neural implementation and
      computational properties of temporal predictive coding networks, a framework that has its roots in predicting
       changes over time.
    We present an adjusted version of the temporal predictive coding model that is more biologically plausible,
     and develop recurrent networks that rely solely on local inputs and Hebbian learning. Our research also explores 
     the mathematical relationship between temporal predictive coding and Kalman Filtering, a popular technique 
     for predicting the behavior of linear systems.
    We demonstrate that our predictive coding networks can approximate the performance of the Kalman Filter and can be
     effectively generalized to non-linear systems. By examining these models, we aim to shed light on how biological
      circuits can predict future stimuli and guide research on understanding specific neural circuits in brain areas
       involved in temporal prediction.

    Take a look around the site to explore our findings and learn more about the fascinating world of
     temporal predictive coding in the brain!
    """)

    # Theory
    st.header("Theory & Algorithm")
    st.markdown(r"""
    In this simplified two-layer network model,
    the observation layer receives dynamic inputs across discrete time steps and the hidden layer contains the hidden 
    activity at each step. The model assumes a hidden Markov model where the hidden state at each time step depends only
     on the previous hidden state, and the observation at a certain time step depends only on its corresponding hidden 
     state. This leads to an objective function, $\mathcal{F}_k$, which consists of squared prediction errors on two 
     levels: top-down prediction error $\epsilon_y:=y_k - Cf(x_k)$ and temporal prediction error 
     $\epsilon_x:=x_k - Af(\hat{x}_{k-1})$. 
     Temporal Predictive Coding (tPC) infers latent states using gradient descent on $\mathcal{F}_k$:
    $$
    \mathcal{F}_k =  \Vert y_k - Cf(x_k) \Vert^2_2 + \Vert x_k - Af(\hat{x}_{k-1})\Vert^2_2
    $$
    $$
    \dot{x}_k = - \frac{\partial \mathcal{F}_k}{\partial x_k} = - \epsilon_{x} + f'(x_k) \odot C^T \epsilon_y
    $$
    $$
    \Delta A \propto - \frac{\partial \mathcal{F}_k}{\partial A} = \epsilon_{x} f(\hat{x}_{k-1})^T; \, \Delta C \propto -\frac{\partial \mathcal{F}_k}{\partial C} = \epsilon_y f(x_k)^T
    $$

    Equations above describe how to perform the gradient descent on $\mathcal{F}_k$ to update the hidden state and learn the transition and output matrices, $A$ and $C$. These computations can be implemented in the networks using local and Hebbian computations. The model introduces an extra set of neurons storing the estimate $\hat{x}_{k-1}$ from the previous time step, which is reloaded with $\hat{x}_k$ after the inference dynamic converges at each step. This mechanism can be achieved by neurons storing the representation of stimuli presented a few seconds earlier in their activity.
    """, unsafe_allow_html=True)  # Create columns for the input fields

    # Relationship to Kalman Filtering:
    st.header("Relationship to Kalman Filter")
    st.markdown(r"The temporal Predictive Coding (tPC) model and Kalman filtering are both approaches to estimate the "
                r"posterior distribution in Bayesian filtering problems, but they differ in their assumptions about the"
                r" prior. tPC assumes a Dirac function with density concentrated at the previous estimate, while Kalman"
                r" filtering assumes a Gaussian distribution with a non-zero covariance. In a standard tracking problem "
                r"with linear dynamics, the performance of the tPC model is close to that of Kalman filtering, "
                r"even when ignoring the variance of the previous estimate. However, unlike Kalman filtering, "
                r"which relies on analytically solving the maximum of the posterior distribution and requires linear "
                r"dynamics, the tPC model uses a gradient-based iterative algorithm that can handle nonlinear dynamics"
                r" and be implemented in a biologically plausible network.")

    kalman_columns = st.columns(2)
    with kalman_columns[0]:
        step_size = st.number_input("Step Size", min_value=1, max_value=10, value=1, step=1, format="%i")
    zs, zs_nkf, zs_kf = run_tracking_inf_steps(step_size)
    plot_columns = st.columns(2)
    # Plot the y_truth and predicted values in the first column
    with plot_columns[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=zs[2, 570:591], mode="lines", name="True"))
        fig.add_trace(go.Scatter(y=zs_nkf[2, 570:591], mode="lines", name=f"tPC ({step_size} Steps)"))
        fig.add_trace(go.Scatter(y=zs_kf[2, 570:591], mode="lines", name=f"Kalman Filter)"))
        fig.update_layout(
            title=f"Estimated Acceleration",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_dark",
        )
        st.plotly_chart(fig)
    with plot_columns[1]:
        st.markdown(r"This interactive figure demonstrates the estimates of the acceleration with different models."
                    r"Specifically we compare different step sizes of inference between observations with Kalman"
                    r"Filter. As you toggle between different step size, you can see that the performance of the "
                    r"temporal predictive coding model significantly improves with larger steps and it becomes "
                    r"comparable to that of Kalman Filter. The tPC with single inference step has the worst "
                    r"performance; however, it still achieves a smooth fit to the system state")

    st.header("Effect of learning A & C matrices")
    ac_columns = st.columns(2)
    ac_plot_columns = st.columns(2)
    with ac_columns[0]:
        init_w = st.selectbox("Initial weights", ["True", "Learn", "Random"])
    with ac_columns[0]:
        if st.button("Run experiment"):
            st.session_state.button1_clicked = True

        if st.session_state.button1_clicked:
            zs_kf1, xs_kf1, zs_nkf1, xs_nkf1, zs1, xs1 = run_tracking_learning(init_w)
            with ac_plot_columns[0]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=zs1[0], mode="lines", name=f"True Value"))
                fig.add_trace(go.Scatter(y=zs_nkf1[0], mode="lines", name=f"{init_w}"))
                fig.add_trace(go.Scatter(y=zs_kf1[0], mode="lines", name=f"Kalman Filter)"))
                fig.update_layout(
                    title=f"State",
                    xaxis_title="Time",
                    yaxis_title=r'$x_1$',
                    template="plotly_dark",
                )
                st.plotly_chart(fig)

            with ac_plot_columns[1]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=xs1[0], mode="lines", name=f"True Value"))
                fig.add_trace(go.Scatter(y=xs_nkf1[0], mode="lines", name=f"{init_w}"))
                fig.add_trace(go.Scatter(y=xs_kf1[0], mode="lines", name=f"Kalman Filter"))
                fig.update_layout(
                    title=f"Observed",
                    xaxis_title="Time",
                    yaxis_title=r'$y_1$',
                    template="plotly_dark",
                )
                st.plotly_chart(fig)


    ### Nonlinear stuff
    st.header("Comparison between linear and non-linear models")
    st.markdown(r"..............................")
    st.markdown(r"..............................")
    # Create columns for the input fields
    input_columns = st.columns(2)

    # Add the input fields within the columns
    with input_columns[0]:
        activation = st.selectbox("Activation", ["Linear", "Non-linear"])
        if activation == "Linear":
            activation = "linear"
        else:
            activation = "nonlinear"

    with input_columns[1]:
        timepoints = st.number_input("Timepoints", min_value=1000, max_value=5500, value=4000, step=1, format="%i")

    # Create the "Run simulation" button
    if st.button("Run simulation"):
        st.session_state.button2_clicked = True

    if st.session_state.button2_clicked:
        solution, error = run_simulation_random_data_np(activation, timepoints)

        # Create columns for the plots
        plot_columns = st.columns(2)

        # Plot the y_truth and predicted values in the first column
        with plot_columns[0]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=solution[:, 0].T, mode="lines", name="True"))
            fig.add_trace(go.Scatter(y=solution[:, 1].T, mode="lines", name=f"Estimated ({activation})"))
            fig.update_layout(
                title=f"True state vs. estimated ({activation})",
                xaxis_title="Time",
                yaxis_title="Magnitude",
                template="plotly_dark",
            )
            st.plotly_chart(fig)

        # Plot the error in the second column
        moving = 100
        with plot_columns[1]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=__moving_average(error, moving).T, mode="lines", name="Error"))
            fig.update_layout(
                title="Mean squared error",
                xaxis_title="Time",
                yaxis_title="Error",
                template="plotly_dark",
            )
            st.plotly_chart(fig)


if __name__ == '__main__':
    main()
