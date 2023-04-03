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
st.set_page_config(page_title='Temporal Predictive Coding', layout='centered', initial_sidebar_state='collapsed',
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
    # with plot_columns[1]:
    st.markdown(r"This interactive figure demonstrates the estimates of the acceleration with different models."
                r"Specifically we compare different step sizes of inference between observations with Kalman "
                r"Filter. As you toggle between different step size, you can see that the performance of the "
                r"temporal predictive coding model significantly improves with larger steps and it becomes "
                r"comparable to that of Kalman Filter. The tPC with single inference step has the worst "
                r"performance; however, it still achieves a smooth fit to the system state")

    st.header("Effect of learning A & C matrices")
    st.markdown(r"""In above example, we only inferred the hidden states of the world by assuming known values for
     $A$ and $C$. However, in many real-world situations, we cannot make this assumption, as the structure of the 
     dynamics or observation functions may be unknown. Instead, we need to learn these matrices from observations.
     Our recurrent predictive coding model incorporates a Hebbian plasticity-based learning rule to learn these 
     matrices directly. We explore the impact of learning these parameters on the model's performance by examining 
     three different methods for setting $A$ and $C$ values:
     
 1. using the true values from the data generation; 
 2. learning them via our model's learning rule; and
 3. setting them to random values. 
     
 We evaluate the performance of these models on both latent state level ($x$) and observation level ($y$), 
 assessing the accuracy of the model's estimates for each level. 
 It's important to note that the observation estimates are calculated as the projected latent estimate by $C$, 
 that is, $\hat{y}_k = C\hat{x}_k$, where the value of $C$ is obtained at each time step $k$.
 
 Check out the effect of different methods mentioned using the plotting tool below:
    """)


    ac_columns = st.columns(2)
    ac_plot_columns = st.columns(2)
    col1, col2 = st.columns(2)
    with ac_columns[0]:
        init_w = st.selectbox("Initial weights", ["True", "Learn", "Random"])
    with ac_columns[0]:
        if st.button("Run experiment"):
            # st.session_state.button1_clicked = True
            # if st.session_state.button1_clicked:
            zs_kf1, xs_kf1, zs_nkf1, xs_nkf1, zs1, xs1 = run_tracking_learning(init_w)
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(y=zs1[0], mode="lines", name=f"True Value"))
            fig1.add_trace(go.Scatter(y=zs_nkf1[0], mode="lines", name=f"{init_w} A & C"))
            fig1.add_trace(go.Scatter(y=zs_kf1[0], mode="lines", name=f"Kalman Filter"))
            fig1.update_layout(
                title=f"State",
                xaxis_title="Time",
                yaxis_title=r'$x_1$',
                template="plotly_dark",
                legend=dict(
                    x=1.05,  # X position (fraction) relative to the right of the plot
                    y=1,  # Y position (fraction) relative to the top of the plot
                    xanchor='right',  # Anchor the legend's left side
                    yanchor='bottom',  # Anchor the legend's top side
                )

            )
            # st.plotly_chart(fig)

            # with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=xs1[0], mode="lines", name=f"True Value"))
            fig2.add_trace(go.Scatter(y=xs_nkf1[0], mode="lines", name=f"{init_w} A & C"))
            fig2.add_trace(go.Scatter(y=xs_kf1[0], mode="lines", name=f"Kalman Filter"))
            fig2.update_layout(
                title=f"Observed",
                xaxis_title="Time",
                yaxis_title=r'$y_1$',
                template="plotly_dark",
                legend=dict(
                    x=1.05,  # X position (fraction) relative to the right of the plot
                    y=1,  # Y position (fraction) relative to the top of the plot
                    xanchor='right',  # Anchor the legend's left side
                    yanchor='bottom',  # Anchor the legend's top side
                )

            )
            # st.plotly_chart(fig)

            col1.plotly_chart(fig1, use_container_width=True)
            col2.plotly_chart(fig2, use_container_width=True)

    ### Nonlinear stuff
    st.header("Comparison between linear and non-linear models")
    st.markdown(r"""
    We explored an adjusted version of the temporal predictive coding model that incorporates nonlinear functions 
    $f$ and $g$ in the prediction error equations. This results in simpler parameter update rules, making it easier
     for neural implementation:
    $$
    \epsilon_y = y_k - Cf(\hat{x}_k)
    $$
    $$
    \epsilon_x = \hat{x}_k - Ag(\hat{x}_{k-1})
    $$
    $$
    \frac{d \hat{x}_k}{dt} = - \frac{\partial \mathcal{F}}{\partial \hat{x}_k} = f'(\hat{x}_k) \odot C^T  \epsilon_y - \epsilon_x
    $$
    $$
    \Delta C = - \eta\frac{\partial \mathcal{F}}{\partial C} = \eta\epsilon_y f(\hat{x}_k^T)
    $$
    $$
    \Delta A = - \eta\frac{\partial \mathcal{F}}{\partial A} = \eta\epsilon_x g(\hat{x}_{k-1}^T)
    $$
    

    We tested the model in two experiments: 1) simulating random synthetic data and inferring immediate future 
    observations, and 2) simulating pendulum swinging dynamics and testing learning capabilities.

    """)

    st.subheader("Experiment 1: Synthetic non-linear data")
    st.markdown(r"""
    For the synthetic stimuli experiment, we can generate a two-dimensional data using the following
    nonlinear equations:

    $$
    x_k = A g(x_{k-1}) + \omega_x
    $$
    $$
    y_k = C f(x_k) + \omega_y
    $$
    
    where the nonlinear functions $f$ and $g$ are both chosen to be the hyperbolic tangent function, 
    and $\omega_y$ and $\omega_x$ denote standard Gaussian noise sampled from an $i.i.d$ Gaussian distribution 
    with a mean of 0 and a variance of 0.01. In this 2D example, parameters of $C$ and $A$ were set to a 
    rotation matrix and the identity matrix.
    
    Use the following interactive tool to run a simulation. "Activation" refers to the non-linearity activation function
    to use in the simulation. The non-linear function is a hyperbolic tan.
    
    Note that each time that you run the simulation, a newly random data is generated. 
    
    """)

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
        # st.session_state.button2_clicked = True
        # if st.session_state.button2_clicked:
        solution, error = run_simulation_random_data_np(activation, timepoints)

        # Create columns for the plots
        plot_columns = st.columns(2)
        col3, col4 = st.columns(2)

        # Plot the y_truth and predicted values in the first column
        # with plot_columns[0]:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=solution[:, 0].T, mode="lines", name="True"))
        fig3.add_trace(go.Scatter(y=solution[:, 1].T, mode="lines", name=f"Estimated ({activation})"))
        fig3.update_layout(
            title=f"True state vs. estimated ({activation})",
            xaxis_title="Time",
            yaxis_title="Magnitude",
            template="plotly_dark",
            legend=dict(
                x=1.05,  # X position (fraction) relative to the right of the plot
                y=1,  # Y position (fraction) relative to the top of the plot
                xanchor='right',  # Anchor the legend's left side
                yanchor='bottom',  # Anchor the legend's top side
            )
        )
        # st.plotly_chart(fig)

        # Plot the error in the second column
        moving = 100
        # with plot_columns[1]:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=__moving_average(error, moving).T, mode="lines", name="Error"))
        fig4.update_layout(
            title="Mean squared error",
            xaxis_title="Time",
            yaxis_title="Error",
            template="plotly_dark",
        )
        # st.plotly_chart(fig)

        col3.plotly_chart(fig3, use_container_width=True)
        col4.plotly_chart(fig4, use_container_width=True)


if __name__ == '__main__':
    main()
