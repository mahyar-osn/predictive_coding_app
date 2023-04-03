import streamlit as st

import altair as alt
import pandas as pd

import numpy as np
import plotly.graph_objects as go

from utils.simulations import run_tracking_learning, run_tracking_inf_steps, run_simulation_random_data_np, \
    run_pendulum_simulation, pendulum_equation
from utils.contents import intro, theory, kalman, kalman_figure, matrix_learning, nonlinear_intro, exp_one, exp_two, \
    exp_two_pendulum, exp_two_result

# Set up the page configuration
st.set_page_config(page_title='Temporal Predictive Coding', layout='centered', initial_sidebar_state='collapsed',
                   page_icon=None)


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


def create_pendulum_animation_altair(time, ground_truth, pred_sol_nl, current_time_step):
    L = 3
    x_ground_truth = L * np.sin(ground_truth[0, :])
    y_ground_truth = -L * np.cos(ground_truth[0, :])
    x_pred_sol_nl = L * np.sin(pred_sol_nl[0, :])
    y_pred_sol_nl = -L * np.cos(pred_sol_nl[0, :])

    data = pd.DataFrame({'time': np.tile(time[-20:], 2),
                         'x': np.hstack((x_ground_truth[-20:], x_pred_sol_nl[-20:])),
                         'y': np.hstack((y_ground_truth[-20:], y_pred_sol_nl[-20:])),
                         'type': np.repeat(['Ground truth', 'Predicted'], 20)})

    chart = alt.Chart(data).mark_circle(size=100).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=(-L, L))),
        y=alt.Y('y:Q', scale=alt.Scale(domain=(-L, L))),
        color='type:N'
    ).properties(
        width=600,
        height=600
    ).transform_filter(
        alt.expr.datum.time == current_time_step
    )

    return chart


def main():
    # Apply the custom theme
    set_custom_theme()  # Introduction section
    st.title("Temporal Predictive Coding in the Brain 🧠")
    st.write(intro)

    # Theory
    st.header("Theory & Algorithm")
    st.markdown(theory, unsafe_allow_html=True)

    # Relationship to Kalman Filtering:
    st.header("Relationship to Kalman Filter")
    st.markdown(kalman)

    kalman_columns = st.columns(2)
    with kalman_columns[0]:
        step_size = st.number_input("👉 Step Size", min_value=1, max_value=10, value=1, step=1, format="%i")
    with st.spinner("Running..."):
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
    st.markdown(kalman_figure)

    st.header("Effect of learning A & C matrices")
    st.markdown(matrix_learning)

    ac_columns = st.columns(2)
    col1, col2 = st.columns(2)
    with ac_columns[0]:
        init_w = st.selectbox("◻️◽▪️ Initial weights", ["True", "Learn", "Random"])
    with ac_columns[0]:
        if st.button("Run experiment 🧪"):
            if init_w == "True":
                prompt = "Using true A & C. This should be relatively quick..."
            elif init_w == "Learn":
                prompt = "Learning A & C. This might take a while. Please be patient..."
            elif init_w == "Random":
                prompt = "Using random A & C. This should be relatively quick..."

            with st.spinner(prompt):
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
            col1.plotly_chart(fig1, use_container_width=True)
            col2.plotly_chart(fig2, use_container_width=True)

    # Nonlinear vs. linear comparisons
    st.header("Comparison between linear and non-linear models")
    st.markdown(nonlinear_intro)

    st.subheader("Experiment 1: Synthetic non-linear data")
    st.markdown(exp_one)

    # Create columns for the input fields
    input_columns = st.columns(2)

    # Add the input fields within the columns
    with input_columns[0]:
        activation = st.selectbox("🧮 Activation", ["Linear", "Non-linear"])
        if activation == "Linear":
            activation = "linear"
        else:
            activation = "nonlinear"

    with input_columns[1]:
        timepoints = st.number_input("⌛ Timepoints", min_value=1000, max_value=5500, value=4000, step=1, format="%i")

    # Create the "Run simulation" button
    if st.button("Run experiment 1 🧪"):
        with st.spinner("Running..."):
            solution, error = run_simulation_random_data_np(activation, timepoints)

        # Create columns for the plots
        col3, col4 = st.columns(2)

        # Plot the y_truth and predicted values in the first column
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

        # Plot the error in the second column
        moving = 100
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(y=__moving_average(error, moving).T, mode="lines", name="Error"))
        fig4.update_layout(
            title="Mean squared error",
            xaxis_title="Time",
            yaxis_title="Error",
            template="plotly_dark",
        )

        col3.plotly_chart(fig3, use_container_width=True)
        col4.plotly_chart(fig4, use_container_width=True)

    # Pendulum
    st.subheader("Experiment 2: Pendulum simulation")
    st.markdown(exp_two)
    image_file = "resources/pendulum.png"
    col = st.columns(3)
    with col[1]:
        st.image(image_file, caption="", use_column_width='auto')

    st.markdown(exp_two_pendulum)

    # Create columns for the input fields
    input_pen_columns = st.columns(2)

    # Add the input fields within the columns
    with input_pen_columns[0]:
        pen_activation = st.selectbox("🧮 Activation function", ["Linear", "Non-linear"])
        if pen_activation == "Linear":
            pen_activation = "linear"
        else:
            pen_activation = "nonlinear"
    flag = False
    with input_pen_columns[0]:
        if st.button("Run experiment 2 🧪"):
            flag = True
            with st.spinner("Running..."):
                time, ground_truth, pred_sol_nl, et, sol, step, data_pred_nl = run_pendulum_simulation(pen_activation)
                fn = 80
                ff = sol[0].shape[0] - fn
                X, Y = np.mgrid[(-np.pi):np.pi:-30j, -4:4:30j]
                stt = 0  # start time (s)
                tss = step  # time step (s)
                theta1_init = 1.8  # initial angular displacement (rad)
                theta2_init = 2.2  # initial angular velocity (rad/s)
                theta_init = [theta1_init, theta2_init]
                t_span = [stt, et + stt]
                U, V = pendulum_equation(t_span, [X, Y])

                fig = go.Figure()

                # Quiver plot
                fig.add_trace(go.Cone(x=X.ravel(), y=Y.ravel(), u=U.ravel(), v=V.ravel(), sizemode='scaled',
                                      sizeref=0.2, showscale=False, colorscale='Viridis'))

                # True line
                fig.add_trace(go.Scatter(x=sol[1, ff:], y=sol[0, ff:], mode='lines', name='True',
                                         line=dict(width=3)))

            fig.add_trace(
                go.Scatter(x=data_pred_nl[0, ff:], y=data_pred_nl[1, ff:], mode='lines',
                           name=f'Inferred ({pen_activation})',
                           line=dict(width=3)))
            fig.update_layout(
                width=600,
                height=600,
                title='Mean phase portrait',
                xaxis=dict(title=r'$\theta_1$', showgrid=True, gridwidth=0.5),
                yaxis=dict(title=r'$\theta_2$', showgrid=True, gridwidth=0.5),
                legend=dict(x=1, y=1, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)')
            )

            st.plotly_chart(fig)
    if flag:
        st.markdown(exp_two_result)

    st.header("References:")
    st.markdown(r"""
    📰 Please refer to the original study for full detail [PLACEHOLDER URL].
    
    🐙 Visit our [GitHub page](https://github.com/C16Mftang/temporal-predictive-coding) for the source code of tPC.
    """)


if __name__ == '__main__':
    main()
