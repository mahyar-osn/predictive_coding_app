import streamlit as st

import numpy as np
import plotly.graph_objects as go

from utils.simulations import run_tracking_learning, run_tracking_inf_steps, run_simulation_random_data_np
from utils.contents import intro, theory, kalman, kalman_figure, matrix_learning, nonlinear_intro, exp_one, exp_two

# Set up the page configuration
st.set_page_config(page_title='Temporal Predictive Coding', layout='centered', initial_sidebar_state='collapsed',
                   page_icon=None)

if "button1_clicked" not in st.session_state:
    st.session_state.button1_clicked = False

if "button2_clicked" not in st.session_state:
    st.session_state.button2_clicked = False


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
        step_size = st.number_input("Step Size", min_value=1, max_value=10, value=1, step=1, format="%i")
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
        init_w = st.selectbox("Initial weights", ["True", "Learn", "Random"])
    with ac_columns[0]:
        if st.button("Run experiment"):
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
        activation = st.selectbox("Activation", ["Linear", "Non-linear"])
        if activation == "Linear":
            activation = "linear"
        else:
            activation = "nonlinear"

    with input_columns[1]:
        timepoints = st.number_input("Timepoints", min_value=1000, max_value=5500, value=4000, step=1, format="%i")

    # Create the "Run simulation" button
    if st.button("Run simulation"):
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
    st.image(image_file, caption="A free-body diagram of the pendulum used in this experiment", use_column_width=True)


if __name__ == '__main__':
    main()
