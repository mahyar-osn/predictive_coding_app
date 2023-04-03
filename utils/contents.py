intro = """
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
"""

theory = r"""
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
"""

kalman = r"""
The temporal Predictive Coding (tPC) model and Kalman filtering are both approaches to estimate the
posterior distribution in Bayesian filtering problems, but they differ in their assumptions about the
prior. tPC assumes a Dirac function with density concentrated at the previous estimate, while Kalman
filtering assumes a Gaussian distribution with a non-zero covariance. In a standard tracking problem
with linear dynamics, the performance of the tPC model is close to that of Kalman filtering,
even when ignoring the variance of the previous estimate. However, unlike Kalman filtering,
which relies on analytically solving the maximum of the posterior distribution and requires linear
dynamics, the tPC model uses a gradient-based iterative algorithm that can handle nonlinear dynamics
and be implemented in a biologically plausible network.
"""

kalman_figure = r"""
The above interactive figure demonstrates the estimates of the acceleration with different models.
Specifically we compare different step sizes of inference between observations with Kalman 
Filter. As you toggle between different step size, you can see that the performance of the
temporal predictive coding model significantly improves with larger steps and it becomes
comparable to that of Kalman Filter. The tPC with single inference step has the worst
performance; however, it still achieves a smooth fit to the system state"""

matrix_learning = r"""
In above example, we only inferred the hidden states of the world by assuming known values for
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
"""

nonlinear_intro = r"""
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
"""

exp_one = r"""
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
"""

exp_two = r"""
Figure below illustrates the free-body diagram of the pendulum simulated in our experiment, highlighting mass, 
length, gravity, and force vectors affecting the system. The pendulum's dynamics can be expressed using the 
rotatio motion equation:

\begin{align}
\label{rotational_motion}
\tau = I \alpha
\end{align}

This equation is a slight modification of Newton's second Law of Motion ($F = ma$). In this case,
 $\tau$ represents torque, which is the force's perpendicular component on the mass; $I$ denotes rotational inertia,
  equal to $mL^2$ where $m$ is the mass attached to a string of length $L$; and $\alpha$ signifies angular acceleration,
   calculated by the second time derivative of $\theta$, the angle between gravity and force vectors. 
   Consequently, Equation \ref{rotational_motion} can be rewritten as:

$$
-mgL\sin(\theta) = mL^2 \ddot{\theta}
$$
$$
\Rightarrow mL^2 \ddot{\theta} -mgL\sin(\theta) = 0
$$

Here, $g$ denotes gravity. We excluded a damping parameter in these equations since our simulations do not involve
damping. The second-order ODE was converted into two first-order equations by introducing $\theta_1$ and
$\theta_2$ for angular displacement and angular velocity, respectively:

$$
\theta_1 = \theta
$$
$$
\theta_2 = \dot{\theta}
$$
From this, we can derive the first-order equations:

$$
\dot{\theta_1} = \theta_2
$$
$$
\dot{\theta_2} = - \frac{g}{L} \sin(\theta_1)
$$
"""

exp_two_pendulum = r"""
Now let's try the pendulum experiment!

System variables are initialized as follows: 

- $g$ = $9.81 m/s^{2}$, 
- $L$ = 3.0 m, and 
- $m$ = 0.1 kg.

We will simulate the system as an initial value problem by numerically integrating the equations using the 
explicit Runge-Kutta method for 2500 seconds with $dt$ = 0.1 second time steps and initial values of
$\theta_1$ = 1.8 rad and $\theta_2$ = 2.2 rad/s. These values are chosen to simulate pendulum motion
with a large amplitude of oscillation, pushing the system into a more nonlinear regime. We also introduced
zero-mean Gaussian noise with a variance of 0.1 to the pendulum simulation. For training, the pendulum 
simulations' solutions (angular displacement and angular velocity) are utilized as ground truth. Parameters
$C$ and $A$ are initialized as an identity matrix and zero, respectively, and learning is conducted as in
the previous experiment (i.e., Synthetic Stimuli).

Choose an activation function and press Run Experiment 2!
"""

exp_two_result = r"""
The figure presents the results using the phase portrait of the system. The phase portrait is created 
by computing the derivatives of $\frac{d \theta_1}{dk}$ and $\frac{d \theta_1}{dk}$ at $t = 0$ on a grid of 30 points 
over the range of $-\pi$ to $+\pi$ and -4 to +4 for $\theta_1$ and $\theta_2$, respectively. The solutions of 
the ground-truth simulation and our temporal prediction coding are plotted on the vector field for the final 80 seconds.
"""