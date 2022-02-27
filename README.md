# MountainCar_Q_value_function_approximation_RFBNN_kernels

 We use tabular state-action value functions (Q-values) to find the optimal policy for controlling a Cart Pole.

<br />

## Task:

The goal is to design a control policy to guide the car to the top of the mountain.



## Solution:

We first use RBF kernels to map the features to higher feature space. We then assume a simple linear policy, and learn the state-action value functions (Q-values) to learn the optimal policy. We also use optimistic initial values as a technique to do exploration.

---



A car is on a one-dimensional track, positioned between two “mountains”. The goal is to drive up the mountain on the right; however, the car’s engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

The car’s state, at any point in time, is given by a **two-dimensional vector** containing its **horizonal position** and **velocity**. The car commences each episode stationary, at the bottom of the valley between the hills (at position approximately -0.5), and the episode ends when either the car reaches the flag (position > 0.5).



At each move, the car has three actions available to it: push **left**, push **right** or **do nothing**, and a penalty of 1 unit is applied for each move taken (including doing nothing). This actually allows for using the exploration technique called **optimistic initial values**.



The steps of this code is summarized in the following flowchart.

1. Create RBF kernels (each one with different $\sigma$), train them on some samples initially. These kernels are used to create new features.
2. We then build three different linear regression models (each for a different action) for the state-action values (Q-values). 
3. We initialize all the state-action values to 0. This is optimistic intitial values. This means that even if we do not use $\epsilon$-greedy, but use the pure greedy action selection, these optimistic intitial values results in automatic explorations.
4. At each step, we use greedy or $\epsilon$-greedy action selection.



We also record (a video for) the performance of the algorithm for the optimal policy.



The general form of the Mountain car is shown in the following figure.



<p float="left">
  <img src="/figs/Mountain_Car_general_form.png" width="450" />
</p>





After training the model, the results are shown as the follows:

The performance of the optimal policy:

<p float="left">
  <img src="/figs/Mountain_Car_optimal_policy_Q_val_apprx_RBF_kernal.gif" width="450" />
</p>



Average total reward over different episodes:

<p float="left">
  <img src="/figs/Mountain_Car_Average_Total_Reward_Q_val_fun_approx.png" width="450" />
</p>



And the average # steps to reach to the top of the mountain at different state-action pairs (which is practically $(-Q^*(s,a))$) is shown in the following figure:

<p float="left">
  <img src="/figs/Mountain_Car_Num_steps_to_Reach_Mountain_Q_val_fun_approx.png" width="450" />
</p>



<br />
