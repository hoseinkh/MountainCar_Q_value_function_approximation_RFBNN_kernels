###############################################################################
# For more info, see https://hoseinkh.github.io/
###############################################################################
#
# This takes 4min 30s to run in Python 2.7
# But only 1min 30s to run in Python 3.5!
#
# Note: gym changed from version 0.7.3 to 0.8.0
# MountainCar episode length is capped at 200 in later versions.
# This means your agent can't learn as much in the earlier episodes
# since they are no longer as long.
#
import gym
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm
###############################################################################
# some default parameters of the SGDRegressor:
# loss = 'squared_loss',
# penalty='l2', # this is the regulizer
# alpha=0.0001, 3 coefficient of the regulizer
# fit_intercept=True,
# learning_rate='invscaling',  # the learning decreases with 1/T
###############################################################################
# in this simulations we first generate the samples, scale them and then ...
# ... perform the RL. Hence the states (observations) are generated ...
# ... in the __init__ of the class FeatureTransformer.
class FeatureTransformer:
  def __init__(self, env, n_components=500):
    # generate states (observations)
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    # define scaler and scale the states (observations) --> mean 0 and variance 1
    scaler = StandardScaler()
    scaler.fit(observation_examples)
    #
    # Now we basically use RBF to for feature generation
    # Each RBFSampler takes each (original) (feature representation) of ...
    # ... a state and converts it to "n_components" new featuers.
    # Hence, after concatenating the new features, we convert each state to ...
    # ... {(# RBF samplers) * n_components} new features.
    #
    # We use RBF kernels with different variances to cover different parts ...
    # ... of the space.
    #
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])
    # For all the generated samples, transform original state representaions ...
    # ... to a new state representation using "featurizer"
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))
    #
    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer
  ######################################
  def transform(self, observations):
    #
    scaled_original_state_representation = self.scaler.transform(observations)
    #
    scaled_higher_dimensions_state_representation = self.featurizer.transform(scaled_original_state_representation)
    return scaled_higher_dimensions_state_representation
###############################################################################
# Holds one SGDRegressor for each action
class Model:
  def __init__(self, env, feature_transformer, learning_rate):
    self.env = env
    self.SGDRegressors_approximate_Q_values_for_different_actions = []
    self.feature_transformer = feature_transformer
    for curr_action in range(env.action_space.n):
      # for each action, we fit a model to learn the corresponding Q-values for that action.
      curr_SGDRegressor_approximate_Q_values_for_curr_action = SGDRegressor(learning_rate=learning_rate)
      # we choose initial values high comparing to the typical rewards. This allows us ...
      # ... to use other exploration ideas, such as optimistic initial values, ...
      # ... , besides the epsiolon greedy algorithm!
      initial_optimistic_target_value = 0
      curr_SGDRegressor_approximate_Q_values_for_curr_action.partial_fit(feature_transformer.transform( [env.reset()] ), [initial_optimistic_target_value])
      self.SGDRegressors_approximate_Q_values_for_different_actions.append(curr_SGDRegressor_approximate_Q_values_for_curr_action)
  ######################################
  def predict(self, s):
    X_higher_dimension_representation_of_state_s = self.feature_transformer.transform([s])
    list_Q_values_for_state_s_and_different_actions = []
    for curr_action_index in range(len(self.SGDRegressors_approximate_Q_values_for_different_actions)):
      curr_SGDRegressor_approximate_Q_values_for_curr_action = self.SGDRegressors_approximate_Q_values_for_different_actions[curr_action_index]
      curr_reward_for_curr_action_at_state_s = curr_SGDRegressor_approximate_Q_values_for_curr_action.predict(X_higher_dimension_representation_of_state_s)[0]
      list_Q_values_for_state_s_and_different_actions.append(curr_reward_for_curr_action_at_state_s)
    #
    return list_Q_values_for_state_s_and_different_actions
  ######################################
  def update(self, s, curr_action_index, G):
    X_higher_dimension_representation_of_state_s = self.feature_transformer.transform([s])
    # now we are going to update the SGDRegressor approximator for the Q-values for the action "curr_action_index"
    self.SGDRegressors_approximate_Q_values_for_different_actions[curr_action_index].partial_fit(X_higher_dimension_representation_of_state_s, [G])
  ######################################
  def epsilon_greedy_action_selection(self, s, eps):
    # Here we do epsilon greedy policy
    # Important: we can set eps = 0 so that it becomes purely greedy, ...
    # ... and we still achieve exploration! The reason is that ...
    # ... we use "optimistic initial values", and hence exploration ...
    # automatically happens even without doing explicit exploration ...
    # ... in the policy selection!
    #
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))
###############################################################################
# returns a list of states_and_rewards, and the total reward
def play_one(model, env, eps, discount_rate):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done and iters < 10000:
    action = model.epsilon_greedy_action_selection(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)
    #
    # update the model
    next = model.predict(observation)
    # assert(next.shape == (1, env.action_space.n))
    G = reward + discount_rate*np.max(next[0])
    model.update(prev_observation, action, G)
    #
    totalreward += reward
    iters += 1
    #
  return totalreward
###############################################################################
# here we plot the negative of the optimal state value functions (i,e, -V*(s))!
# Note that the optimal action values are equal to the negative of the average optimal time ...
# ... that it takes to reach the mountain.
# Hence this plot shows the average optimal time to reach the top of the mountain at each state.
def plot_avg_num_remaining_steps(env, estimator, num_tiles=20):
  x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
  y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
  X, Y = np.meshgrid(x, y)
  # both X and Y will be of shape (num_tiles, num_tiles)
  Z = np.apply_along_axis(lambda _: -1*np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
  # Z will also be of shape (num_tiles, num_tiles)
  #
  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, Y, Z,
    rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
  ax.set_xlabel('Position')
  ax.set_ylabel('Velocity')
  ax.set_zlabel('Num steps to reach mountain == -V(s)')
  ax.set_title("Num steps to Reach Mountain Function")
  fig.colorbar(surf)
  fig.savefig("./figs/Num_steps_to_Reach_Mountain.png")
  # plt.show()
  plt.close(fig)
###############################################################################
def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.ylabel("Running Average Total Reward")
  plt.xlabel("Episode")
  # plt.show()
  plt.savefig("./figs/Running_Average_Total_Reward.png")
  plt.close()
###############################################################################
if __name__ == '__main__':
  #
  env = gym.make('MountainCar-v0').env
  ft = FeatureTransformer(env)
  model = Model(env, ft, "constant")
  discount_rate = 0.99
  #
  if True:
    # filename = os.path.basename(__file__).split('.')[0]
    # monitor_dir = './' + filename + '_' + str(datetime.now())
    monitor_dir = os.getcwd() + "/videos/" + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)
  #
  num_of_episodes = 300
  totalrewards = np.empty(num_of_episodes)
  for i in tqdm(range(num_of_episodes)):
    #
    # curr_eps = 0.1*(0.97**i) # use epsilon-greedy policy selection
    curr_eps = 0             # use pure greedy policy selection
    #
    totalreward = play_one(model, env, curr_eps, discount_rate)
    totalrewards[i] = totalreward
    if (i + 1) % 10 == 0:
      print("episode:", i, "total reward:", totalreward)
  #
  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", -totalrewards.sum())
  #
  plt.plot(totalrewards)
  plt.xlabel("Episode")
  plt.ylabel("Rewards")
  # plt.show()
  plt.savefig("./figs/Average_Total_Reward.png")
  plt.close()
  #
  plot_running_avg(totalrewards)
  # plot the optimal state-value function
  plot_avg_num_remaining_steps(env, model)
###############################################################################
H = 5+6