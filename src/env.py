import gym
from models import alexnet, agent
from tensorflow.keras.optimizers import Adam

env = gym.make('ALE/SpaceInvaders-v5')
# env.action_space.seed(42)

# episodes = 10

# for episode in range(episodes):
#     state = env.reset()
#     done = False
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         state, reward, done, info = env.step(action)

# env.close()

height, width, channels = env.observation_space.shape
model = alexnet.build_model(height, width, channels, env.action_space.n)
agent = agent.build_agent(model, env.action_space.n)
agent.compile(Adam(lr=1e-3))
agent.fit(env, nb_steps=10000, visualize=True, verbose=1)
