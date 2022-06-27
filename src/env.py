import gym

env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)

episodes = 10
for i in range(episodes):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        observation, info = env.reset(return_info=True)
        print("Episode finished after {} timesteps".format(_))
        break

env.close()