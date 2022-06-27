import gym
env = gym.make("LunarLander-v2")
env.action_space.seed(42)

env.reset()

for _ in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()

    if done:
        env.reset()

env.close()