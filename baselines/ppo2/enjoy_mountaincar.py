import gym

from baselines import deepq


def main():
    env = gym.make("MountainCarContinuous-v0")
    # act = deepq.load("mountaincar_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step([-1])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
