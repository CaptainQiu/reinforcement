import gym
import matplotlib.pyplot as plt
import deep_qlearning as dq
import numpy as np

EPISODES = 200
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 300
GAMMA = 0.8
LR = 0.0006
HIDDEN_LAYER = 64
BATCH_SIZE = 32
UPDATE_C = 5
ACTION_SIZE = 2
STATION_SIZE = 4
ISGPU = False
CACHE_LEN = 5000

total_step = 0
episode_durations = []
means=[]
train_is_done=False
env = gym.make('CartPole-v0')
solution = dq.Solution(hidden_layer=HIDDEN_LAYER, gamma=GAMMA, LR=LR,
                       batch_size=BATCH_SIZE, update_C=UPDATE_C,
                       eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY,
                       cache_len=CACHE_LEN, action_size=ACTION_SIZE, station_size=STATION_SIZE, isgpu=ISGPU)


def plot_durations():
    global train_is_done
    plt.figure(2)
    plt.clf()
    durations_t = np.array(episode_durations)
    means.append(np.mean(durations_t[-5:]))
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t)
    # take 100 episode averages and plot them too
    if len(durations_t) >= 10:
        plt.plot(np.array(means))
        if means[-1]==200:
            train_is_done=True

    plt.pause(0.001)  # pause a bit so that plots are updated


for e in range(EPISODES):
    s = env.reset()
    steps = 0
    eps_reward = 0
    total_loss = 0
    while True:
        total_step += 1
        env.render()
        action = solution.select_action(s,train_is_done)
        s_next, r, done, _ = env.step(action)
        if done:
            r = -1

        solution.push_cache([s, action, r, s_next])
        if total_step > BATCH_SIZE and not train_is_done:
            total_loss += solution.train_step()
        s = s_next
        steps += 1
        if done:
            episode_durations.append(steps)
            print(" Episode {0} finished after {1} steps,mean loss {2}"
                  .format(e, steps, total_loss/steps))
            plot_durations()
            break


print('Complete')
plt.ioff()
plt.show()
