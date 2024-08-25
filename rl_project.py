import itertools
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import copy
import random

# Data
MRPs = {
    0: 20,
    1: 100,
    2: 50,
    3: 50,
    4: 100,
    5: 60,
    6: 35,
    7: 216,
    8: 27,
    9: 130,
    10: 160,
    11: 89,
    12: 73,
    13: 27,
    14: 185,
    15: 249,
    16: 199,
    17: 46,
    18: 55,
    19: 99
}

no_shops = 4
no_items = 2

actions = range(no_shops)

reward_buying = 50

# Bernoulli variable for each shop
bernoulli = np.random.rand(no_shops, no_items)

# Price bias for each shop
bias = np.random.normal(0, 5, no_shops)

# Distance Matrix
a = np.random.uniform(1, 10, (no_shops, no_shops))
distance_matrix = np.tril(a) + np.tril(a, -1).T
np.fill_diagonal(distance_matrix, 0)

print('Distance between shops')
print(distance_matrix)


def price_penalty(next_state):
    scaling = 0.1
    shop = next_state[0]
    next_status = next_state[1]
    price = 0
    for item_no in range(len(next_status)):
        if next_state[1][item_no]:
            price += np.random.normal(MRPs[item_no] + bias[shop], 1)

    return -price * scaling


def distance_penalty(distance):
    return -distance


def availability_in_shop(current_state, next_state):
    old_status = current_state[1]
    new_status = next_state[1]
    next_shop = next_state[0]

    prob = 1

    for item_no in range(len(old_status)):
        if old_status[item_no] == 0:
            if new_status[item_no] == 0:
                prob *= (1 - bernoulli[next_shop][item_no])
            else:
                prob *= bernoulli[next_shop][item_no]

    return prob


def M(shop_b, shop_a):
    temp = sum(sum(np.triu(distance_matrix)))
    temp2 = (temp - distance_matrix[shop_b, shop_a]) / ((no_shops - 1) * temp)
    return temp2


def T(V, current_state):
    alpha = 0.9
    min_action = sys.maxsize
    min_V = float('inf')
    for action in actions:
        summ = 0
        for next_state in state_space:
            if P[(current_state, action, next_state)] != 0:
                summ += (-R[(current_state, action, next_state)] / 500.0 + alpha * V[next_state]) * \
                        P[(current_state, action, next_state)]
                
        if summ < min_V:
            min_V = summ
            min_action = action
            
    return min_V, min_action


# Creating State Space
state_space = []
all_possible_buying_statuses = list(itertools.product([0, 1], repeat=no_items))

for shop_no in range(no_shops):
    for buying_status in all_possible_buying_statuses:
        state = (shop_no, buying_status)
        state_space.append(state)

print('State Space')
print(len(state_space))
# Defining Transition Probabilities and Rewards
P = dict()
R = dict()

actions = range(no_shops)
for current_state, action, next_state in list(itertools.product(state_space, actions, state_space)):
    if current_state[1] == tuple(np.ones(no_items)):
        P[(current_state, action, next_state)] = 0
        R[(current_state, action, next_state)] = None
        continue

    if action == current_state[0]:  # action==current_shop
        if next_state[0] != current_state[0]:
            P[(current_state, action, next_state)] = 0
            R[(current_state, action, next_state)] = None
        else:
            P[(current_state, action, next_state)] = availability_in_shop(current_state, next_state)
            bought_items = sum(next_state[1])
            if bought_items > 0:
                R[(current_state, action, next_state)] = bought_items * reward_buying + distance_penalty(
                    distance_matrix[current_state[0]][next_state[0]]) + price_penalty(next_state)
            else:
                R[(current_state, action, next_state)] = distance_penalty(
                    distance_matrix[current_state[0]][next_state[0]])
    else:
        if next_state[0] == action:
            P[(current_state, action, next_state)] = 0.9 * availability_in_shop(current_state, next_state)
            bought_items = sum(next_state[1])
            if bought_items > 0:
                R[(current_state, action, next_state)] = bought_items * reward_buying + distance_penalty(
                    distance_matrix[current_state[0]][next_state[0]]) + price_penalty(next_state)
            else:
                R[(current_state, action, next_state)] = distance_penalty(
                    distance_matrix[current_state[0]][next_state[0]])
        else:
            P[(current_state, action, next_state)] = 0.1 * availability_in_shop(current_state, next_state) * M(
                next_state[0], current_state[0])
            bought_items = sum(next_state[1])
            if bought_items > 0:
                R[(current_state, action, next_state)] = bought_items * reward_buying + distance_penalty(
                    distance_matrix[current_state[0]][action] + distance_matrix[action][next_state[0]]) + price_penalty(
                    next_state)
            else:
                R[(current_state, action, next_state)] = distance_penalty(
                    distance_matrix[current_state[0]][action] + distance_matrix[action][next_state[0]])

    # print "Next transition:"
    # print "Current State, Action, Next State, P, R"
    # print current_state, action, next_state, P[(current_state,action,next_state)],R[(current_state,action,next_state)]


def value_iteration(state_space):
    threshold = 1
    V = dict()
    next_V = dict()
    policy = dict()
    for state in state_space:
        V[state] = 0.0
        next_V[state] = 0.0
        policy[state] = -1
    flag = True
    while flag:
        flag = False
        for current_state in state_space:
            next_V[current_state], policy[current_state] = T(V, current_state)
            if abs(next_V[current_state] - V[current_state]) > threshold:
                flag = True
            V[current_state] = next_V[current_state]

    return V, policy


def takeaction(current_state, action):
    global P
    global R
    global state_space
    r = random.random()
    for next_state in state_space:
        if r <= 0:
            break
        r -= P[(current_state, action, next_state)]
    return next_state, R[(current_state, action, next_state)]


def RPI():
    global state_space
    state_space_enumeration = dict()
    i = 0
    for state in state_space:
        state_space_enumeration[state] = i
        i += 1
    nextpolicy = dict()
    policy = dict()
    for state in state_space:
        nextpolicy[state] = random.randint(0, no_shops)
    flag = True
    while flag:
        flag = False
        policy = copy.deepcopy(nextpolicy)
        samples = sampling(policy, 10000)
        phi = PVF(len(state_space), samples, 20, state_space_enumeration)
        r = LSPE(len(state_space), samples, phi, state_space_enumeration, 0.7)
        V = phi.dot(r)
        for state in state_space:
            P, nextpolicy[state] = T(V, state)
        for state in state_space:
            if policy[state] != nextpolicy[state]:
                flag = True


def sampling(policy, k):
    global state_space
    samples = []
    current_state = random.choice(state_space)
    for i in range(k):
        sample = []
        next_state, reward = takeaction(current_state, policy[current_state])
        sample.append(current_state)
        sample.append(policy[current_state])
        sample.append(reward)
        samples.append(next_state)
        if not all(i == 0 for i in (next_state[1] - np.ones(no_items))):
            current_state = next_state
        else:
            current_state = random.choice(state_space)
    return samples


def PVF(num_states, samples, k, state_space_enumeration):
    G = np.zeros((num_states, num_states))
    for sample in samples:
        G[state_space_enumeration[sample[0]]][state_space_enumeration[sample[3]]] += 1
    denominator = np.sum(G, axis=1)
    P = np.zeros((num_states, num_states))
    for i in range(num_states):
        for j in range(num_states):
            if denominator[i] != 0:
                P[i, j] = G[i, j] / denominator[i]
    evals, evecs = np.linalg.eigh(P)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[idx]
    perron_vec = evecs[-1]
    D = np.zeros((num_states, num_states))
    for i in range(num_states):
        D[i][i] = perron_vec[i]
    L = D - (D.dot(P) + P.T.dot(D)) / 2.0
    evals, evecs = np.linalg.eigh(L)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[idx]
    phi = np.array([evecs[i] for i in range(k)]).T
    print(phi[0, :][np.newaxis].T.shape)
    return phi


def LSPE(num_states, samples, phi, state_space_enumeration, alpha):
    r = np.zeros(phi.shape[1])[np.newaxis].T
    for k in range(len(samples)):
        C = np.zeros((phi.shape[1], phi.shape[1]))
        d = np.zeros(phi.shape[1])[np.newaxis].T
        for i in range(k):
            C += phi[state_space_enumeration[samples[i][0]], :][np.newaxis].T.dot(
                phi[state_space_enumeration[samples[i][0]], :][np.newaxis])
            q = phi[state_space_enumeration[samples[i][0]], :][np.newaxis].dot(r) - alpha * \
                phi[state_space_enumeration[samples[i][3]], :][np.newaxis].dot(r) + samples[i][2] / 60.0
            d += phi[state_space_enumeration[samples[i][0]], :][np.newaxis].T * q
        r_next = r - np.linalg.pinv(C).dot(d)
        if all(i < 0.1 for i in np.abs(r_next - r)):
            break
    return r


def random_policy():
    current_state = (0, (0, 0))
    random_policy_rewards = 0
    while current_state[1] != (1, 1):
        action = random.randint(0, no_shops - 1)
        next_state, reward = takeaction(current_state, action)
        if reward is not None:  # Check if reward is valid
            random_policy_rewards += reward
        current_state = next_state
    return random_policy_rewards



def q_learning(num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    Q = np.zeros((len(state_space), len(actions)))
    q_learning_rewards = []
    for episode in range(num_episodes):
        current_state = (0, (0, 0))
        episode_reward = 0
        while current_state[1] != (1, 1):
            if random.random() < epsilon:
                action = random.randint(0, no_shops - 1)
            else:
                action = np.argmax(Q[state_space.index(current_state), :])
            next_state, reward = takeaction(current_state, action)
            if reward is not None:  # Check if reward is not None
                Q[state_space.index(current_state), action] += alpha * (
                    reward + gamma * np.max(Q[state_space.index(next_state), :]) -
                    Q[state_space.index(current_state), action])
                episode_reward += reward
            current_state = next_state
        q_learning_rewards.append(episode_reward)
    return q_learning_rewards



ran = []  # For random policy
vi = []   # For value iteration policy
ql = []   # For Q-learning policy

# Run random policy
for i in range(1000):
    ran.append(random_policy())

# Run value iteration policy
for i in range(1000):
    J, policy = value_iteration(state_space)
    current_state = (0, (0, 0))
    summ2 = 0
    while current_state[1] != (1, 1):
        action = policy[current_state]
        next_state, reward = takeaction(current_state, action)
        summ2 += reward
        current_state = next_state
        if reward is not None:
            summ2 += reward
        else:
            summ2 += 0  # or handle it in another way that makes sense for your application

    vi.append(summ2)

# Run Q-learning policy
ql = q_learning(num_episodes=1000)

plt.plot(ran, label='Random Policy', color='green')
plt.plot(vi, label='Value Iteration Policy', color='red')
plt.plot(ql, label='Q-Learning Policy', color='blue')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Comparison of Policies')
plt.show()