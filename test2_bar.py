import numpy as np
import math
import matplotlib.pyplot as plt
# parameter
N = 10
K = 10
T = 100000
mu = np.array((N, K))
mu = [[0.9, 0.4, 0.8, 0.1, 0.3, 0.05, 0.2, 0.1, 0.3, 0.2], [0.4, 0.3, 0.3, 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.4],
[0.1, 0.05, 0.1, 0.4, 0.1, 0.2, 0.9, 0.3, 0.4, 0.1], [0.05, 0.1, 0.9, 0.2, 0.9, 0.75, 0.1, 0.9, 0.25, 0.05],
[0.8, 0.3, 0.1, 0.7, 0.1, 0.4, 0.05, 0.2, 0.75, 0.05], [0.4, 0.05, 0.3, 0.7, 0.05, 0.1, 0.25, 0.75, 0.6, 0.05],
[0.9, 0.3, 0.3, 0.8, 0.1, 0.25, 0.7, 0.05, 0.2, 0.3], [0.3, 0.1, 0.4, 0.25, 0.05, 0.9, 0.25, 0.1, 0.05, 0.4],
[0.8, 0.75, 0.1, 0.2, 0.4, 0.05, 0.3, 0.2, 0.1, 0.25], [0.4, 0.4, 0.9, 0.7, 0.25, 0.2, 0.05, 0.1, 0.4, 0.25]]
def generate_reward(i, k, mu):
    return np.random.normal(mu[i][k], 1)

def find_maxmin_matching(mu, N, K):
    gamma = -100
    maxmin_matching = np.zeros(N)
    for i in range(N):
        for k in range(K):
            if mu[i][k] <= gamma:
                continue
            max_matching, match = find_max_matching(mu, N, K, mu[i][k])
            if max_matching == N:
                gamma = mu[i][k]
                maxmin_matching = match
    return gamma, maxmin_matching

def find_max_matching(mu, N, K, gamma):
    vis = np.zeros(N)
    match = np.zeros(N)
    for i in range(N):
        match[i] = -1
    def dfs(u, match):
        u = int(u)
        for i in range(N):
            if mu[i][u] >= gamma and vis[i] == 0:
                vis[i] = 1
                if (match[i]==-1 or dfs(match[i], match)):
                    match[i] = u
                    return True
        return False

    ans = 0
    for k in range(K):
        vis = np.zeros(N)
        if (dfs(k, match)):
            ans = ans + 1
    # print(match)
    return ans, match

def centralized_ucb(mu, N, K, T):
    regret = np.zeros(T+1)
    gamma_ast, maxmin_matching_optimal = find_maxmin_matching(mu, N, K)
    print(gamma_ast, maxmin_matching_optimal)
    ucb = np.ones((N,K)) * 100
    est_mean = np.zeros((N, K))
    number_of_pull = np.zeros((N, K))
    for t in range(T):
        gamma, maxmin_matching = find_maxmin_matching(ucb, N, K)
        min_mu = 100
        for i in range(N):
            if mu[i][int(maxmin_matching[i])] < min_mu:
                min_mu = mu[i][int(maxmin_matching[i])]
        regret_one_round = gamma_ast - min_mu
        print(regret_one_round)
        for i in range(N):
            reward = generate_reward(i, int(maxmin_matching[i]), mu)
            est_mean[i, int(maxmin_matching[i])] = (number_of_pull[i, int(maxmin_matching[i])] * est_mean[i, int(maxmin_matching[i])] + reward) / (number_of_pull[i, int(maxmin_matching[i])] + 1)
            number_of_pull[i, int(maxmin_matching[i])] += 1
            ucb[i, int(maxmin_matching[i])] = est_mean[i, int(maxmin_matching[i])] + math.sqrt(4 * math.log(T) / number_of_pull[i, int(maxmin_matching[i])])
        regret[t] = regret[t-1] + regret_one_round
        # print(ucb)
    print(regret[T-1])
    return regret


def assign_matching(j, l, plausible_set):
    vis = np.zeros(N)
    match = np.zeros(N)
    for i in range(N):
        match[i] = -1
    match[j] = l

    def dfs(u, match):
        u = int(u)
        for i in range(N):
            if plausible_set[i][u] == 1 and vis[i] == 0:
                vis[i] = 1
                if (match[i]==-1 or dfs(match[i], match)):
                    match[i] = u
                    return True
        return False
    ans = 1
    for k in range(K):
        if k == l:
            continue
        vis = np.zeros(N)
        vis[j] = 1
        if (dfs(k, match)):
            ans = ans + 1
    # print(match)
    return ans, match
    
def elimination(mu, N, K, T):
    regret = np.zeros(2*T)
    gamma_ast, maxmin_matching_optimal = find_maxmin_matching(mu, N, K)
    print(gamma_ast, maxmin_matching_optimal)
    ucb = np.ones((N,K)) * 100
    lcb = np.ones((N,K)) * (-100)
    est_mean = np.zeros((N, K))
    number_of_pull = np.zeros((N, K))
    plausible_set = np.ones((N, K))
    t = 0
    while t <= T:
        for i in range(N):
            for k in range(K):
                if plausible_set[i][k] == 1:
                    ans, matching = assign_matching(i, k, plausible_set)
                    # print(ans)
                    if ans < N:
                        plausible_set[i][k] = 0
                        continue
                    min_mu = 10
                    for j in range(N):
                        if mu[j][int(matching[j])] < min_mu:
                            min_mu = mu[j][int(matching[j])]
                    regret_one_round = gamma_ast - min_mu
                    # print(regret_one_round)
                    for j in range(N):
                        reward = generate_reward(j, int(matching[j]), mu)
                        est_mean[j, int(matching[j])] = (number_of_pull[j, int(matching[j])] * est_mean[j, int(matching[j])] + reward) / (number_of_pull[j, int(matching[j])] + 1)
                        number_of_pull[j, int(matching[j])] += 1
                        ucb[j, int(matching[j])] = est_mean[j, int(matching[j])] + math.sqrt(1.5 * math.log(T) / number_of_pull[j, int(matching[j])])
                        lcb[j, int(matching[j])] = est_mean[j, int(matching[j])] - math.sqrt(1.5 * math.log(T) / number_of_pull[j, int(matching[j])])
                    regret[t] = regret[t-1] + regret_one_round
                    t = t + 1
        gamma_under, under_maxmin_matching = find_maxmin_matching(lcb, N, K)
        for i in range(N):
            for k in range(K):
                if ucb[i,k] < gamma_under:
                    plausible_set[i,k] = 0
    print(regret[T])
    return regret



def my_fair_bandit(mu, N, K, T):
    c1 = 1000
    c2 = 2000
    c3 = 3000
    regret = np.zeros(2*T)
    gamma_ast, maxmin_matching_optimal = find_maxmin_matching(mu, N, K)
    print(gamma_ast, maxmin_matching_optimal)
    ucb = np.ones((N,K)) * 100
    lcb = np.ones((N,K)) * (-100)
    confidence = np.ones((N, K)) * 100
    est_mean = np.zeros((N, K))
    number_of_pull = np.zeros((N, K))
    t = 0
    kk = 1
    w = 0
    e_w = 1
    gamma = np.zeros(T)
    epsilon = np.zeros(T)
    epsilon[0] = 1
    S = np.zeros(T)
    A = np.zeros((T, N))
    while t <= T:
        print(kk, regret[t-1])
        # Exploration phase
        for ind in range(int(c1 * math.log(kk+1))+1):
            m = np.ones(N) * (-1)
            pulled = np.zeros(K)
            is_matching = True
            regret_one_round = 0
            for i in range(N):
                m[i] = np.random.randint(0, K)
                # print(m[i])
                pulled[int(m[i])] += 1
            for i in range(N):
                if pulled[int(m[i])] > 1:
                    m[i] = -1
                    regret_one_round = gamma_ast
                else:
                    if regret_one_round < gamma_ast - mu[i][int(m[i])]:
                        regret_one_round = gamma_ast - mu[i][int(m[i])]
                    reward = generate_reward(i, int(m[i]), mu)
                    est_mean[i, int(m[i])] = (number_of_pull[i, int(m[i])] * est_mean[i, int(m[i])] + reward) / (number_of_pull[i, int(m[i])] + 1)
                    number_of_pull[i, int(m[i])] += 1
            regret[t] = regret[t-1] + regret_one_round
            t += 1
        for i in range(N):
            for k in range(K):
                confidence[i][k] = math.sqrt(K / math.log(number_of_pull[i][k]+1))
        # Matching Phase
        w = w + 1
        if w == e_w:
            gamma[kk] = 0
            w = 0
            e_w = int(k/3) + 1
            epsilon[kk] = 1 / (1 + math.log(kk))
        else:
            epsilon[kk] = epsilon[kk-1]
        E = [[] for _ in range(N)]
        for i in range(N):
            for k in range(K):
                if est_mean[i][k] >= gamma[kk] - confidence[i][k]:
                    E[i].append(k)
        m = np.ones(N) * (-1)
        pulled = np.zeros(K)
        is_matching = True
        regret_one_round = 0
        for i in range(N):
            for i in range(N):
                if E[i] != []:
                    m[i] = np.random.choice(E[i])
                    pulled[int(m[i])] += 1
            pulled[int(m[i])] += 1
        for i in range(N):
            if pulled[int(m[i])] > 1:
                regret_one_round = gamma_ast
                is_matching = False
            else:
                if regret_one_round < gamma_ast - mu[i][int(m[i])]:
                    regret_one_round = gamma_ast - mu[i][int(m[i])]
        regret[t] = regret[t-1] + 1
        t += 1
        for ind in range(int(c2 * math.log(kk+1))+1):
            for i in range(N):
                if pulled[int(m[i])] > 1:
                    for i in range(N):
                        if pulled[int(m[i])] > 1:
                            if E[i] != []:
                                m[i] = np.random.choice(E[i])
                            else:
                                m[i] = -1
            for k in range(K):
                pulled[k] = 0
            for i in range(N):
                pulled[int(m[i])] += 1
            regret_one_round = 0
            for i in range(N):
                if pulled[int(m[i])] > 1:
                    regret_one_round = gamma_ast
                    is_matching = False
                else:
                    if regret_one_round < gamma_ast - mu[i][int(m[i])]:
                        regret_one_round = gamma_ast - mu[i][int(m[i])]
            regret[t] = regret[t-1] + 1
            t += 1
        for i in range(N):
            A[kk][i] = m[i]
        is_matching = True
        for i in range(N):
            if pulled[int(m[i])] > 1:
                is_matching = False
        if is_matching == True:
            gamma[kk+1] = gamma[kk] + epsilon[kk]
            S[kk] = 1
        else:
            gamma[kk+1] = gamma[kk]
            S[kk] = 0
        k_ast = int(kk/2) + 1
        max_gamma = gamma[k_ast] * S[k_ast]
        for l in range(int(kk/2) + 1, kk+1):
            if gamma[l] * S[l] > max_gamma:
                max_gamma = gamma[l] * S[l]
                k_ast = l
        # Exploitation Phase
        for ind in range(int(c3 * math.pow(4/3, k))+1):
            m = np.ones(N) * (-1)
            pulled = np.zeros(K)
            regret_one_round = 0
            for i in range(N):
                m[i] = A[k_ast][i]
                pulled[int(m[i])] += 1
            for i in range(N):
                if pulled[int(m[i])] > 1:
                    m[i] = -1
                    regret_one_round = gamma_ast
                else:
                    if regret_one_round < gamma_ast - mu[i][int(m[i])]:
                        regret_one_round = gamma_ast - mu[i][int(m[i])]
                    reward = generate_reward(i, int(m[i]), mu)
                    est_mean[i, int(m[i])] = (number_of_pull[i, int(m[i])] * est_mean[i, int(m[i])] + reward) / (number_of_pull[i, int(m[i])] + 1)
                    number_of_pull[i, int(m[i])] += 1
            regret[t] = regret[t-1] + regret_one_round
            t += 1
        kk = kk + 1
    return regret

def leshem(mu, N, K, T):
    L = 1000
    regret = np.zeros(2*T)
    gamma_ast, maxmin_matching_optimal = find_maxmin_matching(mu, N, K)
    print(gamma_ast, maxmin_matching_optimal)
    ucb = np.ones((N,K)) * 100
    lcb = np.ones((N,K)) * (-100)
    confidence = np.ones((N, K)) * 100
    est_mean = np.zeros((N, K))
    number_of_pull = np.zeros((N, K))
    t = 0
    kk = 1
    while t <= T:
        print(kk, regret[t-1])
        # Exploration phase
        for ind in range(int(L * math.log(kk+1))+1):
            m = np.ones(N) * (-1)
            pulled = np.zeros(K)
            is_matching = True
            regret_one_round = 0
            for i in range(N):
                m[i] = (ind + i) % K
                # print(m[i])
                pulled[int(m[i])] += 1
            for i in range(N):
                if pulled[int(m[i])] > 1:
                    m[i] = -1
                    regret_one_round = gamma_ast
                else:
                    if regret_one_round < gamma_ast - mu[i][int(m[i])]:
                        regret_one_round = gamma_ast - mu[i][int(m[i])]
                    reward = generate_reward(i, int(m[i]), mu)
                    est_mean[i, int(m[i])] = (number_of_pull[i, int(m[i])] * est_mean[i, int(m[i])] + reward) / (number_of_pull[i, int(m[i])] + 1)
                    number_of_pull[i, int(m[i])] += 1
            regret[t] = regret[t-1] + regret_one_round
            t += 1
        for ind in range(10 * N*N*N):
            regret_one_round = gamma_ast
            regret[t] = regret[t-1] + regret_one_round
            t += 1
        ans, matching = find_maxmin_matching(est_mean, N, K)
        for ind in range(int(math.pow(2, kk))+1):
            regret_one_round = 0
            for i in range(N):
                if regret_one_round < gamma_ast - mu[i][int(matching[i])]:
                    regret_one_round = gamma_ast - mu[i][int(matching[i])]
            regret[t] = regret[t-1] + regret_one_round
            t += 1
        kk += 1
    print(regret[T])
    return regret
# regret_cenucb = centralized_ucb(mu, N, K, 100000)

Repeat = 20
m, l, e = [], [], []
for counter in range(Repeat):
    regret_myfair = my_fair_bandit(mu, N, K, T)
    m.append(regret_myfair[:T])
    regret_leshem = leshem(mu, N, K, T)
    l.append(regret_leshem[:T])
    regret_elimination = elimination(mu, N, K, T)
    e.append(regret_elimination[:T])


m = np.array(m)
l = np.array(l)
e = np.array(e)
m_variance = np.std(m, axis=0)
l_variance = np.std(l, axis=0)
e_variance = np.std(e, axis=0)


M, L, E = m[0], l[0], e[0]
for i in range(1, Repeat):
    M = M + m[i]
    L = L + l[i]
    E = E + e[i]
M = M/Repeat
L = L/Repeat
E = E/Repeat

# regret_elimination = elimination(mu, N, K, T)
# regret_leshem = leshem(mu, N, K, T)
# regret_myfair = my_fair_bandit(mu, N, K, T)
#
# regret_elimination = regret_elimination[:T]
# regret_leshem = regret_leshem[:T]
# regret_myfair = regret_myfair[:T]

y0, yLim = -200, L[-1]+200

plt.figure(dpi=600)
x = np.linspace(1, T, T)
plt.grid(True)
plt.xlabel("round")
plt.ylabel("regret")
plt.ylim((y0, yLim))
plt.plot(x, E, label = 'DFE', color="red", linewidth="2")
plt.plot(x, E - e_variance, color='pink', lw=0.8)
plt.plot(x, E + e_variance, color='pink', lw=0.8)
plt.fill_between(x, E - e_variance, E + e_variance, alpha=0.25, color='pink')

plt.plot(x, L, label = 'Leshem', color="blue", linewidth="2")
plt.plot(x, L - l_variance, color='skyblue', lw=0.8)
plt.plot(x, L + l_variance, color='skyblue', lw=0.8)
plt.fill_between(x, L - l_variance, L + l_variance, alpha=0.25, color='skyblue')

plt.plot(x, M, label = 'My fair bandit', color="green", linewidth="2")
plt.plot(x, M + m_variance, color='lightgreen', lw=0.8)
plt.plot(x, M - m_variance, color='lightgreen', lw=0.8)
plt.fill_between(x, M - m_variance, M + m_variance, alpha=0.25, color='lightgreen')

plt.xlabel("Round", fontsize=20)
plt.ylabel("Regret", fontsize=20)
plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
plt.legend(fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("fig10_bar.png", dpi=600, bbox_inches='tight')
# plt.show()
