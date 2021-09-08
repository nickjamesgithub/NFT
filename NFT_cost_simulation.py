import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

make_plot = False

# Draw tau from random normal. Centred at 210 days away
tau_mean = 210 # Days away from Ethereum 1.0 - Ethereum 2.0 switch
tau_std = 30 # Standard deviation for Ethereum 1.0 - Ethereum 2.0 switch
tau = np.random.normal(tau_mean, tau_std, 1)

# Draw m from uniform.
m_lb = 100 # Lower bound for reduction in power
m_ub = 10000 # Upper bound for reduction in power
m_mid = (m_lb + m_ub)/2
m = np.random.uniform(m_lb,m_ub,1)

if make_plot:
    # Number of samples to draw to illustrate prior distributions
    num_samples = 20000

    # Samples from Gaussian \tau distribution
    samples_tau = np.random.normal(tau_mean,tau_std,num_samples)
    plt.hist(samples_tau, bins=30, alpha=0.25)
    plt.axvline(x=tau_mean, color='black', alpha=0.4)
    plt.title("20k samples from tau prior distribution")
    plt.xlabel("Days")
    plt.ylabel("Frequency")
    plt.show()

    # Samples from Uniform m distribution
    samples_m = np.random.uniform(m_lb,m_ub,num_samples)
    plt.hist(samples_m, bins=30, alpha=0.25)
    plt.axvline(x=m_mid, color='black', alpha=0.4)
    plt.title("20k samples from uniform prior distribution")
    plt.xlabel("Reduction in energy percentage")
    plt.ylabel("Frequency")
    plt.show()

    # Samples from Uniform m distribution
    samples_m = np.random.uniform(m_lb,m_ub,num_samples)
    plt.hist(samples_m, bins=30, alpha=0.25)
    plt.axvline(x=m_mid, color='black', alpha=0.4)
    plt.title("20k samples from uniform prior distribution")
    plt.xlabel("Reduction in energy percentage")
    plt.ylabel("Frequency")
    plt.show()

# Cost of electricity in Texas and New Mexico
k_texas_mean = 11.39 # Cost of electricity in Texas in cents
k_texas_std = 1.5 # Cost of electricity in Texas in cents

avg_kwh_mean = 369 # Average kwh used per NFT
avg_kwh_std = 20 # Average kwh used per NFT

cost_path_simulations = 50
# Texas Simulation
cost_paths_list = []
total_cost_list = []
while len(cost_paths_list) < cost_path_simulations:
    num_days_year = 365
    tau_draw = np.random.normal(tau_mean, tau_std, 1).astype("int")
    m_draw = np.random.uniform(m_lb,m_ub,1)
    pre_tau_cost = []
    post_tau_cost = []
    while len(pre_tau_cost) < tau_draw:
        avg_kwh_draw = np.random.normal(avg_kwh_mean, avg_kwh_std, 1)
        k_electricity_draw = np.random.normal(k_texas_mean, k_texas_std, 1)
        nft_cost_day_pre = avg_kwh_draw * k_electricity_draw # We want to produce 1 NFT/day
        pre_tau_cost.append(nft_cost_day_pre[0])
    while len(post_tau_cost) < num_days_year - tau_draw:
        avg_kwh_draw = np.random.normal(avg_kwh_mean, avg_kwh_std, 1)
        k_electricity_draw = np.random.normal(k_texas_mean, k_texas_std, 1)
        nft_cost_day_post = avg_kwh_draw * k_electricity_draw * 1/m_draw  # We want to produce 1 NFT/day. Reduction of m
        post_tau_cost.append(nft_cost_day_post[0])

    # Total cost path: concatenate pre and post and store path
    total_cost_path = pre_tau_cost + post_tau_cost
    cost_paths_list.append(total_cost_path)

    # Total cost append
    total_cost_c = np.sum(total_cost_path)
    total_cost_list.append(total_cost_c)

    # Simulation number
    print("Simulation ",len(cost_paths_list))

# Cost paths simulation
for i in range(len(cost_paths_list)):
    plt.plot(cost_paths_list[i])
    plt.ylabel("NFT cost/day in cents")
    plt.xlabel("Day")
plt.show()

# Distribution of total cost (1 NFT/day)



