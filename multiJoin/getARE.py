import matplotlib.pyplot as plt
import numpy as np

def read_data(filename):
    with open(filename, 'r') as file:
        return list(map(float, file.read().splitlines()))

def calculate_are(actual_values, predicted_values):
    errors = []
    for actual, predicted in zip(actual_values, predicted_values):
        error = abs((actual - predicted) / actual)
        errors.append(error)
    return errors

actual_file = './estimates/stats_CEB_sub_queries_true.txt'
actual_values = read_data(actual_file)

# value_files = ['./results/stat10000', './results/stat1000000', './results/stat100000000',
#                './results/2_stat10000', './results/2_stat1000000', './results/2_stat100000000',
#                './results/stat10000-conv', './results/stat1000000-conv', './results/stat100000000-conv']
value_files = ['./results/job_10000', './results/job_1000000', './results/job_100000000',
                './results/2_job_10000', './results/2_job_1000000', './results/2_job_100000000',
                './results/job_10000-conv', './results/job_1000000-conv', './results/job_100000000-conv']


# 假设的数据集
x = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])
y = np.empty((9,21))
for i in range(len(value_files)):
    predicted_values = read_data(value_files[i])
    error_array = calculate_are(actual_values, predicted_values)
    error_list = [np.percentile(error_array, xx) for xx in x]
    y[i] = np.array(error_list)
# 创建图形
fig, axs = plt.subplots(1,3, figsize=(16,6))

axs[0].plot(x, y[0], label='our-10000', color='red')
axs[1].plot(x, y[1], label='our-1000000', color='red')
axs[2].plot(x, y[2], label='our-100000000', color='red')
axs[0].plot(x, y[3], label='our2-10000', color='orange')
axs[1].plot(x, y[4], label='our2-1000000', color='orange')
axs[2].plot(x, y[5], label='our2-100000000', color='orange')
axs[0].plot(x, y[6], label='conv-10000', color='blue')
axs[1].plot(x, y[7], label='conv-1000000', color='blue')
axs[2].plot(x, y[8], label='conv-100000000', color='blue')

axs[0].set_yscale('log')
axs[0].set_xlabel('Quantile')
axs[0].set_ylabel('ARE')
axs[0].set_title('results')
axs[0].grid(True, which="both", ls="--", c='0.7')
axs[0].legend()
axs[1].set_yscale('log')
axs[1].set_xlabel('Quantile')
axs[1].set_ylabel('ARE')
axs[1].set_title('results')
axs[1].grid(True, which="both", ls="--", c='0.7')
axs[1].legend()
axs[2].set_yscale('log')
axs[2].set_xlabel('Quantile')
axs[2].set_ylabel('ARE')
axs[2].set_title('results')
axs[2].grid(True, which="both", ls="--", c='0.7')
axs[2].legend()

plt.title('results')

plt.tight_layout()

plt.show()