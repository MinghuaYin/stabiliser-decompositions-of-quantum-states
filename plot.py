import matplotlib.pyplot as plt
import numpy as np
import pickle
from benchmarking.Benchmarking_Data import Benchmarking_Data

base_data_path = 'benchmarking/data/'

pre_string = 'testing C1'
function_strings = ['brute force', 'our method']
generation_strings = ['random unitary', 'random clifford', 'random perturbed clifford']

data_to_plot = []

for function_string in function_strings:
    for generation_string in generation_strings:
        file_name = f'{base_data_path}{pre_string} {function_string} on {generation_string}.npy'

        with open(file_name, 'rb') as fl:
            data : Benchmarking_Data = pickle.load(fl)
            data_to_plot.append(data)

_, ax = plt.subplots()

for data in data_to_plot:
    ax.plot(data.number_qubits, data.times, label = f'{data.function_description} with {data.generator_description}')

ax.set_xlabel(f'n')
ax.set_ylabel('execution time (s)')
ax.set_title(f'Benchmarking Clifford testing')
ax.legend()

plt.show()