import pickle
import matplotlib.pyplot as plt


WORKING_DIR = 'C:/Users/Amogh/OneDrive - University of Cambridge/Programming-New/CaracalChitalDetector/'

with open(WORKING_DIR + 'data/py_obj/all_results_templating.pkl', 'rb') as f:
    data = pickle.load(f)

for k, v in data.items():
    print(f'{k}\t\t\t{v}')

plt.figure(figsize=(10, 4))
plt.scatter([x[0] for x in data.values()], [x[1] for x in data.values()])
plt.title(f'ROC Curve of Templating Model with {len(list(data.items()))} different parameters')
plt.xlabel("FP rate")
plt.ylabel("TP rate")
plt.savefig(WORKING_DIR + f'data/results/ROC_curve_templating.jpeg', format='jpeg')
plt.show()