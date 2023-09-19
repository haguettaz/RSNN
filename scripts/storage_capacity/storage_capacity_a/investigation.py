import os

from tqdm import tqdm, trange

from rsnn.utils.utils import load_object_from_file

list_of_dict = []

ref_firing_times = load_object_from_file(os.path.join("spike_trains", f"firing_times_300.pkl"))

for l in trange(100):
    for num_inputs in [250,500,1000]:
        if not os.path.exists(os.path.join("neurons", f"neuron_{l}_300_500.pkl")):
            print(os.path.join("neurons", f"neuron_{l}_300_500.pkl"), "does not exist")
            continue
        neuron = load_object_from_file(os.path.join("neurons", f"neuron_{l}_300_500.pkl"))
        if neuron.status != "optimal":
            print(f"neuron:", l)
            print(f"firing_times:", ref_firing_times[l])
            print(f"spike distances:", (ref_firing_times[l][1:] - ref_firing_times[l][:-1])%300)

