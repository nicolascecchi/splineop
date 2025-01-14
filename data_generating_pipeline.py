
from splineop import splineop as spop
import pickle
import numpy as np
from scipy import stats
import os

# Polynomes
n_points = 500
n_polys = 50

for delta in [5,10,20]:
    for poly_nb in range(50):
        for n_bkps in range(1,6):
            poly = spop.generate_pw_quadratic(n_bkps=n_bkps,
                                        n_points=1000,
                                        normalized=True,
                                        random_state=poly_nb,
                                        delta=delta,
                                        strategy="geq")
            with open(f"./data/delta{delta}/polynomes/polynome{poly_nb}_bkps{n_bkps}.pkl","wb") as file:
                pickle.dump(poly, file)
            

# Sampling data

sample_sizes = [500, 1000, 2000]
for delta in [5, 10, 20]:
    for sample_size in sample_sizes:
        x = np.linspace(start=0, stop=1, num=sample_size, endpoint=False)
        for n_bkps in range(1,6):
            place_holder = np.empty((50,sample_size))
            for poly_nb in range(50):
                with open(f"./data/delta{delta}/polynomes/polynome{poly_nb}_bkps{n_bkps}.pkl","rb") as file:
                    poly = pickle.load(file)
                    sample = poly(x)
                    place_holder[poly_nb] = sample
            np.savetxt(fname=f"./data/delta{delta}/signals/clean_signals/signals_{sample_size}points_{n_bkps}bkps.txt", 
                    X=place_holder,
                    delimiter=';')

# Add noise to data


sample_sizes = [500, 1000, 2000]
noise_std_list = [0.1, 0.13, 0.16]
for delta in [5, 10, 20]:
    for e,noise_std in enumerate(noise_std_list):
        for n_bkps in range(1,6):
            for n_samples in sample_sizes:
                data = np.loadtxt(f"./data/delta{delta}/signals/clean_signals/signals_{n_samples}points_{n_bkps}bkps.txt",
                delimiter=";")
                noise = stats.norm(loc=0, scale=noise_std).rvs(size=data.shape, random_state=n_bkps)
                noised_signal = data + noise
                np.savetxt(fname = f"./data/delta{delta}/signals/noise{e+1}/noised_{n_samples}points_{n_bkps}bkps.txt",
                    X=noised_signal,
                    delimiter=";")



