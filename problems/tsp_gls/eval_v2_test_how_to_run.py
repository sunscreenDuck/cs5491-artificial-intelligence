import numpy as np
import logging
from gen_inst import TSPInstance, load_dataset, dataset_conf
from gls import guided_local_search
from tqdm import tqdm

try:
    from gpt import heuristics_v2 as heuristics
except:
    from gpt import heuristics

perturbation_moves = 30
iter_limit = 1200


def calculate_cost(inst: TSPInstance, path: np.ndarray) -> float:
    return inst.distmat[path, np.roll(path, 1)].sum().item()


def solve(inst: TSPInstance) -> float:
    heu = heuristics(inst.distmat.copy())
    assert tuple(heu.shape) == (inst.n, inst.n)
    result = guided_local_search(inst.distmat, heu, perturbation_moves, iter_limit)
    # print(result)
    return calculate_cost(inst, result)


#
# if __name__ == "__main__":
#     import sys
#     import os
#
#     print("[*] Running ...")
#
#  # process = subprocess.Popen(['python', '-u', f'{self.root_dir}/problems/{self.problem}/eval.py', f'{self.problem_size}', self.root_dir, "train"],
#                                         # stdout=f, stderr=f)
#     problem_size = int(200)
#     mood = "train"
#     assert mood in ['train', 'val', "test"]
#
#     basepath = os.path.dirname(__file__)
#     # automacially generate dataset if nonexists
#     if not os.path.isfile(os.path.join(basepath, f"dataset/train{dataset_conf['train'][0]}_dataset.npy")):
#         from gen_inst import generate_datasets
#
#         generate_datasets()
#
#     if mood == 'train':
#         dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npy")
#         dataset = load_dataset(dataset_path)
#         n_instances = dataset[0].n
#
#         print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
#
#         objs = []
#         for i, instance in enumerate(dataset):
#             obj = solve(instance)
#             print(f"[*] Instance {i}: {obj}")
#             objs.append(obj)
#
#         print("[*] Average:")
#         print(np.mean(objs))
#
#     else:  # mood == 'val'
#         for problem_size in dataset_conf['val']:
#             dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npy")
#             dataset = load_dataset(dataset_path)
#             n_instances = dataset[0].n
#             logging.info(f"[*] Evaluating {dataset_path}")
#
#             objs = []
#             for i, instance in enumerate(tqdm(dataset)):
#                 obj = solve(instance)
#                 objs.append(obj)
#
#             print(f"[*] Average for {problem_size}: {np.mean(objs)}")


if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")
    basepath = os.path.dirname(__file__)
    _tsp_file = os.path.join(basepath, f"tsp/lin105.tsp")
    coordinates = []
    with open(_tsp_file, 'r') as file:
        for line in file:
            parts = line.split()
            coordinates.append((int(parts[1]), int(parts[2])))
    nd_array = np.array(coordinates)

    npy_fp = r'./tsp_npy/lin105.npy'
    np.save(npy_fp, nd_array)
    data = np.load(npy_fp)
    dataset = [TSPInstance(data)]

    for i, instance in enumerate(dataset):
        obj = solve(instance)
        print(obj)