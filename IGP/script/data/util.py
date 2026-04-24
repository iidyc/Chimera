import numpy as np
import os
from os import listdir
from os.path import isfile, join
import re
import numpy as np

from script.data import dataset_io


CPU_ONLY_TARGETS = {"IGP", "BruteForce"}
EXECUTABLE_TARGETS = {"IGPGPUPP"}


def paper_n_centroid(n_vec: int) -> int:
    return max(1, int(16 * np.sqrt(n_vec)))


def _configure_build(build_path, build_type: str, use_cuda: bool):
    os.makedirs(build_path, exist_ok=True)
    source_root = dataset_io.project_root()
    use_cuda_flag = "ON" if use_cuda else "OFF"
    os.system(
        f'cmake -S {source_root} -B {build_path} '
        f'-DCMAKE_BUILD_TYPE={build_type} -DUSE_CUDA={use_cuda_flag}'
    )


def _copy_module_artifact(build_path, script_target_dir, module_name: str):
    if module_name in EXECUTABLE_TARGETS:
        return
    module_glob = f'{module_name}.cpython-*.so'
    os.system(f'cp {build_path}/{module_glob} {script_target_dir}/')


def compile_file(username: str, module_name: str, is_debug: bool = False, move_path='data'):
    del username
    build_path = dataset_io.project_root() / 'build'
    build_type = 'Debug' if is_debug else 'Release'
    script_target_dir = dataset_io.project_root() / 'script' / move_path
    use_cuda = module_name not in CPU_ONLY_TARGETS
    _configure_build(build_path=build_path, build_type=build_type, use_cuda=use_cuda)
    os.system(f'cmake --build {build_path} --target {module_name} -j')
    _copy_module_artifact(build_path=build_path, script_target_dir=script_target_dir, module_name=module_name)


def compile_file_batch_module(username: str, module_name_l: list, is_debug: bool = False, move_path='data'):
    del username
    build_path = dataset_io.project_root() / 'build'
    build_type = 'Debug' if is_debug else 'Release'
    script_target_dir = dataset_io.project_root() / 'script' / move_path
    use_cuda = any(module_name not in CPU_ONLY_TARGETS for module_name in module_name_l)
    _configure_build(build_path=build_path, build_type=build_type, use_cuda=use_cuda)
    targets = " ".join(module_name_l)
    os.system(f'cmake --build {build_path} --target {targets} -j')
    for module_name in module_name_l:
        _copy_module_artifact(build_path=build_path, script_target_dir=script_target_dir, module_name=module_name)


def item_vecs_in_chunk(vecs_l: np.ndarray, itemlen_l: np.ndarray, itemID: int):
    vecs_start_idx = int(np.sum(itemlen_l[:itemID]))
    n_item_vecs = int(itemlen_l[itemID])
    item_vecs = vecs_l[vecs_start_idx: vecs_start_idx + n_item_vecs]
    return item_vecs


def get_n_chunk(base_dir: str):
    filename_l = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]

    doclen_patten = r'doclens(.*).npy'
    embedding_patten = r'encoding(.*)_float32.npy'

    match_obj_l = [re.match(embedding_patten, filename) for filename in filename_l]
    match_chunkID_l = np.array([int(_.group(1)) if _ else None for _ in match_obj_l])
    match_chunkID_l = match_chunkID_l[match_chunkID_l != np.array(None)]
    assert len(match_chunkID_l) == np.sort(match_chunkID_l)[-1] + 1
    return len(match_chunkID_l)


def get_DEFAULT_SIZE(username: str, dataset: str):
    del username
    for itemlen_l_chunk, _ in dataset_io.iter_embedding_chunks("", dataset):
        return len(itemlen_l_chunk)
    raise ValueError(f"no embedding chunks available for {dataset}")
