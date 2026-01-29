import os
import numpy as np
import csv
import yaml
import random
import sys
import jittor as jt


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        jt.set_global_seed(seed)
    except Exception:
        try:
            jt.set_seed(seed)
        except Exception:
            pass


def setup_device(gpu_id):
    try:
        has_cuda = bool(jt.has_cuda)
    except Exception:
        has_cuda = False
    if has_cuda:
        jt.flags.use_cuda = 1
        if hasattr(jt.flags, "gpu"):
            jt.flags.gpu = gpu_id
        elif hasattr(jt.flags, "cuda_device_id"):
            jt.flags.cuda_device_id = gpu_id
        print(f"Jittor 使用 CUDA，GPU id: {gpu_id}")
    else:
        jt.flags.use_cuda = 0
        print("Jittor 使用 CPU")
    return jt.flags.use_cuda


def normalize_val_only_metric(val_only_metric):
    if val_only_metric is None:
        metrics = ["average"]
    elif isinstance(val_only_metric, str):
        metrics = [val_only_metric]
    else:
        metrics = list(val_only_metric)
    seen = set()
    out = []
    for metric in metrics:
        if metric is None:
            continue
        metric = str(metric).lower()
        if metric and metric not in seen:
            out.append(metric)
            seen.add(metric)
    return out or ["average"]


def try_make_dir(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        print("Folder was created by another process")


def vit_normalization(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "name" and value == "normalize":
                data["mean"] = [0.5, 0.5, 0.5]
                data["std"] = [0.5, 0.5, 0.5]
            else:
                vit_normalization(value)
    elif isinstance(data, list):
        for item in data:
            vit_normalization(item)


def generate_points(range_tuple, points, log_scale=False):
    start, end = range_tuple
    if log_scale:
        if start == 0:
            start = sys.float_info.min
        points = np.logspace(np.log10(start), np.log10(end), points)
    else:
        points = np.linspace(start, end, points)
    return list(points)


def add_free_log(data, save_dir):
    final_data = []
    for i, col in enumerate(data):
        if col != []:
            final_data.append(data[i])
    rows = zip(*final_data)
    try:
        with open(save_dir, "a", newline="") as file:
            writer = csv.writer(file)
            for row in rows:
                writer.writerow(row)
        return ()
    except PermissionError:
        return ()


def report_time(seconds):
    seconds = round(seconds)
    minutes = seconds // 60
    seconds = seconds % 60
    hours = minutes // 60
    minutes = minutes % 60
    return f"Hours: {hours} Min: {minutes} Sec: {seconds}"


def float_tuple(value):
    try:
        if isinstance(value, float):
            value = (value, value)
        elif isinstance(value, str):
            value = (float(value), float(value))
        elif isinstance(value, list):
            if len(value) < 2:
                value = (float(value[0]), float(value[0]))
            else:
                value = tuple([float(i) for i in value])
        return value
    except ValueError:
        print("input cannot change to float")


def dirlist(dirs):
    if dirs == "None":
        return None, None

    if isinstance(dirs, str):
        dirs = [dirs]
    images = []
    classes = []
    for current_dir in dirs:
        cache_file = current_dir + ".npy"
        csv_mtime = os.path.getmtime(current_dir) if os.path.exists(current_dir) else 0
        cache_mtime = os.path.getmtime(cache_file) if os.path.exists(cache_file) else 0
        if os.path.exists(cache_file) and cache_mtime >= csv_mtime:
            data = np.load(cache_file, allow_pickle=True).item()
            images.extend(data["images"])
            classes.extend(data["classes"])
        else:
            with open(current_dir, mode="r", encoding="utf-8") as file:
                for line in file:
                    if line != "":
                        if len(line.split()) > 1:
                            images.append(line.split()[0])
                            classes.append(str(line.split()[1]))
            data = {"images": images.copy(), "classes": classes.copy()}
            np.save(cache_file, data)
    return images, classes


def save_architecture(network, direct, name="architecture"):
    file_path = os.path.join(direct, f"{name}.txt")
    if os.path.exists(file_path):
        return
    with open(file_path, "w") as file:
        print(network, file=file)
        print("", file=file)
        try:
            for param_name, param in network.named_parameters():
                print(param_name, param.requires_grad, file=file)
        except Exception:
            pass


def save_model(network, optimizer, epoch, direct, is_nan=False):
    if is_nan:
        state = {"epoch": None, "state_dict": None, "optimizer": None}
    else:
        try:
            opt_state = optimizer.state_dict()
        except Exception:
            opt_state = None
        state = {
            "epoch": epoch,
            "state_dict": network.state_dict(),
            "optimizer": opt_state,
        }
    jt.save(state, direct)
    return


def load_model(network, model_location):
    state = jt.load(model_location)
    if state.get("state_dict") is not None:
        network.load_state_dict(state["state_dict"])
    network.eval()


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def nan_in_grad(model):
    # Gracefully scan gradients for NaNs without spamming errors when grads are missing
    found_nan = False
    try:
        params = model.backbone.parameters()
    except Exception:
        params = []
    for param in params:
        try:
            if hasattr(param, "is_stop_grad") and param.is_stop_grad():
                continue
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            if jt.isnan(grad).any():
                found_nan = True
                break
        except Exception:
            continue
    return found_nan


def save_yaml(structure, direct):
    with open(direct, "w") as file:
        documents = yaml.dump(structure, file, width=10000, Dumper=NoAliasDumper)


def read_yaml(direct):
    with open(direct) as file:
        structure = yaml.full_load(file)
    return structure


def initialize_csv_file(
    loader_info, csv_file_name, test_idx_list, domains, val_only_metric=None
):
    if not os.path.isfile(csv_file_name):
        metrics = normalize_val_only_metric(val_only_metric)
        metric_labels = {
            "average": "Imgaug_average",
            "worst": "Imgaug_worst",
            "cvar": "Imgaug_cvar",
        }
        metric_headers = []
        for metric in metrics:
            label = metric_labels.get(metric)
            if label is None:
                continue
            metric_headers.extend(
                [[f"{label}_{name}"] for name in loader_info["output_names_val"]]
            )
        headers = (
            [["lr"]]
            + [["method_loss"]]
            + [[name] for name in loader_info["output_names_val"]]
            + metric_headers
            + [[f"test_{name}"] for name in loader_info["output_names_val"]]
            + [
                [f"{val}_{name}" for name in loader_info["output_names_val"]]
                for val in loader_info["pd_names_val_only"]
            ]
            + [
                [f"{domains[idx]}_{name}" for name in loader_info["output_names_val"]]
                for idx in test_idx_list
            ]
        )
        add_free_log(
            data=[[item] for sublist in headers for item in sublist],
            save_dir=csv_file_name,
        )
