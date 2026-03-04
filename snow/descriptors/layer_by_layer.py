import csv
import os
import numpy as np
import matplotlib.pyplot as plt


"DA SISTEMARE"

# ============================================================
# Layer analysis (numpy-based)
# ============================================================

def cut_layers_from_frame(
    coords_frame: np.ndarray,
    elements: np.ndarray,
    lattice_parameter: float,
    species_A: str,
    species_B: str,
):
    """
    Cuts a single frame into layers using z-coordinates.

    Parameters
    ----------
    coords_frame : np.ndarray
        Shape (n_atoms, 3)
    elements : np.ndarray
        Shape (n_atoms,)
    """

    z = coords_frame[:, 2]
    min_z = z.min()
    max_z = z.max()

    n_layers = int((max_z - min_z) / lattice_parameter) + 1

    layer_info = []

    for i in range(n_layers):
        z_min = min_z + i * lattice_parameter
        z_max = min_z + (i + 1) * lattice_parameter

        mask = (z >= z_min) & (z < z_max)

        tot = np.count_nonzero(mask)
        n_A = np.count_nonzero(mask & (elements == species_A))
        n_B = np.count_nonzero(mask & (elements == species_B))

        layer_info.append((i + 1, tot, n_A, n_B))

    return layer_info


# ============================================================
# CSV I/O
# ============================================================

def save_layer_info_to_csv(
    coords: np.ndarray,
    elements: np.ndarray,
    lattice_parameter: float,
    species_A: str,
    species_B: str,
    output_file: str,
):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Layer", "Total_Atoms", "Atoms_A", "Atoms_B"])

        for frame_idx, frame_coords in enumerate(coords, start=1):
            layer_info = cut_layers_from_frame(
                frame_coords,
                elements,
                lattice_parameter,
                species_A,
                species_B,
            )

            for layer, tot, a, b in layer_info:
                writer.writerow([frame_idx, layer, tot, a, b])


def load_layer_info_from_csv(csv_file: str):
    layers_info_per_frame = []

    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)

        current_frame = None
        current_info = []

        for row in reader:
            frame, layer, tot, a, b = map(int, row)

            if current_frame is None:
                current_frame = frame

            if frame != current_frame:
                layers_info_per_frame.append(current_info)
                current_info = []
                current_frame = frame

            current_info.append((layer, tot, a, b))

        if current_info:
            layers_info_per_frame.append(current_info)

    return layers_info_per_frame


def save_max_layers_to_csv(
    coords: np.ndarray,
    lattice_parameter: float,
    output_file: str,
):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "N_Layers"])

        for frame_idx, frame_coords in enumerate(coords, start=1):
            z = frame_coords[:, 2]
            n_layers = int((z.max() - z.min()) / lattice_parameter) + 1
            writer.writerow([frame_idx, n_layers])


# ============================================================
# Plotting
# ============================================================

def plot_layer_statistics(
    layers_info_per_frame,
    output_prefix: str,
    frames_to_plot,
):
    max_layers = max(len(f) for f in layers_info_per_frame)
    max_atoms = max(info[1] for f in layers_info_per_frame for info in f)

    for frame_idx in frames_to_plot:
        if frame_idx - 1 >= len(layers_info_per_frame):
            continue

        layer_info = layers_info_per_frame[frame_idx - 1]
        layers = [x[0] for x in layer_info]
        atoms = [x[1] for x in layer_info]

        plt.figure()
        plt.plot(layers, atoms, marker="o")
        plt.xlabel("Layer")
        plt.ylabel("Number of Atoms")
        plt.title(f"Atoms per Layer (Frame {frame_idx})")
        plt.grid()
        plt.xlim(1, max_layers)
        plt.ylim(0, max_atoms)

        plt.savefig(
            f"{output_prefix}_layer_statistics_frame_{frame_idx}.png"
        )
        plt.close()


def compute_and_plot_avg_std(
    base_path: str,
    metal: str,
    dir_name: str,
    max_layers_csv: str,
):
    sim_root = os.path.join(base_path, metal, dir_name)
    frames_data = {}

    for sim in os.listdir(sim_root):
        sim_path = os.path.join(sim_root, sim)
        if not os.path.isdir(sim_path):
            continue

        csv_path = os.path.join(sim_path, max_layers_csv)
        if not os.path.exists(csv_path):
            continue

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)

            for frame, n_layers in reader:
                frame = int(frame)
                n_layers = int(n_layers)
                frames_data.setdefault(frame, []).append(n_layers)

    frames = sorted(frames_data.keys())
    avg = [np.mean(frames_data[f]) for f in frames]
    std = [np.std(frames_data[f]) for f in frames]

    plt.figure(figsize=(10, 6))
    plt.errorbar(frames, avg, yerr=std, fmt="-o", capsize=4, alpha=0.6)
    plt.xlabel("Frame")
    plt.ylabel("N_Layers")
    plt.title(f"Average & Std of Max Layers ({dir_name})")
    plt.grid()

    out = os.path.join(
        sim_root, f"Avg_Std_Max_Layers_{dir_name}.png"
    )
    plt.savefig(out)
    plt.close()
