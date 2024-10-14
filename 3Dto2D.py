from Bio.PDB import PDBParser
import numpy as np
import os
import torch
import re

def load_structure(file_path):
    parser = PDBParser()
    structure = parser.get_structure('protein', file_path)
    return structure

def extract_ca_atoms(structure):
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atoms.append(residue['CA'])
    return ca_atoms

def compute_distance_matrix(ca_atoms):
    num_atoms = len(ca_atoms)
    distance_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            distance = ca_atoms[i] - ca_atoms[j]
            distance_matrix[i][j] = distance_matrix[j][i] = distance
    return distance_matrix

def create_contact_map(distance_matrix, threshold=8.0):
    return (distance_matrix < threshold).astype(int)

def pad_tensor1(tensor, pad_size=(1024, 1024), pad_value=0):
    padding = [0, pad_size[1] - tensor.size(1), 0, pad_size[0] - tensor.size(0)]
    return torch.nn.functional.pad(tensor, padding, value=pad_value)
def pad_tensor2(tensor, pad_size=(1024, 1024), pad_value=0):
    padding = [0, pad_size[1] - tensor.size(1), 0, pad_size[0] - tensor.size(0)]
    return torch.nn.functional.pad(tensor, padding, value=pad_value)

def sorted_pdb_files(directory):
    files = os.listdir(directory)
    filtered_files = []
    for file in files:
        match = re.match(r'(\d+)_', file)
        if match:
            filtered_files.append((file, int(match.group(1))))
        else:
            print(f"Skipping file due to naming mismatch: {file}")
    return sorted(filtered_files, key=lambda x: x[1])

def process_pdb_files(directory):
    all_distances = {}
    all_contacts = {}
    file_index = 0  # Initialize file index
    for file_name, _ in sorted_pdb_files(directory):
        if file_name.endswith('.pdb'):
            file_path = os.path.join(directory, file_name)
            try:
                structure = load_structure(file_path)
                ca_atoms = extract_ca_atoms(structure)
                distance_matrix = compute_distance_matrix(ca_atoms)
                contact_map = create_contact_map(distance_matrix)

                # Convert to PyTorch tensors
                distance_tensor = torch.tensor(distance_matrix)
                contact_tensor = torch.tensor(contact_map, dtype=torch.int32)

                # Pad tensors
                distance_tensor_padded = pad_tensor1(distance_tensor, pad_size=(1024, 1024))
                contact_tensor_padded = pad_tensor2(contact_tensor, pad_size=(1024, 1024))

                all_distances[file_index] = distance_tensor_padded.numpy()
                all_contacts[file_index] = contact_tensor_padded.numpy()
                print(f"{file_index}: Processed {file_name}")
                np.savez('data/node/davis_node.npz', dict=all_distances)
                np.savez('data/edge/kiba_edge.npz', dict=all_contacts)
                file_index += 1
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")

#     # Save all distance and contact maps to two npz files
#     np.savez('data/node/davis_node.npz', **all_distances)
#     np.savez('data/edge/davis_edge.npz', **all_contacts)

# Call function to process the directory of PDB files
process_pdb_files('data/davis')
