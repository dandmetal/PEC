import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import os
#import open3d as o3d


class ReadSet(Dataset):
    def __init__(self, root_dir, device='cpu', type = None,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        self.inputs = []
        self.annotations = []
        list_files = os.listdir(root_dir)
        print(list_files)
        num_class = len(list_files)
        zero_class = np.zeros(num_class)
        for i in range(num_class):
            output = np.copy(zero_class)
            if type is None:
                obj_location = root_dir + "/" + list_files[i]
            elif type == "train":
                obj_location = root_dir + "/" + list_files[i] + "/" + "train"
            else:
                obj_location = root_dir + "/" + list_files[i] + "/" + "test"
            list_objs = os.listdir(obj_location)
            num_objs = len(list_objs)

            output[i] = i
            for j in range(num_objs):

                input_location = obj_location + "/" + list_objs[j]

                input_data = np.load(input_location)
                print("Reading and processing file ", input_location)
                #size_data = len(input_data)
                #input_data = data_tools.random_remove(input_data)
                #input_data = data_tools.fps(input_data,64)

                #mean = np.mean(input_data, axis=0)
                #input_data = input_data - mean

                #groups = data_tools.get_close_groups(input_data, 16)
                #input_data = data_tools.sort_groups(groups)
                #input_data = data_tools.get_curvature_points(input_data, 64)
               
                self.inputs.append(input_data)
                # self.annotations.append(output)
                self.annotations.append(i)
                #print("Done")
        print(list_files)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = torch.tensor(self.inputs[idx]).to(self.device)
        output_data = torch.tensor(self.annotations[idx]).to(self.device)
        sample_set = [input_data.float(), output_data]
        return sample_set


"""
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actual = sys.path[0]
    dataset_dir = '\dataset'
    final_path = actual + dataset_dir
    train_loader = ReadSet(final_path, device)
    dataloader = DataLoader(train_loader, batch_size=32,
                            shuffle=True, num_workers=0)


if __name__ == "__main__":
    main()
"""
