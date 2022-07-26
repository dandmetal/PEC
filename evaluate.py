from typing import final
from PEC import PEC
import torch
import numpy as np
import sys
import os
#import open3d as o3d
import data_tools


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    actual = sys.path[0]
    class_name = "controller"
    dataset_dir = '/objs_isaac4/' + class_name
    final_path = actual + dataset_dir
    class_names = ['mug', 'tuna', 'controller', 'bowl', 'mustard']

    list_files = os.listdir(final_path)
    #print(list_files)
    path = "/home/daniel/Cloud_Classi/modelPEC.ckpt"
    model = PEC()
    net = model.load_from_checkpoint(path).to(device)
    net.freeze()
    sum = 0
    for i in range(len(list_files)):
        obj_localtion = final_path + "/" +list_files[i]
        pts = np.load(obj_localtion)
        predicted = net.predict(pts, device, class_names)
        print(class_names[predicted[0]])
        if class_names[predicted[0]] == class_name:
            sum = sum + 1
    print((sum/1000) * 100, "%", " correto")
        


if __name__ == '__main__':
    main()