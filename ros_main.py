#!/usr/bin/env python3

from os import O_TRUNC
import rospy
import ros_numpy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pcl
import open3d as o3d
from PEC import PEC
import torch
import data_tools

def cloud_callBack(cloud_msg):
    global cloud_state
    cloud_state = cloud_msg

def filter_cloud(cloud):
    #Remove Gripper from field of view
    passthrough = cloud.make_passthrough_filter()
    passthrough.set_filter_field_name("z")
    passthrough.set_filter_limits(0.4, 1)
    cloud_filtered = passthrough.filter()
    """
    #Downscale Cloud
    sor = cloud_filtered.make_voxel_grid_filter()
    sor.set_leaf_size(0.015, 0.015, 0.015)
    cloud_down = sor.filter()
    """
    #Remove table
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(0.02)
    indices, model = seg.segment()

    extract = cloud_filtered.extract(indices, negative=True)

    #Remove noise
    fil = extract.make_statistical_outlier_filter()
    fil.set_mean_k (50)
    fil.set_std_dev_mul_thresh (1.0)
    cloud_outlined = fil.filter()
    
    return cloud_outlined
    
    #return cloud_filtered

def main():

    rospy.init_node('listener', anonymous=True)
    sub_cloud = rospy.Subscriber("/camera/depth/points", PointCloud2, cloud_callBack)
    #class_names = ['ape', 'bleach', 'can', 'cat', 'clamp', 'drill', 'duck', 'eggbox', 'glue', 'tomato']
    class_names = ['mug', 'tuna', 'controller', 'bowl', 'mustard']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    path = "/home/daniel/Cloud_Classi/modelPEC.ckpt"
    model = PEC()
    net = model.load_from_checkpoint(path).to(device)
    net.freeze()

    while not rospy.is_shutdown():
        rate = rospy.Rate(1)
        cloud_global = "cloud_state" in globals()
    
        if cloud_global:
            #Get cloud in PCL format
            pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud_state, remove_nans=True)
            cloud=pcl.PointCloud(np.array(pc, dtype=np.float32))
            
            cloud_filterd = filter_cloud(cloud)

            cloud_array = np.asarray(cloud_filterd)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud_array)
            labels = np.array(
                    pcd.cluster_dbscan(eps=0.05, min_points=1000, print_progress=True))
            print(labels)

            clusters = []
            for i in range(labels.max() + 1):
                indice = np.where(labels == i)
                clusters.append(cloud[indice])


            for i in range(labels.max() + 1):
                predicted = net.predict(clusters[i], device, class_names)
                print(predicted)
                print(class_names[predicted[0]])
            #o3d.visualization.draw_geometries([cloudv])

        rate.sleep()



if __name__ == '__main__':
    
	main()