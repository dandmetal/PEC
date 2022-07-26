#from tokenize import group
import numpy as np
import random
#import open3d as o3d
import pcl
import math

def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def fps_calc(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_pts_idx = np.zeros(K)
    upper_bound = pts.shape[0] - 1
    first_idx = random.randint(0, upper_bound)
    #first_idx = 0
    farthest_pts[0] = pts[first_idx]
    farthest_pts_idx[0] = first_idx
    distances = calc_distances(farthest_pts[0, 0:3], pts[:, 0:3])
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        farthest_pts_idx[i] = np.argmax(distances)
        distances = np.minimum(distances, calc_distances(farthest_pts[i, 0:3], pts[:, 0:3]))
    return farthest_pts_idx.astype(np.int64)


def fps(pts, k = 256):
    indices = fps_calc(pts, k)
    sampled_cloud = np.empty([k, 3], dtype=np.float32)
    for i in range(k):
        sampled_cloud[i] = pts[indices[i]]
    return sampled_cloud.astype(np.float32)


def get_close_groups(pts, k = 64):
    pts = pts.astype(np.float32)
    array3d = pts
    pcd = pcl.PointCloud(pts)

    groups = []
    group_num = int(len(pts) / k)
    for i in range(group_num):

        upper_bound = array3d.shape[0] - 1
        #idx = random.randint(0, upper_bound)
        idx = 0
        searchPoint = pcd[idx]
        point = np.array([[searchPoint[0], searchPoint[1], searchPoint[2]]], dtype=np.float32)
        pc = pcl.PointCloud()
        pc.from_array(point)

        kd = pcd.make_kdtree_flann()
        indices, sqr_distances = kd.nearest_k_search_for_cloud(pc, k)
        array3d = np.asarray(pcd)
        group = np.take(array3d, indices=indices, axis=0)
        array3d = np.delete(array3d, indices, axis=0)
        if len(array3d) != 0:
            pcd = pcl.PointCloud(array3d)
        groups.append(group)

    return groups



def sort_groups(groups):
    size = len(groups)
    mean = np.empty(size)

    groups = np.concatenate(groups, axis=0)

    for i in range(size):
        group_mean = np.mean(groups[i], axis=0)
        d = math.sqrt(group_mean[0]*group_mean[0] + group_mean[1]*group_mean[1] + group_mean[2]*group_mean[2])
        mean[i] = d
    indices = np.argsort(mean)
    sorted_array = [groups[k] for k in indices]
    output = np.array(sorted_array)
    
    output = np.concatenate(output, axis=0)

    return output


def random_remove(pts, input_size = 256):
    pts = pts.astype(np.float32)
    size = len(pts)
    excess_size = size - input_size
    idx = random.randint(0, size - 1)
    pcd = pcl.PointCloud(pts)

    if excess_size < 0:
        print("input to smoll")
        exit()
    elif excess_size <= 180:
        num_remove = random.randint(0, excess_size)

    elif excess_size > 180:
        excess_size = int(size * 0.4)
        num_remove = random.randint(0, excess_size)
    #print("Tamanho execesso: ",excess_size)
    #print("Numero para remover: ",num_remove)
    #print("indice incial:", idx)

    if num_remove >= 0:
        return pts

    searchPoint = pts[idx]
    point = np.array([[searchPoint[0], searchPoint[1], searchPoint[2]]], dtype=np.float32)
    pc = pcl.PointCloud()
    pc.from_array(point)
    kd = pcd.make_kdtree_flann()
    indices, sqr_distances = kd.nearest_k_search_for_cloud(pc, num_remove)
    #print(indices)
    #print(num_remove)
    pts_removed = np.delete(pts, indices, axis=0)
    if (len(pts_removed) < 256):
        print("panik")
        exit()
    return pts_removed


def get_curvature(cloud, indices):

    points = np.asarray(cloud)
    M = np.array([ points[i] for i in indices[0] ]).T
    M = np.cov(M)
    
    # eigen decomposition
    V, E = np.linalg.eig(M)
    # h3 < h2 < h1
    h1, h2, h3= V

    curvature = h3 / (h1 + h2 + h3)
    return curvature

def get_curvature_points(pts, radius = 8): 
    cloud = pcl.PointCloud()
    cloud.from_array(pts)

    kd = cloud.make_kdtree_flann()
    curvatures = []
    if radius <= 2:
        radius = 3
    for i in range(cloud.size):
        searchPoint = cloud[i]
        points = np.array([[searchPoint[0], searchPoint[1], searchPoint[2]]], dtype=np.float32)
        pc= pcl.PointCloud()
        pc.from_array(points)
        indices, sqr_distances = kd.nearest_k_search_for_cloud(pc, radius)
        curv = get_curvature(cloud,indices)
        curvatures.append(curv)
    
    #print(pts)
    curvatures = np.array(curvatures)
    cloud4d = np.column_stack((pts, curvatures))
    #print(cloud4d)
    #print(curvatures)
    return cloud4d