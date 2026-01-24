#!/usr/bin/env python
import os
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
from typing import Optional, Tuple, Union
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation 

################ customized parameters #################
################ please modify them based on your dataset #################
# dataset: 
DATASET_ODIR = "/home/xzt/data/semantic_lidar_v2/2024-04-04-12-16-41"  # the directory path of the raw data
DATASET_NAME = "train" # select the train, dev, and test 
# map: parameters from the map configuration file
MAP_ORIGIN = np.array([-21.200000, -34.800000, 0.000000]) 
MAP_RESOLUTION = 0.025000
# labeling:
MAP_LABEL_PATH = '../manually_labeling/labelme_output/label.png'
MAP_PATH = '../manually_labeling/labelme_output/img.png'

# Hokuyo UTM-30LX-EW:
POINTS = 1081 # the number of lidar points
AGNLE_MIN = -2.356194496154785
AGNLE_MAX = 2.356194496154785
RANGE_MAX = 60.0
# urdf: laser_joint
JOINT_XYZ = [-0.12, 0.0, 0.0]
JOINT_RPY = [0.0, 0.0, 0.0]

# # WLR-716:
# POINTS = 811 # the number of lidar points
# AGNLE_MIN = -2.356194496154785
# AGNLE_MAX = 2.356194496154785
# RANGE_MAX = 25.0
# # urdf: laser_joint
# JOINT_XYZ = [0.065, 0.0, 0.182]
# JOINT_RPY = [3.1415926, 0.0, 0.0]
# # RPLIDAR-S2:
# POINTS = 1972 # the number of lidar points
# AGNLE_MIN = -3.1415927410125732
# AGNLE_MAX = 3.1415927410125732
# RANGE_MAX = 16.0
# # urdf: laser_joint
# JOINT_XYZ = [0.065, 0.0, 0.11]
# JOINT_RPY = [0.0, 0.0, 3.1415926]

################# read dataset ###################
NEW_LINE = "\n"
class Semantic2DLidarDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, file_name):
        # initialize the data and labels
        # read the names of image data:
        self.scan_file_names = []
        self.line_file_names = []
        self.pos_file_names = []
        self.vel_file_names = []
        # open train.txt or dev.txt:
        fp_file = open(img_path+'/'+file_name+'.txt', 'r')
        # for each line of the file:
        for line in fp_file.read().split(NEW_LINE):
            if('.npy' in line):
                self.scan_file_names.append(img_path+'/scans_lidar/'+line)
                self.line_file_names.append(img_path+'/line_segments/'+line)
                self.pos_file_names.append(img_path+'/positions/'+line)
                self.vel_file_names.append(img_path+'/velocities/'+line)
        # close txt file:
        fp_file.close()
        self.length = len(self.scan_file_names)

        print("dataset length: ", self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        scan = np.zeros(POINTS)
        position = np.zeros(3)
        vel = np.zeros(2)
        # get the scan data:
        scan_name = self.scan_file_names[idx]
        scan = np.load(scan_name)

        # get the line segments data:
        line_name = self.line_file_names[idx]
        line_segs = np.load(line_name)
       
        # get the scan_ur data:
        pos_name = self.pos_file_names[idx]
        position = np.load(pos_name)

        # get the velocity data:
        vel_name = self.vel_file_names[idx]
        vel = np.load(vel_name)
        
        # initialize:
        scan[np.isnan(scan)] = 0.
        scan[np.isinf(scan)] = 0.
        scan[scan==60] = 0.

        position[np.isnan(position)] = 0.
        position[np.isinf(position)] = 0.

        vel[np.isnan(vel)] = 0.
        vel[np.isinf(vel)] = 0.

        # transfer to pytorch tensor:
        scan_tensor = torch.FloatTensor(scan)
        line_tensor = torch.FloatTensor(line_segs)
        pose_tensor = torch.FloatTensor(position)
        vel_tensor =  torch.FloatTensor(vel)

        data = {
                'scan': scan_tensor,
                'line': line_tensor,
                'position': pose_tensor,
                'velocity': vel_tensor, 
                }

        return data

################# ICP algorithm ###################
class MapICP:
    def __init__(self, map_pts):
        # map_pts: 2xN numpy.ndarray of points
        self.map_pts = map_pts
        self.neighbors = NearestNeighbors(n_neighbors=1).fit(self.map_pts.T)

    def setMapPoints(self, pts: np.ndarray) -> None:
        '''
        Initializes a set of points to match against
        Inputs:
            pts: 2xN numpy.ndarray of 2D points
        '''
        assert pts.shape[0] == 2
        self.map_pts = pts
        self.neighbors = NearestNeighbors(n_neighbors=1).fit(pts.T)

    def nearestNeighbor(self, pts: np.ndarray) -> Tuple[np.array, np.array]:
        '''
        Find the nearest (Euclidean) neighbor in for each point in pts
        Input:
            pts: 2xN array of points
        Output:
            distances: np.array of Euclidean distances of the nearest neighbor
            indices: np.array of indices of the nearest neighbor
        '''
        assert pts.shape[0] == 2
        distances, indices = self.neighbors.kneighbors(pts.T, return_distance=True)

        return distances.ravel(), indices.ravel()

    def bestFitTransform(self, pts: np.array, map_pts: np.array) -> np.ndarray: #indices: np.array) -> np.ndarray:
        '''
        Calculates the least-squares best-fit transform that maps pts on to map_pts
        Input:
            pts: 2xN numpy.ndarray of points
            # indices: 1xN numpy.array of corresponding map_point indices
            map_pts: 2xN numpy.ndarray of points
        Returns:
            T: 3x3 homogeneous transformation matrix that maps pts on to map_pts
        '''
        # get number of dimensions
        m = pts.shape[0]
        assert m == 2

        # Extract points
        map = map_pts # self.map_pts[:,indices]
        assert pts.shape == map.shape

        # translate points to their centroids
        centroid_pts = np.mean(pts, axis=1)
        centroid_map = np.mean(map, axis=1)
        PTS = pts - centroid_pts.reshape((-1,1))
        MAP = map - centroid_map.reshape((-1,1))

        # rotation matrix
        H = MAP @ PTS.T
        U, _, Vt = np.linalg.svd(H)
        R = U @ Vt

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[:,-1] *= -1
            R = U @ Vt

        # translation
        t = centroid_map - R @ centroid_pts

        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        return T

    def icp(self, pts: np.ndarray, init_pose: Optional[Union[np.array, None]]=None, max_iterations: Optional[int]=20, tolerance: Optional[float]=0.05) -> Tuple[np.ndarray, np.array, int]:
        '''
        The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
        Input:
            pts: 2xN numpy.ndarray of source points
            init_pose: 3x3 homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
        Outputs:
            T: final homogeneous transformation that maps pts on to map_pts
            distances: Euclidean distances (errors) of the nearest neighbor
            i: number of iterations to converge
        '''
        # Get number of dimensions
        m = pts.shape[0]
        assert m == 2

        # Make points homogeneous, copy them to maintain the originals
        src = np.ones((m+1, pts.shape[1]))
        dst = np.ones((m+1, self.map_pts.shape[1]))
        src[:m,:] = np.copy(pts)
        dst[:m,:] = np.copy(self.map_pts)

        # Apply the initial pose estimate
        T = np.eye(3)
        if init_pose is not None:
            src = init_pose @ src
            T = init_pose @ T

        # Run ICP
        prev_error = 1e6
        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            distances, indices = self.nearestNeighbor(src[:m,:])
            map_pts = self.map_pts[:, indices]
            # compute the transformation between the current source and nearest destination points
            T_delta = self.bestFitTransform(src[:m,:], map_pts) #indices)

            # update the current source and transform
            src = T_delta @ src
            T = T_delta @ T

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        #T = self.bestFitTransform(pts, src[:m,:]) 
        # Calculate final transformation
        return T, distances, i

################# map converter: homogeneous transformation  ###################
class MapConverter:
    # Constructor
    def __init__(self, label_file, origin=np.array([-12.200000, -12.200000, 0.000000]), resolution=0.05):
        # map parameters
        self.origin = origin 
        self.resolution = resolution 
        # homogeneous transformation matrix:
        self.map_T_pixel = np.array([[np.cos(self.origin[2]), -np.sin(self.origin[2]), self.origin[0]],
                                [np.sin(self.origin[2]),  np.cos(self.origin[2]), self.origin[1]],
                                [0, 0, 1]
                                ])
        self.pixel_T_map = np.linalg.inv(self.map_T_pixel)
        # load semantic map
        label_map = np.asarray(PIL.Image.open(label_file))
        self.label_map = label_map.T # transpose 90 
        self.x_lim = self.label_map.shape[0]
        self.y_lim = self.label_map.shape[1]
        print(self.x_lim, self.y_lim)
    
    def coordinate2pose(self, px, py):
        pixel_pose = np.array([px*self.resolution, py*self.resolution, 1])
        map_pose = np.matmul(self.map_T_pixel, pixel_pose.T)
        x = map_pose[0]
        y = map_pose[1]

        return x, y

    def pose2coordinate(self, x, y):
        map_pose = np.array([x, y, 1])
        pixel_pose = np.matmul(self.pixel_T_map, map_pose.T)
        px = int(pixel_pose[0] / self.resolution)
        py = int(pixel_pose[1] / self.resolution)

        return px, py

    def get_semantic_label(self, x, y):
        px, py = self.pose2coordinate(x, y)
        py = self.y_lim - py # the y axis is inverse
        label_loc = np.zeros(10) # 10 labels
        # filtering:
        for i in range(px-2, px+2):
            for j in range(py-2, py+2):
                if(i >= 0 and i < self.x_lim and j >=0 and j < self.y_lim):
                    label = self.label_map[i, j]
                    if(label != 0):
                        label_loc[label] += 1

        if(np.sum(label_loc) == 0): # people
            semantic_label = 4
        else:   # static objects
            semantic_label = np.argmax(label_loc)
       
        return semantic_label
    
    def transform_lidar_points(self, points_sensor, xyz, rpy, inverse=True):
        """
        Transform 2D lidar points from sensor frame to robot base frame, given URDF translation and RPY.
        Args:
            points_sensor (Nx2 ndarray): Lidar points [x, y] in the sensor frame.
            translation (3-tuple/list/ndarray): [x, y, z] translation from base to sensor (URDF).
            rpy (3-tuple/list/ndarray): [roll, pitch, yaw] rotation from base to sensor (URDF, radians).
            inverse (bool): If True (default), transforms from sensor to base frame (i.e., applies inverse transform).
                            If False, transforms from base to sensor frame.
        Returns:
            Nx2 ndarray: Transformed points in base frame.
        """
        translation = np.asarray(xyz).reshape((3,))
        rpy = np.asarray(rpy).reshape((3,))
        
        rot = Rotation.from_euler('xyz', rpy)  # Roll, Pitch, Yaw
        
        N = points_sensor.shape[0]
        # Add z=0 to points
        points_3d = np.hstack([points_sensor, np.zeros((N, 1))])  # Nx3

        if inverse:
            # Transform from sensor -> base: x_base = R.T @ (x_sensor - t)
            points_shifted = points_3d - translation
            points_base = rot.inv().apply(points_shifted)
        else:
            # Transform from base -> sensor: x_sensor = R @ x_base + t
            points_base = rot.apply(points_3d) + translation

        return points_base[:, :2]

    def lidar2map(self, lidar_pos, lidar_scan):
        # get laser points: polar to cartesian
        points_laser = np.zeros((POINTS, 2))
        angles = np.linspace(AGNLE_MIN, AGNLE_MAX, num=POINTS) #np.linspace(-(135*np.pi/180), 135*np.pi/180, num=POINTS)
        dis = lidar_scan
        points_laser[:, 0] = dis*np.cos(angles)
        points_laser[:, 1] = dis*np.sin(angles)
        #lidar_points[:, 2] = 1
        
        # coordinate transformation: lidar -> footprint
        lidar_points = self.transform_lidar_points(points_sensor=points_laser, xyz=JOINT_XYZ, rpy=JOINT_RPY)

        # coordinate transformation: footprint -> map
        lidar_points_in_map = np.zeros((POINTS, 2))
        lidar_points_in_map[:, 0] = lidar_points[:, 0]*np.cos(lidar_pos[2]) - lidar_points[:, 1]*np.sin(lidar_pos[2]) + lidar_pos[0]
        lidar_points_in_map[:, 1] = lidar_points[:, 0]*np.sin(lidar_pos[2]) + lidar_points[:, 1]*np.cos(lidar_pos[2]) + lidar_pos[1]

        return lidar_points_in_map, points_laser


if __name__ == '__main__':
    # input parameters:
    dataset_odir = DATASET_ODIR
    dataset_name = DATASET_NAME
    scan_label_odir = dataset_odir + "/" + "semantic_label"
    if not os.path.exists(scan_label_odir):
        os.makedirs(scan_label_odir)

    map_path = MAP_PATH
    map_label_path = MAP_LABEL_PATH

    map_origin = MAP_ORIGIN
    map_resolution = MAP_RESOLUTION

    # initialize semantic scan label:
    scan_label = np.zeros(POINTS)

    # initialize the map converter: homogeneous transformation  
    mc = MapConverter(map_label_path, origin=map_origin, resolution=map_resolution)

    ## extract the valid map points fromt the map image:
    # load map image:
    map_img = np.asarray(PIL.Image.open(map_path)) 
    map_img = map_img.T # transpose 90 
    print(map_img.shape)
    # get map valid points:
    map_idx = np.where(map_img == 0)
    map_idx_x = map_idx[0]
    map_idx_y = map_idx[1]
    map_points = []
    for n in range(len(map_idx_x)):
        px = map_idx_x[n]
        py = map_idx_y[n]
        py = map_img.shape[1] - py # the x axis is inverse
        [x, y] = mc.coordinate2pose(px, py)
        map_points.append([x, y])
    map_pts = np.array(map_points)

    # initialize ICP:
    icp = MapICP(map_pts.T)

    # load dataset:
    eval_dataset = Semantic2DLidarDataset(dataset_odir, dataset_name)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, \
                                                   shuffle=False, drop_last=True) #, pin_memory=True)
    
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(eval_dataset)/eval_dataloader.batch_size)
    for i, batch in tqdm(enumerate(eval_dataloader), total=num_batches):
        # collect the samples as a batch:
        scan = batch['scan']
        scan = scan.detach().cpu().numpy()
        lines = batch['line']
        lines = lines.detach().cpu().numpy()
        position = batch['position']
        position = position.detach().cpu().numpy()

        # transfer lidar points: 
        lidar_pos = position.reshape(3)
        lidar_pos[0] += LIDAR_BASE_DIS # the distance from lidar_mount to base_link
        lidar_scan = scan.reshape(POINTS)
        lidar_points_in_map, lidar_points = mc.lidar2map(lidar_pos, lidar_scan)

        # use line segments (key line features) to remove outliers in the lidar scan:
        correspondences = []
        for line in lines[0]:
            x1, y1, x2, y2 = line
            # construct line formular: y = ax +b
            a = (y2 - y1) / (x2 - x1 + 1e-10)
            b = y1 - a*x1
            for n in range(POINTS):
                x, y = lidar_points[n, :]
                if(x >= x1-1 and x <= x2+1 and y >= y1-1 and y <= y2+1): # in the area of the line
                    if(abs(y - (a*x+b)) <= 0.3): # on the line
                        correspondences.append([x, y])

        correspondences = np.array(correspondences)
        correspondences_length = len(correspondences)

        if(correspondences_length > 280): # reliable correspondences
            # coordinate transformation: lidar -> map
            correspondences_in_map = np.zeros((correspondences_length, 2))
            correspondences_in_map[:, 0] = correspondences[:, 0]*np.cos(lidar_pos[2]) - correspondences[:, 1]*np.sin(lidar_pos[2]) + lidar_pos[0]
            correspondences_in_map[:, 1] = correspondences[:, 0]*np.sin(lidar_pos[2]) + correspondences[:, 1]*np.cos(lidar_pos[2]) + lidar_pos[1]
            
            # use ICP scan matching to correct lidar pose:
            mapc_T_map, _, _ = icp.icp(correspondences_in_map.T, max_iterations=500, tolerance=1e-6)

            # corrected lidar pose:
            lidar_pose_corrected = np.matmul(mapc_T_map, np.array([lidar_pos[0], lidar_pos[1], 1]))
            lidar_points_in_map_old =  np.concatenate((lidar_points_in_map, np.ones((POINTS, 1))), axis=1)
            point_corrected = np.matmul(mapc_T_map, lidar_points_in_map_old.T)
        else: # no icp correction
            lidar_pose_corrected = lidar_pos
            point_corrected = lidar_points_in_map.T

        # semantic pixel mataching:
        for j in range(POINTS):
            if(point_corrected[0, j] == lidar_pose_corrected[0] and point_corrected[1, j] == lidar_pose_corrected[1]): # scan == 0
                scan_label[j] = 0
            else:
                scan_label[j] = mc.get_semantic_label(point_corrected[0, j], point_corrected[1, j])

        # write scan_label in lidar frame into np.array:
        scan_label_name = scan_label_odir + "/" + str(i).zfill(7) 
        np.save(scan_label_name, scan_label)
