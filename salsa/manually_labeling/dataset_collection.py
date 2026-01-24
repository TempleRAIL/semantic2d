#!/usr/bin/env python
from random import choice
import rospy
import tf
# custom define messages:
from geometry_msgs.msg import Point, PoseStamped, Twist, TwistStamped, Pose, PoseArray, PoseWithCovarianceStamped
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from laser_line_extraction.msg import LineSegmentList, LineSegment
# python: 
import os
import numpy as np
import threading

################ customized parameters #################
################ please modify them based on your dataset #################
POINTS = 1081
DATA_PATH = "~/semantic2d_data/2024-04-11-15-24-29"

class DataCollection:
    # Constructor
    def __init__(self):
        
        # initialize data:  
        self.scan_lidar = np.zeros(POINTS)
        self.intensities_lidar = np.zeros(POINTS)
        self.lines = []
        self.vel_cmd = np.zeros(2)
        self.pos = np.zeros(3)
        self.record = True
        self.rosbag_cnt = 0
        self.rosbag_cnt_reg = np.array([-4, -3, -2, -1])
        self.reg_cnt = 0
        
        # store directory:
        data_odir = DATA_PATH
        self.scan_lidar_odir = data_odir + "/" + "scans_lidar"
        if not os.path.exists(self.scan_lidar_odir):
            os.makedirs(self.scan_lidar_odir)
        self.intensities_lidar_odir = data_odir + "/" + "intensities_lidar"
        if not os.path.exists(self.intensities_lidar_odir):
            os.makedirs(self.intensities_lidar_odir)
        self.line_odir = data_odir + "/" + "line_segments"
        if not os.path.exists(self.line_odir):
            os.makedirs(self.line_odir)
        self.vel_odir = data_odir + "/" + "velocities"
        if not os.path.exists(self.vel_odir):
            os.makedirs(self.vel_odir)
        self.pos_odir = data_odir + "/" + "positions"
        if not os.path.exists(self.pos_odir):
            os.makedirs(self.pos_odir)

        # timer:
        self.timer = None
        self.rate = 20  # 20 Hz velocity controller
        self.idx = 0
        # Lock
        self.lock = threading.Lock() 

        # initialize ROS objects:
        self.tf_listener = tf.TransformListener()
        self.scan_sub = rospy.Subscriber("scan", LaserScan, self.scan_callback)
        self.line_sub = rospy.Subscriber("line_segments", LineSegmentList, self.line_segments_callback)
        self.dwa_cmd_sub = rospy.Subscriber('bluetooth_teleop/cmd_vel', Twist, self.dwa_cmd_callback) #, queue_size=1)
        self.robot_pose_pub = rospy.Subscriber('robot_pose', PoseStamped, self.robot_pose_callback)


    # Callback function for the scan measurement subscriber
    def scan_callback(self, laserScan_msg):
        # get the laser scan data:
        scan_data = np.array(laserScan_msg.ranges, dtype=np.float32)
        scan_data[np.isnan(scan_data)] = 0.
        scan_data[np.isinf(scan_data)] = 0.
        self.scan_lidar = scan_data
        # get the laser intensity data:
        intensity_data = np.array(laserScan_msg.intensities, dtype=np.float32)
        intensity_data[np.isnan(intensity_data)] = 0.
        intensity_data[np.isinf(intensity_data)] = 0.
        self.intensities_lidar = intensity_data
        self.rosbag_cnt += 1
    
    # Callback function for the line segments subscriber
    def line_segments_callback(self, lineSeg_msg):
        self.lines = []
        # get the laser line segments data:
        line_segs = lineSeg_msg.line_segments
        for line_seg in line_segs:
            line = [line_seg.start[0], line_seg.start[1], line_seg.end[0], line_seg.end[1]]
            self.lines.append(line)

    # Callback function for the local map subscriber
    def dwa_cmd_callback(self, robot_vel_msg):
        self.vel_cmd = np.array([robot_vel_msg.linear.x, robot_vel_msg.angular.z])

    # Callback function for the current robot pose subscriber
    def robot_pose_callback(self, robot_pose_msg):
        # Cartesian coordinate:
        #self.pos = np.array([robot_pose_msg.pose.position.x, robot_pose_msg.pose.position.y, robot_pose_msg.pose.orientation.z])
        quaternion = [
            robot_pose_msg.pose.orientation.x, robot_pose_msg.pose.orientation.y,
            robot_pose_msg.pose.orientation.z, robot_pose_msg.pose.orientation.w
        ]
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
        self.pos = np.array([robot_pose_msg.pose.position.x, robot_pose_msg.pose.position.y, yaw])

        # start the timer if this is the first path received
        if self.timer is None:
            self.start()
                
    # Start the timer that calculates command velocities
    def start(self):
        # initialize timer for controller update
        self.timer = rospy.Timer(rospy.Duration(1./self.rate), self.timer_callback)

    # function that runs every time the timer finishes to ensure that vae data are sent regularly
    def timer_callback(self, event):  
        # lock data:
        #self.lock.acquire()
        scan_ranges = self.scan_lidar
        scan_intensities = self.intensities_lidar
        line_segs = self.lines
        rob_pos = self.pos
        vel_cmd = self.vel_cmd
        #self.lock.release()
        # check if the rosbag is done:
        if(self.reg_cnt > 3): 
            self.reg_cnt = 0

        self.rosbag_cnt_reg[self.reg_cnt] = self.rosbag_cnt
        self.reg_cnt += 1
        cnt_unique = np.unique(self.rosbag_cnt_reg)
        # write array into npy:
        if(self.record and len(cnt_unique)>1): #and self.idx < 15000):
            # write lidar scan in lidar frame into np.array:
            scan_name = self.scan_lidar_odir + "/" + str(self.idx).zfill(7) 
            np.save(scan_name, scan_ranges)
            # write lidar intensity in lidar frame into np.array:
            intensity_name = self.intensities_lidar_odir + "/" + str(self.idx).zfill(7) 
            np.save(intensity_name, scan_intensities)
            # write line segments in lidar frame into np.array:
            line_name = self.line_odir + "/" + str(self.idx).zfill(7) 
            np.save(line_name, line_segs)
            # write velocoties into np.array:
            vel_name = self.vel_odir + "/" + str(self.idx).zfill(7)
            np.save(vel_name, vel_cmd)
            # write robot position into np.array:
            pos_name = self.pos_odir + "/" + str(self.idx).zfill(7)
            np.save(pos_name, rob_pos)

            #img.show()
            self.idx +=  1
            print("idx: ", self.idx)
        else:
            print("idx: ", self.idx)
        
if __name__ == '__main__':
    rospy.init_node('data_collection')
    data = DataCollection()
    rospy.spin()


