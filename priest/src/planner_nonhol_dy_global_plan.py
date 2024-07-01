#!/usr/bin/env python3

import rospy 
from laser_assembler.srv import *
from sensor_msgs.msg import PointCloud
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Twist
import tf
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped 
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray  
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from tf.transformations import quaternion_from_euler

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from functools import partial
from jax import jit, random
import bernstein_coeff_order10_arbitinterval
import mpc_non_dy
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy
import time
from jax import vmap, random
from scipy.interpolate import UnivariateSpline
from sensor_msgs.msg import LaserScan
from threading import Lock
from tf2_msgs.msg import TFMessage
import threading
import random as rnd
import open3d
from std_srvs.srv import Empty
from tempfile import TemporaryFile
import message_filters
import bernstein_coeff_order10_arbitinterval
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import cProfile
from nav_msgs.srv import GetPlan, GetPlanRequest
from robot_localization.srv import SetPose
from std_msgs.msg import Header
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]='false'

unpause_physics_service = None
pause_physics_service = None

# @partial(jit, static_argnums=(0,),backend="gpu")
def path_spline(x_waypoint, y_waypoint, way_point_shape):

    x_diff = jnp.diff(x_waypoint)
    y_diff = jnp.diff(y_waypoint)

    arc = jnp.cumsum(jnp.sqrt( x_diff**2 + y_diff**2))
    arc_length = arc[-1]

    arc_vec = jnp.linspace(0, arc_length, way_point_shape)
    # print(arc_vec)
    return arc_length, arc_vec, x_diff, y_diff


class planning_traj():

    def __init__(self):

        rospy.init_node("MPC_planning")
        rospy.on_shutdown(self.shutdown)

        self.cmd_vel_pub = rospy.Publisher("priest/cmd_vel", Twist, queue_size=10)
        self.feas_path_publisher = rospy.Publisher("feas_path", Marker, queue_size=1)
        self.visible_obs_publisher = rospy.Publisher("visible_obs", MarkerArray, queue_size=10)
        self.homotopy_path_publisher = rospy.Publisher("homotopy", MarkerArray, queue_size=1)
        self.cmd_vel_obs_1_pub = rospy.Publisher("obs_1/cmd_vel", Twist, queue_size=20)
        self.cmd_vel_obs_2_pub = rospy.Publisher("obs_2/cmd_vel", Twist, queue_size=20)
        self.cmd_vel_obs_3_pub = rospy.Publisher("obs_3/cmd_vel", Twist, queue_size=20)
        self.cmd_vel_obs_4_pub = rospy.Publisher("obs_4/cmd_vel", Twist, queue_size=20)
        self.cmd_vel_obs_5_pub = rospy.Publisher("obs_5/cmd_vel", Twist, queue_size=20)
        self.cmd_vel_obs_6_pub = rospy.Publisher("obs_6/cmd_vel", Twist, queue_size=10)
        self.cmd_vel_obs_7_pub = rospy.Publisher("obs_7/cmd_vel", Twist, queue_size=10)
        self.cmd_vel_obs_8_pub = rospy.Publisher("obs_8/cmd_vel", Twist, queue_size=10)
        self.cmd_vel_obs_9_pub = rospy.Publisher("obs_9/cmd_vel", Twist, queue_size=10)
        self.cmd_vel_obs_10_pub = rospy.Publisher("obs_10/cmd_vel", Twist, queue_size=20)
        
        # self.global_plan_pub = rospy.Publisher("/global_plan", Path, queue_size=20)
        
        ################################

        self.vx_init = 0.05
        self.vy_init = 0.1
        self.ax_init = 0.0
        self.ay_init = 0.0
        self.vx_fin = 0.0
        self.vy_fin = 0.0
        self.ax_fin = 0.0
        self.ay_fin = 0.0

        ######################################### MPC parameters

        self.v_max = 1.0
        self.v_min = 0.02
        self.a_max = 1.0
        ##################################################

        self.maxiter = 1
        self.maxiter_cem = 12
        self.weight_track = 0.001
        self.weight_smoothness = 1

        self.a_obs_1 = 0.5
        self.b_obs_1 = 0.5

        self.a_obs_2 = 0.68
        self.b_obs_2 = 0.68

        #####################

        self.t_fin = 10 #### time horizon
        self.num = 100 #### number of steps in prediction.
        self.num_batch = 30
        self.tot_time = np.linspace(0, self.t_fin, self.num)
    
        self.x_init = 1.0
        self.y_init = 2
    
        self.x_fin = 2.0
        self.y_fin = 15

        self.x_fin_t = 2.0
        self.y_fin_t = 15

        self.i = 221

        ########### Dynamic obstacles position
        data_obs_mat = loadmat('/home/laksh/priest_ws/src/priest/src/obstacle_pos/obstacles_dy_21.mat')
        data_obs = data_obs_mat["obs"]
        # print(data_obs)

        self.x_obs_init_dy = data_obs[:,0] 
        self.y_obs_init_dy = data_obs[:,1]

        self.x_obs_1 = self.x_obs_init_dy[0]
        self.y_obs_1 = self.y_obs_init_dy[0]

        self.x_obs_2 = self.x_obs_init_dy[1]
        self.y_obs_2 = self.y_obs_init_dy[1]

        self.x_obs_3 = self.x_obs_init_dy[2]
        self.y_obs_3 = self.y_obs_init_dy[2]

        self.x_obs_4 = self.x_obs_init_dy[3]
        self.y_obs_4 = self.y_obs_init_dy[3]

        self.x_obs_5 = self.x_obs_init_dy[4]
        self.y_obs_5 = self.y_obs_init_dy[4]

        self.x_obs_6 = self.x_obs_init_dy[5]
        self.y_obs_6 = self.y_obs_init_dy[5]

        self.x_obs_7 = self.x_obs_init_dy[6]
        self.y_obs_7 = self.y_obs_init_dy[6]

        self.x_obs_8 = self.x_obs_init_dy[7]
        self.y_obs_8 = self.y_obs_init_dy[7]        
        
        self.x_obs_9 = self.x_obs_init_dy[8]
        self.y_obs_9 = self.y_obs_init_dy[8]

        self.x_obs_10 = self.x_obs_init_dy[9]
        self.y_obs_10 = self.y_obs_init_dy[9]

        # self.x_best_global = jnp.linspace(self.x_init, self.x_init , 100)
        # self.y_best_global = jnp.linspace(self.y_init, self.y_init, 100)
            
        self.theta_des = np.arctan2(self.y_fin-self.y_init, self.x_fin-self.x_init) 
        self.v_des = 1.0

        self.maxiter_mpc = 300
        self.mutex = Lock()
        self.updated = False
        self.updated_obs = False
        self.num_update_state = 1
        self.num_up = 100

        self.x_waypoint = jnp.linspace(self.x_init, self.x_fin + 10.0 * jnp.cos(self.theta_des) , 1000)
        self.y_waypoint = jnp.linspace(self.y_init, self.y_fin + 10.0 * jnp.sin(self.theta_des),  1000)
        # self.way_point_shape = 1000
        # self.received_global_plan = False
        # self.global_plan_sub = rospy.Subscriber("/move_base/TrajectoryPlannerROS/global_plan", Path, self.update_waypoints)
        
        # self.set_pose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=10)
        # self.set_pose()
        
        # self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        # self.send_goal()
        
        # time.sleep(5)
        self.received_global_plan = False
        self.global_plan_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.update_waypoints_from_subscriber)
              
        # self.global_plan_pub = rospy.Publisher("/global_plan", Path, queue_size=20)
        
        # print("Waiting for Service")
        # rospy.wait_for_service('/move_base/make_plan')
        # self.get_waypoints_from_service()
        # print("Received Global Plan")

        self.x_obs_init_1 = np.ones(800) * 100
        self.y_obs_init_1 = np.ones(800) * 100 

        self.x_obs_down = np.ones(420) * 100
        self.y_obs_down = np.ones(420) * 100

        self.x_obs_init = np.ones(420) * 100
        self.y_obs_init = np.ones(420) * 100 
      
        self.vx_obs = 0.0*jnp.ones(420)
        self.vy_obs = 0.0*jnp.ones(420)

        self.vx_obs_dy = 0.0*jnp.ones(10)
        self.vy_obs_dy = -0.1*jnp.ones(10)

        self.x_obs_visible = 10 * jnp.ones(60)
        self.y_obs_visible = 10 * jnp.ones(60)

        self.x_obs_visible_1 = 10 * jnp.ones(420)
        self.y_obs_visible_1 = 10 * jnp.ones(420)

        self.xyz = np.random.rand(800, 3)
        self.pcd =open3d.geometry.PointCloud()

    ###################################

    def convert_angle(self, angle):
        if angle >= 0 and angle <= 2*jnp.pi:
            return angle  

        if angle < 0:
            angle += 2*jnp.pi  

        return angle
    

    ####################################
    
    def set_pose(self):
        # header = Header
        # set_pose_srv = rospy.ServiceProxy("/set_pose", SetPose)
        pose = PoseWithCovarianceStamped()
        # pose.header = header
        pose.pose.pose.position.x = self.x_init
        pose.pose.pose.position.y = self.y_init
        # header = header()
        self.set_pose_pub.publish(pose)
        
    def send_goal(self):
        pose = PoseStamped()
        pose.pose.position.x = self.x_fin
        pose.pose.position.y = self.y_fin
        header = Header()
        pose.header = header

        self.goal_pub.publish(pose)
    
    # def get_waypoints_from_service(self):
    #     get_global_plan_service = rospy.ServiceProxy("/move_base/make_plan", GetPlan)
    #     header = Header()
    #     header.frame_id = "map"
    #     start = PoseStamped()
    #     start.header = header
    #     start.pose.position.x = self.x_init
    #     start.pose.position.y = self.y_init
        
    #     end = PoseStamped()
    #     end.header = header
    #     end.pose.position.x = self.x_fin
    #     end.pose.position.y = self.y_fin
        
    #     plan = get_global_plan_service(start, end, 0.2)
        
    #     # print(plan)
        
    #     plan = plan.plan
    #     x_waypoint = []
    #     y_waypoint = []
    #     for pose in plan.poses:
    #         x_waypoint.append(pose.pose.position.x)
    #         y_waypoint.append(pose.pose.position.y)
    #         pose.header = header
            
    #     self.x_waypoint = jnp.asarray(x_waypoint).flatten()
    #     self.y_waypoint = jnp.asarray(y_waypoint).flatten()
    #     self.way_point_shape = len(plan.poses)
        
    #     plt.plot(x_waypoint, y_waypoint)
        
    #     plt.savefig("global_plan.png")
        
    #     # self.plan = plan
    #     # self.global_plan_pub.publish(plan)
        
    def update_waypoints_from_subscriber(self, msg:Path):
        self.received_global_plan = True
        poses = msg.poses
        x_waypoint = []
        y_waypoint = []
        for pose in poses:
            x_waypoint.append(pose.pose.position.x)
            y_waypoint.append(pose.pose.position.y)
            
        self.x_waypoint = jnp.asarray(x_waypoint).flatten()
        self.y_waypoint = jnp.asarray(y_waypoint).flatten()
        
        # self.x_waypoint = jnp.linspace(self.x_init, self.x_fin + 10.0 * jnp.cos(self.theta_des) , 1000)
        # self.y_waypoint = jnp.linspace(self.y_init, self.y_fin + 10.0 * jnp.sin(self.theta_des),  1000)
        
        self.way_point_shape = len(self.x_waypoint)
        
        # start = time.time()
        _, self.arc_vec, self.x_diff, self.y_diff = path_spline(self.x_waypoint, self.y_waypoint, self.way_point_shape)
        # print(f"Spline calc = {time.time() - start}, waypoint shape = {self.way_point_shape}")
        
    def Callback(self, msg_1, msg_2):

        self.x_init = msg_2.pose.pose.position.x
        self.y_init = msg_2.pose.pose.position.y

        orientation_q = msg_2.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]

        (_, _, yaw) = euler_from_quaternion (orientation_list)
        self.theta_init = yaw

        counter = 0
        increment_value = 1
        for i in range (0, len(msg_1.points), increment_value):

            self.x_obs_init_1[counter] = msg_1.points[i].x 
            self.y_obs_init_1[counter] = msg_1.points[i].y 
            counter += 1

        idxes = np.argwhere((self.x_obs_init_1[:]>=30) | (self.y_obs_init_1[:]>=30))
        self.x_obs_init_1[idxes] = self.x_obs_init_1[0]
        self.y_obs_init_1[idxes] = self.y_obs_init_1[0]

        self.xyz[:,0] = self.x_obs_init_1
        self.xyz[:,1] = self.y_obs_init_1
        self.xyz[:,2] = 0.0

        self.pcd.points = open3d.utility.Vector3dVector(self.xyz)
        downpcd = self.pcd.voxel_down_sample(voxel_size = 0.16) 
        downpcd_array = np.asarray(downpcd.points)
        num_down_samples = downpcd_array[:,0].shape[0]

        self.x_obs_down[0:num_down_samples] = downpcd_array[:,0]
        self.y_obs_down[0:num_down_samples] = downpcd_array[:,1]

        self.x_obs_down[num_down_samples-1:-1] = 100
        self.y_obs_down[num_down_samples-1:-1] = 100

        self.x_obs_init = self.x_obs_down
        self.y_obs_init = self.y_obs_down


    def OdomCallback_1(self, msg_2):

        self.x_obs_1 = msg_2.pose.pose.position.x 
        self.y_obs_1 = msg_2.pose.pose.position.y 

    ###########################################
    def OdomCallback_2(self, msg_2):

        self.x_obs_2 = msg_2.pose.pose.position.x 
        self.y_obs_2 = msg_2.pose.pose.position.y 
    
    ####################################
    def OdomCallback_3(self, msg_2):

        self.x_obs_3 = msg_2.pose.pose.position.x 
        self.y_obs_3 = msg_2.pose.pose.position.y 
    
    ######################################
    def OdomCallback_4(self, msg_2):

        self.x_obs_4 = msg_2.pose.pose.position.x 
        self.y_obs_4 = msg_2.pose.pose.position.y 

    ###############################
    def OdomCallback_5(self, msg_2):

        self.x_obs_5 = msg_2.pose.pose.position.x 
        self.y_obs_5 = msg_2.pose.pose.position.y 
    
    ##############################
    def OdomCallback_6(self, msg_2):

        self.x_obs_6 = msg_2.pose.pose.position.x 
        self.y_obs_6 = msg_2.pose.pose.position.y 
    
    ###########################
    def OdomCallback_7(self, msg_2):

        self.x_obs_7 = msg_2.pose.pose.position.x 
        self.y_obs_7 = msg_2.pose.pose.position.y

    #################################
    def OdomCallback_8(self, msg_2):

        self.x_obs_8 = msg_2.pose.pose.position.x 
        self.y_obs_8 = msg_2.pose.pose.position.y  
    
    ##############################
    def OdomCallback_9(self, msg_2):

        self.x_obs_9 = msg_2.pose.pose.position.x 
        self.y_obs_9 = msg_2.pose.pose.position.y 

    ##############################
    def OdomCallback_10(self, msg_2):

        self.x_obs_10 = msg_2.pose.pose.position.x 
        self.y_obs_10 = msg_2.pose.pose.position.y 
        
    ######################################

    def drawVisibleObstacles(self, obs_pose, index,marker_array_points):

        marker= Marker()
        marker.header.frame_id = "world"
        marker.ns = ''
        marker.id = index
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.ns = ""
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.001
        marker.pose.orientation.w = 1.0
        marker.color.a = 1.0
        marker.color.r = 0.1
        marker.color.g = 0.6
        marker.color.b = 0.3
        marker.pose = obs_pose
        marker_array_points.markers.append(marker)

        return marker_array_points

    ###############################
    def PlotObstacles(self):

        marker_array_points = MarkerArray()
        anti_direction = False 
        while(True):  
            if(self.updated_obs==True):

                self.mutex.acquire()
                temp_obs_x = self.x_obs_visible
                temp_obs_y = self.y_obs_visible
                self.mutex.release()

                for j in range(jnp.shape(temp_obs_x)[0]):
                    obs_pose = Pose()
                    obs_pose.position.x = temp_obs_x[j]
                    obs_pose.position.y = temp_obs_y[j]
                    obs_pose.position.z = 0

                    q = quaternion_from_euler(0, 0, 1.57)
                    obs_pose.orientation.x = q[0]
                    obs_pose.orientation.y = q[1]
                    obs_pose.orientation.z = q[2]
                    obs_pose.orientation.w = q[3]
                    marker_array_points = self.drawVisibleObstacles(obs_pose,j,marker_array_points)

                self.visible_obs_publisher.publish(marker_array_points)
                self.updated_obs = False

    ######################################

    def planner(self, ):

        vel_cmd = Twist()
     
        sub_PointCloud = message_filters.Subscriber("/pointcloud", PointCloud)
        sub_Odom = message_filters.Subscriber("/odom", Odometry)
        ts = message_filters.ApproximateTimeSynchronizer([sub_PointCloud, sub_Odom], 1,1, allow_headerless=True )
        ts.registerCallback(self.Callback)
        
        obs_subscribers = [rospy.Subscriber(f"obs_{i}/odom", Odometry, getattr(self, f"OdomCallback_{i}")) for i in range(1, 11)]
        vel_cmds = [Twist() for _ in range(10)]
        cmd_vel_obs_pubs = [getattr(self, f"cmd_vel_obs_{i}_pub") for i in range(1, 11)]
    
        num_obs_1 = 50
        num_obs_2 = 10

        while not self.received_global_plan:
            time.sleep(0.1)
        prob = mpc_non_dy.batch_crowd_nav(self.a_obs_1, self.b_obs_1, self.a_obs_2, self.b_obs_2, self.v_max, self.v_min, self.a_max, num_obs_1, num_obs_2, self.t_fin, self.num, self.num_batch, self.maxiter, self.maxiter_cem, self.weight_smoothness, self.weight_track, self.way_point_shape, self.v_des)

        key = random.PRNGKey(0)

    
        # c_x_data = []
        # c_y_data = []

        # x_obs_save = []
        # y_obs_save = []

        # total_time = []
        # theta_init = []

        initial_state = jnp.hstack(( self.x_init, self.y_init, self.vx_init, self.vy_init, self.ax_init, self.ay_init ))

        vx_obs = self.vx_obs
        vy_obs = self.vy_obs

        vx_obs_dy = self.vx_obs_dy
        vy_obs_dy = self.vy_obs_dy

        mpc_count = 0
        
        
        for i in range(0, self.maxiter_mpc):
            
            # if i == 0:
            x_guess_per, y_guess_per = prob.compute_warm_traj(initial_state, self.v_des, self.x_waypoint, self.y_waypoint, self.arc_vec, self.x_diff, self.y_diff)
                
            initial_state = jnp.hstack(( self.x_init, self.y_init, self.vx_init, self.vy_init, self.ax_init, self.ay_init ))

            lamda_x = jnp.zeros((self.num_batch, prob.nvar))
            lamda_y = jnp.zeros((self.num_batch, prob.nvar))

            start = time.time()

            x_obs_init_dy = self.x_obs_init_dy
            y_obs_init_dy = self.y_obs_init_dy

            x_obs_init = self.x_obs_init
            y_obs_init = self.y_obs_init

            x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, x_obs_trajectory_dy, y_obs_trajectory_dy = prob.compute_obs_traj_prediction( jnp.asarray(x_obs_init_dy).flatten(), jnp.asarray(y_obs_init_dy).flatten(), vx_obs_dy, vy_obs_dy, jnp.asarray(x_obs_init).flatten(), jnp.asarray(y_obs_init).flatten(), vx_obs, vy_obs, initial_state[0], initial_state[1] ) ####### obstacle trajectory prediction

            sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess,c_mean, c_cov, x_fin, y_fin = prob.compute_traj_guess( initial_state, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_dy, y_obs_trajectory_dy , self.v_des, self.x_waypoint, self.y_waypoint, self.arc_vec, x_guess_per, y_guess_per, self.x_diff, self.y_diff)
            
            self.x_fin = x_fin
            self.y_fin = y_fin  

            pause_physics_service()

            x, y, c_x_best, c_y_best, x_best, y_best, x_guess_per , y_guess_per= prob.compute_cem(key, initial_state, self.x_fin, self.y_fin, lamda_x, lamda_y, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, x_obs_trajectory_dy, y_obs_trajectory_dy , sol_x_bar, sol_y_bar, x_guess, y_guess,  xdot_guess, ydot_guess, xddot_guess, yddot_guess, self.x_waypoint,  self.y_waypoint, self.arc_vec, c_mean, c_cov )
            
            vx_control, vy_control, ax_control, ay_control, norm_v_t, angle_v_t= prob.compute_controls(c_x_best, c_y_best)

            zeta = self.convert_angle(self.theta_init) - self.convert_angle(angle_v_t)
            v_t_control = norm_v_t * jnp.cos(zeta)
            omega_control = - zeta / (6*self.t_fin*0.01)
            self.x_obs_visible = x_obs_trajectory[:,0]
            self.y_obs_visible = y_obs_trajectory[:,0]

            self.updated_obs = True
            self.updated = True
            ###########################
            self.vx_init = vx_control
            self.vy_init = vy_control

            self.ax_init = ax_control
            self.ay_init = ay_control

            vel_cmd.linear.x = 1*v_t_control
            vel_cmd.angular.z = omega_control

            if i <=3 :
                vel_cmd.linear.x = 0.0*v_t_control
                vel_cmd.angular.z = omega_control
                self.cmd_vel_pub.publish(vel_cmd)
                

            else:
                unpause_physics_service()

                for j in range(10):
                    vel_cmds[j].linear.x = self.vx_obs[j]
                    vel_cmds[j].linear.y = self.vy_obs_dy[j]
                    cmd_vel_obs_pubs[j].publish(vel_cmds[j])

                # self.global_plan_pub.publish(self.plan)
                self.cmd_vel_pub.publish(vel_cmd)
                rospy.sleep(0.005)
                # total_time.append(time.time()- start)

                self.x_obs_init_dy = jnp.hstack(( self.x_obs_1, self.x_obs_2, self.x_obs_3, self.x_obs_4, self.x_obs_5, self.x_obs_6, self.x_obs_7, self.x_obs_8, self.x_obs_9, self.x_obs_10 ))
                self.y_obs_init_dy = jnp.hstack(( self.y_obs_1, self.y_obs_2, self.y_obs_3, self.y_obs_4, self.y_obs_5, self.y_obs_6, self.y_obs_7, self.y_obs_8, self.y_obs_9, self.y_obs_10 ))

            print (time.time() - start, "time_2")

            # x_obs_save.append(x_obs_trajectory[:,0])
            # y_obs_save.append(y_obs_trajectory[:,0])

            # c_x_data.append(c_x_best)
            # c_y_data.append(c_y_best)

            # theta = self.theta_init 
            # theta_init.append(theta)

            mpc_count = mpc_count +1
            dist_final_point = np.sqrt((self.x_init - self.x_fin_t)**2 + (self.y_init - self.y_fin_t)**2)

            if (dist_final_point <= 1):
                vel_cmd.linear.x = 0.0
                vel_cmd.linear.y = 0.0
                self.cmd_vel_pub.publish(vel_cmd)
                print("Distance is satisfied")
                break
                

        print("Stop mpc")

        
        # theta_init = np.array(theta_init)

        # c_x_data = np.array(c_x_data)
        # c_y_data = np.array(c_y_data)

        # x_obs_save = np.array(x_obs_save)
        # y_obs_save = np.array(y_obs_save)

        # total_time = np.array(total_time)

        # cx_data =  c_x_data.reshape(mpc_count , 11)
        # cy_data =  c_y_data.reshape(mpc_count , 11)

        # world_num = self.i
        # np.save("cx_data_dy_"+str(world_num)+".npy", cx_data)
        # np.save("cy_data_dy_"+str(world_num)+".npy", cy_data)

        # np.save("total_time_dy_"+str(world_num)+".npy", total_time)

        # np.save("x_obs_save_dy_"+str(world_num)+".npy", x_obs_save)
        # np.save("y_obs_save_dy_"+str(world_num)+".npy", y_obs_save)

        # np.save("theta_init"+str(world_num)+".npy", theta_init)

        # print(np.sum(total_time, axis = 0), "total_time")
        print("Done")
        rospy.signal_shutdown('Finished running node')
        
    def shutdown(self):

        rospy.loginfo("Stop")
        rospy.sleep(2)

    
if __name__ == "__main__":

    pause_physics_service = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
    unpause_physics_service = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

    motion_planning = planning_traj()
    motion_planning.planner()

    




    


    
    




    


    

















	







