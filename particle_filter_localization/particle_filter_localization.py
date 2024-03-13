import time

import numpy as np
import rclpy
import tf2_ros
import tf_transformations as tf
from geometry_msgs.msg import (Pose, PoseArray, PoseStamped,
                               PoseWithCovarianceStamped, TransformStamped,
                               Twist)
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from scipy.stats import norm
from sensor_msgs.msg import LaserScan
from tf_transformations import (euler_from_matrix, euler_from_quaternion,
                                euler_matrix, quaternion_from_euler,
                                quaternion_matrix)


class ParticleFilterLocalization(Node):
    def __init__(self,
                 particle_count: int = 20,
                 ) -> None:
        super().__init__(node_name="ParticleFilterLocalization")
        # For testing
        init_ros = True
        self.use_odom = True
        self.verbose = True
        self.lidar_init = False

        # Flag to initialize sampler
        self.initializer = True

        # Constants
        self.POSITION_DIMENSIONS = 2
        self.ORIENTATION_DIMENSIONS = 1
        self.WEIGHT_DIMENSION = 1
        self.NUM_PARTICLES = particle_count

        # Random
        self.ERROR_MEAN = 0
        self.ERROR_LINEAR_STD = 0.01
        self.ERROR_ANGULAR_STD = np.pi/64
        self.VARIANCE_THRESHOLD = 1e-6

        # Initial guess
        self.use_initial_guess = True
        # Initial guess mean in x, y, theta
        self.initial_particle_guess = np.array([
            0.0, 0.0, 0.0
            ])
        # Standard deviations for sampling around initial guess
        self.initial_particle_std = np.array([
            self.ERROR_LINEAR_STD, self.ERROR_LINEAR_STD, self.ERROR_ANGULAR_STD,
            ])

        self.CMD_VEL_TOPIC = "/cmd_vel"
        self.MAP_TOPIC = "/map_loaded"
        self.SCAN_TOPIC = "/scan"
        self.POSE_TOPIC = "/pose"
        self.ODOM_TOPIC = "/odom"
        self.PARTICLE_ARRAY_TOPIC = "/particle_cloud"
        self.ESTIMATED_POSE_TOPIC = "/pose_estimated"
        self.ROBOT_FRAME = "base_link"
        self.ODOM_TOPIC = "odom"
        self.MAP_FRAME = "map"
        self.SCANNER_FRAME = "base_laser_front_link"
        
        # Input
        self.control_input = np.zeros((self.POSITION_DIMENSIONS + \
                                    self.ORIENTATION_DIMENSIONS))

        # Particles
        self.particles = np.zeros((self.NUM_PARTICLES, \
                                   self.POSITION_DIMENSIONS + self.ORIENTATION_DIMENSIONS + self.WEIGHT_DIMENSION))
        
        self.origin = Pose()

        # time delta
        self.filter_time_step = 0.1
        self.previous_prediction_time = None

        # LIDAR params
        self.scanner_info = {}
        self.variance = 2.0

        self.ray_step = 1.0

        self.RESAMPLE_FRACTION = 0.2
        self.SCAN_EVERY = 2
        self.scan_ranges: np.ndarray = None

        # expected pose
        self.expected_pose = self.set_posestamped(
                        PoseStamped(),
                        [0.0, 0.0, 0.0], 
                        [0.0, 0.0, 0.0],
                        self.MAP_FRAME,
                        )
        # Robot pose from SLAM
        self.robot_pose = np.zeros(3)
        self.robot_pose_particle = np.zeros(3)

        # Odometry
        self.odometry_position = np.zeros(3)
        self.odometry_orientation = np.identity(3)
        self.previous_odometry_position = np.zeros(3)
        self.previous_odometry_orientation = np.identity(3)
        
        # Map
        self.map = None
        self.width = None
        self.height = None
        self.timestamp = None

        self.start_localization = False
        self.samples_initialized = False

        # Setup subscribers and publishers
        if init_ros:
            # self.tf_buffer = tf2_ros.Buffer()
            # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

            # Command input subscriber
            self.cmd_vel_subscriber = self.create_subscription(
                Odometry if self.use_odom else Twist,
                self.ODOM_TOPIC if self.use_odom else self.CMD_VEL_TOPIC,
                self.odom_callback if self.use_odom else self.cmd_vel_callback,
                10,
                )

            # Map subscriber
            self.map_subscriber = self.create_subscription(
                OccupancyGrid,
                self.MAP_TOPIC,
                self.map_callback,
                qos_profile=rclpy.qos.qos_profile_sensor_data,
                )
       
            # scan subscriber
            self.scan_subscriber = self.create_subscription(
                LaserScan,
                self.SCAN_TOPIC,
                self.scan_callback,
                10,
                )
            
            # Test pose subscriber
            # self.pose_subscriber = self.create_subscription(
            #     PoseWithCovarianceStamped,
            #     self.POSE_TOPIC,
            #     self.pose_callback,
            #     10,
            #     )

            # Particle publisher (for display in RViz)
            self.particle_array_publisher = self.create_publisher(
                PoseArray,
                self.PARTICLE_ARRAY_TOPIC,
                10,
                )

            self.estimated_pose_publisher = self.create_publisher(
                PoseStamped, 
                self.ESTIMATED_POSE_TOPIC,
                10,
                )
            
            # Test map publisher
            self.test_map_msg = OccupancyGrid()
            self.map_publisher = self.create_publisher(
                OccupancyGrid, 
                '/map_test',
                10,
                )
            
            timer = self.create_timer(self.filter_time_step, self.localization_loop)
            
        else:
            self.tf_buffer = None
            self.tf_listener = None

            self.cmd_vel_subscriber = None
            self.map_subscriber = None
            self.scan_subscriber = None
            self.particle_array_publisher = None
            self.estimated_pose_publisher = None
        
            
    
    def scan_callback(self, msg: LaserScan) -> None:

        if not self.lidar_init:

            # tf_lidar_wrt_robot: TransformStamped = self.tf_buffer.lookup_transform(
                                                    # self.ROBOT_FRAME, self.SCANNER_FRAME, \
                                                    # rclpy.time.Time(), timeout=rclpy.time.Duration(seconds=2.0))
            # self.tf_lidar_wrt_robot_matrix = self.get_homogeneous_transformation_from_transform(tf_lidar_wrt_robot)
            
            self.scanner_info.update({
                "frame_id" : msg.header.frame_id,
                "angle_min" : msg.angle_min,
                "angle_max" : msg.angle_max,
                "range_min" : msg.range_min,
                "range_max" : msg.range_max,
                "angle_increment" : msg.angle_increment,
                "num_scans": len(msg.ranges),
                })

            self.lidar_init = True
            self.get_logger().info(f"LiDAR parameters initialized: \n{self.scanner_info}")

        self.scan_ranges = np.array(msg.ranges)

        # self.scan_cartesian = self.convert_scan_to_cartesian(self.scan_ranges)

        # self.scan_cartesian = self.transform_with_homogeneous_transform(self.scan_cartesian, \
                                                                        # self.tf_lidar_wrt_robot_matrix)

        return None
            
    
    def pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self.robot_pose[0] = msg.pose.pose.position.x
        self.robot_pose[1] = msg.pose.pose.position.y
        self.robot_pose[2] = euler_from_quaternion(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
             ]
        )[2]

        self.robot_pose_particle[0] = (self.robot_pose[0] - self.origin.position.x) / self.resolution
        self.robot_pose_particle[1] = (self.robot_pose[1] - self.origin.position.y) / self.resolution 
        self.robot_pose_particle[2] = self.robot_pose[2]
        
        # This initializes particles at the current pose; only for testing motion model
        # self.particles = np.zeros((1, 4))
        # self.particles[0, :-1] = self.robot_pose_particle

        return None
 

    def cmd_vel_callback(self, msg: Twist) -> None:
        """
        Extract linear and angular velocity from cmd_vel message
        """
        self.control_input[0] = msg.linear.x * 8
        self.control_input[1] = msg.linear.y * 8

        self.control_input[2] = msg.angular.z / 4
        self.start_localization = True

        return None
    

    def odom_callback(self, msg: Odometry) -> None:
        self.odometry_position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])
        self.odometry_orientation = np.array(quaternion_matrix([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ])[:3, :3])

        # print(f"Odom update: position {self.odometry_position}")

        return None


    def map_callback(self, msg: OccupancyGrid) -> None:
        # self.get_logger().info(f'I am recieving the map...')

        self.map = np.array(msg.data, dtype=np.int8)
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin
        self.timestamp = msg.header.stamp

        self.get_logger().info(f'Map width is: {self.width} height is: {self.height} Resolution is {self.resolution}')

        # Converting the map to a 2D array
        self.state = np.reshape(self.map, (self.width, self.height), order='F')

        if not self.samples_initialized:
            if self.use_initial_guess:
                self.sampler_initializer(self.initial_particle_guess, self.initial_particle_std)

            else:
                self.sampler_initializer()
        
            self.samples_initialized = True

        return None


    def sampler_initializer(self, 
                            initial_particle_guess: np.ndarray = None,
                            initial_particle_std: np.ndarray = None,
                            ) -> None:
        if initial_particle_guess is None or initial_particle_std is None:
            samples = self.sample_n_particles(self.NUM_PARTICLES)

        else:
            samples = self.sample_n_normal_particles(initial_particle_guess, initial_particle_std, self.NUM_PARTICLES)

        self.particles[:,:-1] = samples
        self.particles[:,-1] = 1/self.NUM_PARTICLES
        
        if self.verbose:
            self.get_logger().info(f'Particles initialized\n{self.particles}')

        self.previous_prediction_time = time.time()

        return None


    def sample_n_particles(self, 
                           n,
                           ) -> np.ndarray:
        x = np.random.uniform(0.0, self.width, n)
        y = np.random.uniform(0.0, self.height, n)
        theta = np.random.uniform(-np.pi, np.pi, size=n)

        samples = np.vstack((x, y, theta)).T

        return samples
    

    def sample_n_normal_particles(self, 
                                  initial_particle_guess, 
                                  initial_particle_std, 
                                  n,
                                  ) -> np.ndarray:
        x = np.random.normal(initial_particle_guess[0], initial_particle_std[0], n)
        y = np.random.normal(initial_particle_guess[1], initial_particle_std[1], n)
        theta = np.random.normal(initial_particle_guess[2], initial_particle_std[2], n)
        # Wrapping between -180 to 180
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        x = x - (self.origin.position.x / self.resolution)
        y = y - (self.origin.position.y / self.resolution)

        samples = np.vstack((x, y ,theta)).T

        return samples
    

    # Methods for converting scan to cartesian and transforming to robot frame
    def convert_scan_to_cartesian(self, 
                                  scan_ranges: np.ndarray,
                                  ) -> np.ndarray:
        '''
        Converts scan point to the cartesian coordinate system
        '''
        scan_points = np.zeros((len(scan_ranges), 3))

        thetas = np.linspace(self.scanner_info["angle_min"], self.scanner_info["angle_max"], self.scanner_info["num_scans"])

        scan_points[:, 0] = scan_ranges * np.cos(thetas)
        scan_points[:, 1] = scan_ranges * np.sin(thetas)
        scan_points[:, 2] = 0.0

        return scan_points
    

    def transform_with_homogeneous_transform(self, 
                                             points: np.ndarray, 
                                             transformation_matrix: np.ndarray,
                                             ) -> np.ndarray:
        points = np.hstack((points, np.ones((points.shape[0], 1))))

        transformed_points = (transformation_matrix @ points.T).T
        transformed_points = transformed_points[:, :-1]

        return transformed_points


    def get_homogeneous_transformation_from_transform(self, 
                                                      transform: TransformStamped,
                                                      ) -> np.ndarray:
        '''
        Return the equivalent homogeneous transform of a TransformStamped object
        '''
        transformation_matrix = tf.quaternion_matrix([
                                    transform.transform.rotation.x,
                                    transform.transform.rotation.y,
                                    transform.transform.rotation.z,
                                    transform.transform.rotation.w,
                                    ])
        transformation_matrix[0,-1] = transform.transform.translation.x
        transformation_matrix[1,-1] = transform.transform.translation.y
        transformation_matrix[2,-1] = transform.transform.translation.z

        return transformation_matrix


    def set_posestamped(self, 
                        pose: PoseStamped, 
                        position, 
                        orientation_euler, 
                        frame_id,
                        ) -> PoseStamped:
        '''
        Sets the fields of a PoseStamped object
        '''
        pose.header.frame_id = frame_id
        
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]

        orientation_quat = tf.quaternion_from_euler(*orientation_euler)

        pose.pose.orientation.x = orientation_quat[0]
        pose.pose.orientation.y = orientation_quat[1]
        pose.pose.orientation.z = orientation_quat[2]
        pose.pose.orientation.w = orientation_quat[3]

        return pose


    def motion_model_prediction(self,
                                particles: np.ndarray,
                                time_delta: float = None,
                                ) -> np.ndarray:
        # Update position of particles

        if self.use_odom:
            translation = self.odometry_position - self.previous_odometry_position
            rotation = np.linalg.pinv(self.previous_odometry_orientation) @ self.odometry_orientation 

            current_orientation_matrices = np.array(
                [euler_matrix(0.0, 0.0, particle[2])[:3, :3] for particle in self.particles]
                )

            rotated_orientation_matrices = current_orientation_matrices @ rotation
            rotated_angles = np.array(
                [tf.euler_from_matrix(rotated_orientation_matrix) for rotated_orientation_matrix in rotated_orientation_matrices]
                )
            
            particles[:, :2] += ((translation[:2] + \
                                  np.random.normal(self.ERROR_MEAN, self.ERROR_LINEAR_STD, (self.NUM_PARTICLES, 2))) \
                                / self.resolution)
            particles[:, 2] = rotated_angles[:, 2] + np.random.normal(self.ERROR_MEAN, self.ERROR_ANGULAR_STD, self.NUM_PARTICLES)

            self.previous_odometry_position = self.odometry_position.copy()
            self.previous_odometry_orientation = self.odometry_orientation.copy()

            return particles

        else:
            # Needs to be separate
            if np.isclose(np.linalg.norm(self.control_input), 0.0):

                return particles

            particles[:, 0] += (self.control_input[0] * np.cos(particles[:, 2]) \
                            + self.control_input[1] * np.sin(particles[:, 2])) \
                            * time_delta \
                            + np.random.normal(self.ERROR_MEAN, self.ERROR_LINEAR_STD, self.NUM_PARTICLES)

            particles[:, 1] += (self.control_input[1] * np.cos(particles[:, 2]) \
                            + self.control_input[0] * np.sin(particles[:, 2])) \
                            * time_delta \
                            + np.random.normal(self.ERROR_MEAN, self.ERROR_LINEAR_STD, self.NUM_PARTICLES)

            # Update heading of particles
            particles[:, 2] += self.control_input[2] * time_delta \
                            + np.random.normal(self.ERROR_MEAN, self.ERROR_ANGULAR_STD, self.NUM_PARTICLES)

            return particles


    def resample_particles(self) -> None:   
        # mean_weight = np.mean(self.particles[:, -1])
        # weight_diff = self.particles[:, -1] - mean_weight
        # resample_weight = 1 - weight_diff
        # resample_weight = self.normalize_weights(resample_weight)

        resample_weight = self.particles[:, -1]

        indices: np.ndarray = np.random.choice(range(self.NUM_PARTICLES), 
                                               self.NUM_PARTICLES, 
                                               p=resample_weight)
        self.particles: np.ndarray = self.particles[indices]

        self.particles[:, -1] = self.normalize_weights(self.particles[:, -1])
    
        return None


    def particle_array_generation(self, 
                                  particles: np.ndarray,
                                  ) -> None:
        """
        Create and publish particle array
        """

        particle_array = PoseArray()
                
        current_time = self.get_clock().now().seconds_nanoseconds()
        particle_array.header.stamp.sec = current_time[0]
        particle_array.header.stamp.nanosec = current_time[1]

        particle_array.header.frame_id = self.MAP_FRAME

        for particle in particles:
            pose = Pose()
            pose.position.x = particle[0] * self.resolution + self.origin.position.x
            pose.position.y = particle[1] * self.resolution + self.origin.position.y
            pose.position.z = 0.0

            orientation_quat = quaternion_from_euler(*[0.0,0.0,particle[2]])

            pose.orientation.x = orientation_quat[0]
            pose.orientation.y = orientation_quat[1]
            pose.orientation.z = orientation_quat[2]
            pose.orientation.w = orientation_quat[3]

            particle_array.poses.append(pose)
        
        self.particle_array_publisher.publish(particle_array)
    

    def simulate_lidar_measurement(self, particle, map_data):
        angle_max = self.scanner_info.get('angle_max')
        angle_min = self.scanner_info.get('angle_min')
        range_max = self.scanner_info.get("range_max")
        num_scans = self.scanner_info.get("num_scans")

        angles = np.linspace(angle_min, angle_max, num_scans)[::-1]#[::self.SCAN_EVERY]

        measurements = np.zeros((num_scans))

        for index, scan_theta in enumerate(angles):
            x = particle[0]
            y = particle[1]
            theta = particle[2] - scan_theta

            # Cast a ray from the sensor's position
            # Check map dimensions in map callback
            while True:
                x += self.ray_step * np.cos(theta)
                y += self.ray_step * np.sin(theta)
                # print(f"x: {x}, y: {y}")

                distance: float = np.linalg.norm([x - particle[0], y - particle[1]]) * self.resolution
                
                # The point is out of map bounds
                if  0 > x or 0 > y  or x > map_data.shape[0] or y > map_data.shape[1]:
                    # print("Miss!")
                    measurements[index] = -1
                    break

                # Hit obstacle
                elif map_data[int(x), int(y)] > 0:
                    # print("Hit!")   
                    measurements[index] = distance 
                    break

                # If no obstacle is hit within the sensor's range, set measurement to max range
                elif distance >= range_max :
                    # print("Miss!")
                    measurements[index] = -1
                    break

                else:
                    # Update test map with trace
                    self.test_state[int(x), int(y)] = 0

                    pass

        return measurements
    

    def measurement_model_correspondance(self,
                                         particles: np.ndarray,
                                         measurements: np.ndarray,
                                         map: np.ndarray,
                                         ) -> np.ndarray:

        measurements = measurements.copy()
        range_max = self.scanner_info.get("range_max")

        measurements[measurements > range_max] = -1

        for index, particle in enumerate(particles):
            expected_measurements = self.simulate_lidar_measurement(particle, map)

            diff_mask = (measurements != -1) & (expected_measurements != -1)
            diff = np.linalg.norm(measurements[diff_mask] - expected_measurements[diff_mask])

            if np.abs(diff) < 1e-10:
                particles[index, -1] *= 1e10
            else:
                # particles[index, -1] *= np.exp(-diff)
                particles[index, -1] *= 1/diff

            # with np.printoptions(suppress=True):
                # print(f"particle: {particle} | diff: {diff} | weight: {particles[index, -1]}")

        particles[:, -1] = self.normalize_weights(particles[:, -1])

        # print(f"particles: \n{particles}")

        return particles
    

    def normalize_weights(self, weights):
        weights = weights / np.sum(weights)

        return weights
    

    def low_variance_resample(self, 
                              resample_fraction=0.2,
                              ) -> None:
        particles_to_resample = int(self.NUM_PARTICLES * resample_fraction)

        print(f"estimated_particle: {self.estimated_particle}")

        samples = self.sample_n_normal_particles(self.estimated_particle[:3], self.initial_particle_std, particles_to_resample)

        self.particles[:particles_to_resample,:3] = samples
        self.particles[:particles_to_resample, 3] = 1/self.NUM_PARTICLES

        self.particles[:, -1] = self.normalize_weights(self.particles[:, -1])

        return None
    

    def localization_loop(self) -> None:
        if (self.map is None) or (self.scan_ranges is None):
            self.get_logger().info(f"Waiting for map or scan")

            return None
        
        if self.verbose:
            self.get_logger().info(f'Current weight variance: {self.variance} over {self.NUM_PARTICLES} particles')

        # Motion prediction model
        current_time = time.time()
        time_step = current_time - self.previous_prediction_time
        self.motion_model_prediction(self.particles, time_step)
        self.previous_prediction_time = current_time

        # Test map - shows traces made by particles
        self.test_state = np.ones_like(self.state) * 100

        # Measurement correspondence and weight update
        self.measurement_model_correspondance(self.particles, self.scan_ranges, self.state)
        self.resample_particles()

        self.publish_map(self.test_state)

        # Compute and publish estimated pose
        self.publish_estimated_pose()

        self.variance = np.var(self.particles[:, -1])
        # if self.variance < self.VARIANCE_THRESHOLD:
        #     self.low_variance_resample()

        # Create and publish particle array
        self.particle_array_generation(self.particles)

        return None


    def publish_estimated_pose(self) -> None:
        self.estimated_particle = np.average(self.particles[:,:-1], 
                                             axis=0, 
                                             weights=self.particles[:,-1],
                                             )
        
        self.estimated_particle[0] = self.estimated_particle[0] * self.resolution + self.origin.position.x
        self.estimated_particle[1] = self.estimated_particle[1] * self.resolution + self.origin.position.y

        # TODO: Add base_laser_front_link to base_link transform

        estimated_pose: PoseStamped = self.set_posestamped(self.expected_pose, 
                                         [self.estimated_particle[0], self.estimated_particle[1], self.origin.position.z], 
                                         [0.0, 0.0, self.estimated_particle[2]], 
                                         self.MAP_FRAME,
                                         )

        self.estimated_pose_publisher.publish(estimated_pose)


    def publish_map(self, map_array: np.ndarray) -> None:
        map_msg = OccupancyGrid()

        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.width
        map_msg.info.height = self.height

        map_msg.info.origin.position.x = self.origin.position.x
        map_msg.info.origin.position.y = self.origin.position.y

        map_msg.header.stamp = self.timestamp
        map_msg.header.frame_id = self.MAP_FRAME

        map_msg.data = np.ravel(map_array, order='F').tolist()

        self.map_publisher.publish(map_msg)



def main(args=None):
    rclpy.init(args=args)

    node = ParticleFilterLocalization()

    try:
        while rclpy.ok():
            rclpy.spin_once(node)

    except KeyboardInterrupt as e:
        print(e)

        rclpy.shutdown()


if __name__ == '__main__':
    main()
