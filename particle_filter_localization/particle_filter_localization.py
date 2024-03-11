import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseArray, TransformStamped, Twist, PoseStamped, Pose, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import tf_transformations as tf
from tf_transformations import quaternion_from_euler, euler_from_quaternion

class ParticleFilterLocalization(Node):
    def __init__(self,
                 particle_count: int = 20,
                 ) -> None:
        super().__init__(node_name="ParticleFilterLocalization")
        # For testing
        init_ros = True
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
        self.ERROR_LINEAR_STD = 1.0
        self.ERROR_ANGULAR_STD = 0.5
        self.VARIANCE_THRESHOLD = 1e-3

        self.CMD_VEL_TOPIC = "/cmd_vel"
        self.MAP_TOPIC = "/map_loaded"
        self.SCAN_TOPIC = "/scan"
        self.POSE_TOPIC = "/pose"
        self.PARTICLE_ARRAY_TOPIC = "/particle_cloud"
        self.ESTIMATED_POSE_TOPIC = "/pose_estimated"
        self.ROBOT_FRAME = "base_link"
        self.ODOM = "odom"
        self.MAP_FRAME = "map"
        self.SCANNER_FRAME = "base_laser_front_link"
        

        # Input
        self.control_input = np.zeros((self.POSITION_DIMENSIONS + \
                                    self.ORIENTATION_DIMENSIONS))

        # Particles
        self.particles = np.zeros((self.NUM_PARTICLES, \
                                   self.POSITION_DIMENSIONS + self.ORIENTATION_DIMENSIONS + self.WEIGHT_DIMENSION))
        
        
        self.origin = Pose()

        #time delta
        self.filter_time_step = 0.5

         # LIDAR params
        self.scanner_info = {}
        self.variance = 1.0
        self.ray_step = 1.5
        self.RESAMPLE_FRACTION = 0.2
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
        
        # Map
        self.map = None
        self.width = None
        self.height = None
        self.timestamp = None

        self.start_localization = False

        # Setup subscribers and publishers
        if init_ros:
            # self.tf_buffer = tf2_ros.Buffer()
            # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

            # Command input subscriber
            self.cmd_vel_subscriber = self.create_subscription(
                Twist,
                self.CMD_VEL_TOPIC,
                self.cmd_vel_callback,
                10,
                )

            # Map subscriber
            self.map_subscriber = self.create_subscription(
                OccupancyGrid,
                self.MAP_TOPIC,
                self.map_callback,
                10,
                # qos_profile=rclpy.qos.qos_profile_sensor_data,
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
        
        # TODO: transform scan points
            
    
    def pose_callback(self, msg: PoseWithCovarianceStamped):
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

        # self.particles = np.zeros((1, 4))
        # self.particles[0, :-1] = self.robot_pose_particle
 

    def cmd_vel_callback(self, msg: Twist) -> None:
        """
        Extract linear and angular velocity from cmd_vel message
        """
        self.control_input[0] = msg.linear.x * 4
        self.control_input[1] = msg.linear.y * 4

        self.control_input[2] = msg.angular.z / 4
        # self.start_localization = True

        return None


    def map_callback(self, msg: OccupancyGrid) -> None:
        self.get_logger().info(f'I am recieving the map...')

        self.map = np.array(msg.data, dtype=np.int8)
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin
        self.timestamp = msg.header.stamp

        self.get_logger().info(f'Map width is: {self.width} height is: {self.height} Resolution is {self.resolution}')

        # Converting the map to a 2D array
        self.state = np.reshape(self.map,(self.width,self.height),order='F')

        self.sampler_initializer()
        

        return None


    def sampler_initializer (self) -> None:

        if self.initializer:
            # x = np.linspace(0.0,self.width,num=round(self.width/self.resolution))
            # y = np.linspace(0.0 , self.height, num= round(self.height/self.resolution))
            # nx,ny = np.meshgrid(x,y)
            # grid_points = np.dstack((nx,ny))
            # shape = np.shape(grid_points)
            # samples = grid_points.reshape(shape[0]*shape[1],2)
            # theta = theta[np.newaxis].T

            x = np.random.uniform(0.0,self.width,self.NUM_PARTICLES)
            y = np.random.uniform(0.0,self.height,self.NUM_PARTICLES)
            theta = np.random.uniform(-np.pi,np.pi,size=self.NUM_PARTICLES)

            samples = np.vstack((x,y,theta)).T
            
            self.particles[:,:-1] = samples
            self.particles[:,-1] = 1/self.NUM_PARTICLES
        
            if self.verbose:
                self.get_logger().info(f'Sample particles initialized\n{self.particles}')

        self.initializer = False  

        return None


    def sample_n_particles(self, n):
        x = np.random.uniform(0.0, self.width, n)
        y = np.random.uniform(0.0, self.height, n)
        theta = np.random.uniform(-np.pi, np.pi, size=n)

        samples = np.vstack((x,y,theta)).T

        return samples

    
    def scan_callback(self, msg: LaserScan) -> None:

        if not self.lidar_init:

            # self.tf_lidar_wrt_robot: TransformStamped = self.tf_buffer.lookup_transform(
                                                    # self.ROBOT_FRAME, self.SCANNER_FRAME, rclpy.time.Time())
            
            self.scanner_info.update({
                "frame_id" : msg.header.frame_id,
                "angle_min" : msg.angle_min,
                "angle_max" : msg.angle_max,
                "range_min" : msg.range_min,
                "range_max" : msg.range_max,
                "angle_increment" : msg.angle_increment,
                "num_scans": len(msg.ranges),
                })
            
            # scan_cartesian = self.convert_scan_to_cartesian(msg.ranges)
            # scan_cartesian = self.transform_coordinates(self.ROBOT_FRAME, \
            #                                 self.SCANNER_FRAME, \
            #                                 scan_cartesian)
            # self.scanner_info.update({"range_data" : scan_cartesian})

            self.lidar_init = True
            self.get_logger().info(f"LiDAR parameters initialized: \n{self.scanner_info}")

        self.scan_ranges = np.array(msg.ranges)

        return None
    
    def get_homogenous_transformation(self, transform:TransformStamped):
        '''
        Return the equivalent homogenous transform of a TransformStamped object
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

    
    def calculate_transform(self, target_frame, source_frame):
        '''
        Calculates the transform from the source frame to the target frame
        '''
        transform = self.tf_buffer.lookup_transform(target_frame, source_frame, \
                                                    rclpy.time.Time())
        
        if self.verbose:
            self.get_logger().info(f"transform btw target {target_frame} and " + \
                                   f"source {source_frame}: {transform}")

        return transform


    def set_posestamped(self, pose:PoseStamped, position, orientation_euler, frame_id):
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

        # if self.verbose:
        #     self.get_logger().info(f"pose set: {pose}")

        return pose
    
    def transform_posestamped(self, pose_object:PoseStamped, frame_id):
        '''
        Transforms pose_object to frame_id
        '''

        transform = self.calculate_transform(frame_id, pose_object.header.frame_id)
        transformation_matrix = self.get_homogenous_transformation(transform)
        # transformed_pose = self.tf_buffer.transform(pose_object, frame_id, \
        #                                             rclpy.duration.Duration(seconds=0.05))

        point = np.array([pose_object.pose.position.x, pose_object.pose.position.y, pose_object.pose.position.z,1.0])      
        point = np.matmul(transformation_matrix, point)[0:3]

        pose_object.pose.position.x = point[0]
        pose_object.pose.position.y = point[1]
        pose_object.pose.position.z = point[2]
        pose_object.header.frame_id = frame_id

        return pose_object

    def motion_model_prediction(self,
                                particles: np.ndarray,
                                time_delta: float,
                                ) -> np.ndarray:
        # Update position of particles
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

        # if self.verbose:
        #     self.get_logger().info(f'Motion model running\n{particles}')

        return particles


    def measurement_model_correspondance(self,
                                         particles: np.ndarray,
                                         map: np.ndarray,
                                         ) -> np.ndarray:

        measurements = self.scan_ranges
        max_range = self.scanner_info.get('range_max')
        measurements = np.where(np.isinf(measurements), max_range, measurements)

        for particle in particles:
            # Implement measurement likelihood calculation based on LiDAR measurements
            # Update particle weight
           likelihood = self.measurement_likelihood(measurements, particle, map)
           particle[3] = likelihood

        #    if self.verbose:
            # self.get_logger().info(f'weight.....{particle[3]}')

        particles[:,3] = particles[:,3] / np.sum(particles[:,3])

        return particles
    

    def measurement_likelihood(self, measurement, particle, map) -> float:
        # self.get_logger().info(f"particle: {particle}") 
        # self.get_logger().info(f"measurement: \n{measurement}") 

        expected_measurement = self.simulation_lidar_measurement(particle, map)
        
        # self.get_logger().info(f"expected_measurement: \n{expected_measurement}")
        
        # Assuming Gaussian noise for simplicity
        measurement_difference = np.sum((measurement - expected_measurement)**2)
        # self.get_logger().info(f"measurement_difference: {measurement_difference}")

        # measurement_probability = np.exp(-measurement_difference)
        measurement_probability = 1 / measurement_difference
        # self.get_logger().info(f'Measurement likelihood: {measurement_probability}')

        return measurement_probability


    def particle_array_generation(self, particles):

        # if self.verbose:
        #     self.get_logger().info(f'Particle array getting generated.....')

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

        # if self.verbose:
        #     self.get_logger().info(f"pose array set: {particle_array}")
        
        self.particle_array_publisher.publish(particle_array)


    def simulation_lidar_measurement(self, particle, map_data):
        angle_max = self.scanner_info.get('angle_max')
        angle_min = self.scanner_info.get('angle_min')
        angle_inc = self.scanner_info.get('angle_increment')
        max_range = self.scanner_info.get('range_max') / self.resolution
        num_scans = self.scanner_info.get("num_scans")

        # if self.verbose:
        #     self.get_logger().info('Simulation Lidar measurement running.......')
        #     self.get_logger().info(f'angle_min, angle_max, angle_inc,{angle_max},{angle_min},{angle_inc}')

        measurements = []

        for scan_index in range(num_scans):
            x = particle[0]
            y = particle[1]
            theta = particle[2] - (angle_max - (scan_index * angle_inc))

            # Cast a ray from the sensor's position
            # Check map dimensions in map callback
            while True:
                # print(f"x: {x}, y: {y}")
                distance = np.linalg.norm([x - particle[0], y - particle[1]])

                # the point is out of map bounds
                if  0 > x or 0 > y  or x > map_data.shape[0] or y > map_data.shape[1]:
                    # print("Miss!")
                    measurements.append(max_range)
                    break

                # If no obstacle is hit within the sensor's range, set measurement to max range
                elif distance >= max_range :
                    # print("Miss!")
                    measurements.append(max_range)
                    break

                # Hit an obstacle 
                elif map_data[int(x), int(y)] > 0:
                    # print("Hit!")   
                    measurements.append(min(distance, max_range))
                    break

                else:
                    # Update test map with trace
                    self.test_state[int(x), int(y)] = 0

                    pass

    
                x += self.ray_step * np.cos(theta)
                y += self.ray_step * np.sin(theta)

        measurements = np.array(measurements) * self.resolution
            
        return measurements
    

    def resampling(self):
        # if self.verbose:
        #     self.get_logger().info(f'Resampling running.......{self.particles[:,-1]}')
        
        indices = np.random.choice(range(self.NUM_PARTICLES),self.NUM_PARTICLES, p = self.particles[:,-1])
        self.particles = self.particles[indices]
    
        return None
        

    def low_variance_resample(self, resample_fraction=0.2):
        # position_variance = np.linalg.norm(np.var(particles[:, :2], axis=1)) \
        #                         * (self.resolution) 

        # self.get_logger().info(f"position_variance: {position_variance}")
        particles_to_resample = int(self.NUM_PARTICLES * resample_fraction)

        samples = self.sample_n_particles(particles_to_resample)

        self.particles[:particles_to_resample,:3] = samples
        self.particles[:, 3] = self.particles[:, 3] / np.sum(self.particles[:, 3])


    def localization_loop(self) -> None:

        ''' call for laser scan 
            convert it into cartesion                           
            call sampling method '''
            
        # try: 
        # if not self.start_localization or not self.map:
        #     return None
        
        if (self.map is None) or (self.scan_ranges is None):
            self.get_logger().info(f"Waiting for map or scan")

            return None
        
        if self.verbose:
            self.get_logger().info(f'In localization loop with variance: {self.variance}')

        self.motion_model_prediction(self.particles,self.filter_time_step)

        # Test map
        self.test_state = np.ones_like(self.state) * 100

        self.particles = self.measurement_model_correspondance(self.particles, self.state)
        # self.get_logger().info(f'After measurement model: \n{self.particles}')

        # Publish test map
        self.publish_map(self.test_state)

        self.resampling()

        # map_particles = self.particles.copy()
        # map_particles[:, 0] = map_particles[:, 0] * self.resolution + self.origin.position.x
        # map_particles[:, 1] = map_particles[:, 1] * self.resolution + self.origin.position.y

        self.variance = np.var(self.particles[:,3])
        if self.variance < self.VARIANCE_THRESHOLD:
            self.low_variance_resample(resample_fraction=self.RESAMPLE_FRACTION)

        # x_est, y_est, theta_est = np.average(self.particles[:,:-1], axis=0, weights=self.particles[:,-1])
        # x_est = x_est * self.resolution + self.origin.position.x
        # y_est = y_est * self.resolution + self.origin.position.y
        # z_est = self.origin.position.z

        # if self.verbose:
        #     self.get_logger().info(f'Estimated values: {x_est},{y_est},{theta_est}') 
        #     self.get_logger().info(f'Variance of the weights: {self.variance}')
        #     self.get_logger().info(f'Publishing estimated pose.....')

        # curr_pose = self.set_posestamped(self.expected_pose,[x_est,y_est,z_est],[0.0, 0.0, theta_est],self.MAP_FRAME)
        # # curr_pose = self.transform_posestamped(curr_pose,self.MAP_FRAME)
        
        # self.estimated_pose_publisher.publish(curr_pose)
        
        self.particle_array_generation(self.particles)

        return None
    

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
    rclpy.spin(node)
    rclpy.shutdown()

    # try:
    #     while rclpy.ok():
    #         rclpy.spin_once(node)

    # except Exception as e:
    #     print(e)

    #     rclpy.shutdown()


if __name__ == '__main__':
    main()
