import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseArray, TransformStamped, Twist, PoseStamped, Pose
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf_transformations import quaternion_from_euler

class ParticleFilterLocalization(Node):
    def __init__(self,
                 particle_count: int = 1000,
                 ) -> None:
        # For testing
        init_ros = False
        self.verbose = True

        # Flag to initialize sampler
        self.initializer = True

        # Constants
        self.POSITION_DIMENSIONS = 2
        self.ORIENTATION_DIMENSIONS = 1
        self.WEIGHT_DIMENSION = 1
        self.NUM_PARTICLES = particle_count

        
        # Random
        self.ERROR_MEAN = 0
        self.ERROR_LINEAR_STD = 0.5
        self.ERROR_ANGULAR_STD = 0.5

        self.CMD_VEL_TOPIC = "cmd_vel"
        self.MAP_TOPIC = "map"
        self.SCAN_TOPIC = "scan"
        self.PARTICLE_ARRAY_TOPIC = "particle_cloud"
        self.ESTIMATED_PARTICLE = "pose_estimated"
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
        self.time_delta = 1.0

         # LIDAR params
        self.variance = 1
        self.step = 1

        # expected pose
        self.expected_pose = self.set_posestamped(
                        PoseStamped(),
                        [0.0, 0.0, 0.0], 
                        [0.0, 0.0, 0.0],
                        self.ROBOT_FRAME
                        ) if init_ros else None

        # Setup subscribers and publishers
        
        # Command input subscriber
        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            self.CMD_VEL_TOPIC,
            self.cmd_vel_callback,
            10,
            ) if init_ros else None

        # Map subscriber
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            self.MAP_TOPIC,
            self.map_callback,
            qos_profile=rclpy.qos.qos_profile_sensor_data,
            ) if init_ros else None

        self.width = None
        self.height = None
       
        # scan subscriber
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            self.SCAN_TOPIC,
            self.scan_callback,
            10,
            ) if init_ros else None
        self.scanner_info: dict

        # Particle publisher (for display in RViz)
        self.particle_array_publisher = self.create_publisher(
            PoseArray,
            self.PARTICLE_ARRAY_TOPIC,
            10) if init_ros else None

        self.estimated_pose_publisher = self.create_publisher(PoseStamped,
                                                              self.ESTIMATED_PARTICLE,
                                                              10) if init_ros else None
        
        self.tf_buffer = tf2_ros.Buffer() if init_ros else None
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self) if init_ros else None

        self.sampler_initilizer()
 

    def cmd_vel_callback(self, msg: Twist) -> None:
        """
        Extract linear and angular velocity from cmd_vel message
        """

        self.control_input[0] = msg.linear.x
        self.control_input[1] = msg.linear.y

        self.control_input[2] = msg.angular.z

        return None


    def map_callback(self, msg: OccupancyGrid) -> None:
        self.get_logger().info(f'I am recieving the map...')

        self.map = msg.data
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin

        self.get_logger().info(f'Map width is: {self.width} height is: {self.height} Resolution is {self.resolution}')

        # Converting the map to a 2D array
        self.state = np.reshape(self.map,(self.width,self.height),order='F')

        return None


    def sampler_initilizer (self) -> None:

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

        self.initializer = False  

        return None
    
    def scan_callback(self, msg: LaserScan) -> None:

        if not self.lidar_init:

            self.scanner_info.update({
                "frame_id" : msg.header.frame_id,
                "angle_min" : msg.angle_min,
                "angle_max" : msg.angle_max,
                "range_min" : msg.range_min,
                "range_max" : msg.range_max,
                "angle_increment" : msg.angle_increment,
                "range_data" : msg.ranges
                })

            self.tf_lidar_wrt_robot: TransformStamped = self.tf_buffer.lookup_transform(
                                                    self.ROBOT_FRAME, self.SCANNER_FRAME, rclpy.time.Time())

            self.lidar_init = True

            self.get_logger().info(f"LiDAR parameters initialized")

        return None
    
    def set_posestamped(self, pose:PoseStamped, position, orientation_euler, frame_id):
        '''
        Sets the fields of a PoseStamped object
        '''
        pose.header.frame_id = frame_id
        
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]

        orientation_quat = quaternion_from_euler(*orientation_euler)

        pose.pose.orientation.x = orientation_quat[0]
        pose.pose.orientation.y = orientation_quat[1]
        pose.pose.orientation.z = orientation_quat[2]
        pose.pose.orientation.w = orientation_quat[3]

        if self.verbose:
            self.get_logger().info(f"pose set: {pose}")

        return pose

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

        return particles


    def measurement_model_correspondance(self,
                                         particles: np.ndarray,
                                         map: np.ndarray,
                                         ) -> np.ndarray:


        for i, particle in enumerate(particles):
            # Implement measurement likelihood calculation based on LiDAR measurements
            # Update particle weight
           likelihood = self.measurement_likelihood(self.scanner_info.get('range_data'), particle, map)
           particle[3] *= likelihood

        particles[:,3] = particles[:,3] / np.sum(particles[:,3])
        
        return particles
    
    def measurement_likelihood(self, measurements, particle, map):

        expected_measurement = self.simulation_lidar_measurement(particle, map)
            # Assuming Gaussian noise for simplicity
        measurement_difference = np.sum(np.square(measurements - expected_measurement))/np.size(measurements)

        measurement_probability = (
            1.0 / np.sqrt(2 * np.pi * self.variance) * np.exp(-0.5 * (measurement_difference / self.variance) ** 2)
        )
    
        return measurement_probability

    def particle_array_generation(self):

        particle_array = PoseArray()
        particle_array.header.frame_id = self.ODOM
        for particle in self.particle_array:
            pose = Pose()
            pose.position.x = particle[0] * self.resolution + self.origin.position.x
            pose.position.y = particle[1] * self.resolution + self.origin.position.y
            pose.position.z = particle[2] * self.resolution + self.origin.position.z

            orientation_quat = quaternion_from_euler(*[0.0,0.0,particle[2]])

            pose.pose.orientation.x = orientation_quat[0]
            pose.pose.orientation.y = orientation_quat[1]
            pose.pose.orientation.z = orientation_quat[2]
            pose.pose.orientation.w = orientation_quat[3]

            particle_array.poses.insert(pose)

        if self.verbose:
            self.get_logger().info(f"pose array set: {particle_array}")
        
        self.particle_array_publisher.publish(particle_array)


    def simulation_lidar_measurement(self,particle, map_data):

        num_beams = (self.scanner_info.get('angle_max') - self.scanner_info.get('angle_min')) / self.scanner_info.get('angle_increment')

        angles = np.linspace(0, 2*np.pi, num_beams, endpoint=False)
        measurements = []

        for angle in angles:
            x = particle[0]
            y = particle[1]
            theta = particle[2] + angle  # TODO: Don't know abouit; Adjust for sensor orientation
            max_range = self.scanner_info.get('range_max')
            min_range = self.scanner_info.get('range_min')

            # Cast a ray from the sensor's position
            # Check map dimensions in map callback
            while 0 <= x < map_data.shape[0] and 0 <= y < map_data.shape[1]:
                if map_data[int(x), int(y)] == 1:  # Hit an obstacle
                    distance = np.sqrt((x - particle[0])**2 + (y - particle[1])**2)
                    measurements.append(min(distance, max_range))
                    break
                
                x += self.step * np.cos(theta)
                y += self.step * np.sin(theta)

            # If no obstacle is hit within the sensor's range, set measurement to max range
            if 0 <= x < map_data.shape[0] and 0 <= y < map_data.shape[1]:
                measurements.append(max_range)
            else:
                measurements.append(0)  # Out of bounds

        return measurements

    def resampling():

        return None

        

    def localization_loop(self) -> None:

        ''' call for laser scan 
            convert it into cartesion                           
            call sampling method '''
        x_est = 0.0
        y_est = 0.0
        theta_est = 0.0 
            
        
        while np.allclose(self.variance,0):

            self.particles = self.motion_model_prediction(self.particles,self.time_delta)
            self.particles = self.measurement_model_correspondance(self.particles,self.state)
            # TODO: call sampling 
            
            for particle in self.particles:
                x_est += particle[0] * particle[3]
                y_est += particle[1] * particle[3]
                theta_est += particle[2] * particle[3]

            self.variance = np.var(self.particles[3])

        self.estimated_pose_publisher(self.set_posestamped(self.expected_pose,[x_est,y_est,0.0],[0.0, 0.0, theta_est],self.ROBOT_FRAME))
        self.particle_array_generation()

            
        return None



def main(args=None):
    rclpy.init(args=args)

    node = ParticleFilterLocalization()

    try:
        while rclpy.ok():
            rclpy.spin_once(node)

    except Exception as e:
        print(e)

        rclpy.shutdown()


if __name__ == '__main__':
    main()
