import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseArray, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ParticleFilterLocalization(Node):
    def __init__(self,
                 particle_count: int = 1000,
                 ) -> None:
        # For testing
        init_ros = False

        # Constants
        self.POSITION_DIMENSIONS = 2
        self.ORIENTATION_DIMENSIONS = 1
        self.NUM_PARTICLES = particle_count

        # Random
        self.ERROR_MEAN = 0
        self.ERROR_LINEAR_STD = 0.5
        self.ERROR_ANGULAR_STD = 0.5

        self.CMD_VEL_TOPIC = "cmd_vel"
        self.MAP_TOPIC = "map"
        self.SCAN_TOPIC = "scan"
        self.PARTICLE_ARRAY_TOPIC = "particle_cloud"

        self.ROBOT_FRAME = "base_link"
        self.MAP_FRAME = "map"
        self.SCANNER_FRAME = "base_laser_front_link"

        # Input
        self.control_input = np.zeros((self.POSITION_DIMENSIONS + \
                                    self.ORIENTATION_DIMENSIONS))

        # Particles
        self.particles = np.zeros((self.NUM_PARTICLES, \
                                   self.POSITION_DIMENSIONS + self.ORIENTATION_DIMENSIONS))

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

        self.tf_buffer = tf2_ros.Buffer() if init_ros else None
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self) if init_ros else None

        # LIDAR params
        self.variance = 1

        return None


    def cmd_vel_callback(self, msg: Twist) -> None:
        """
        Extract linear and angular velocity from cmd_vel message
        """

        self.control_input[0] = msg.linear.x
        self.control_input[1] = msg.linear.y

        self.control_input[2] = msg.angular.z

        return None


    def map_callback(self, msg: OccupancyGrid) -> None:

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

        weights: np.ndarray = np.zeros(particles.shape[0])

        for i, particle in enumerate(self.particles):
            # Implement measurement likelihood calculation based on LiDAR measurements
            # Update particle weight
           weights[i] = self.measurement_likelihood(self.scanner_info.get('range_data'), particle, map)

        return weights
    
    def measurement_likelihood(self, measurements, particle, map):
        likelihood = 1.0
        for expected_measurement in self.simulation_lidar_measurement(particle, map):
            # Assuming Gaussian noise for simplicity
            measurement_difference = measurements - expected_measurement
            measurement_probability = (
                1.0 / np.sqrt(2 * np.pi * self.variance) * np.exp(-0.5 * (measurement_difference / self.variance) ** 2)
            )
            likelihood *= measurement_probability
        return likelihood


    def simulation_lidar_measurement(self,particle, map_data):

        num_beams = (self.scanner_info.get('angle_max') - self.scanner_info.get('angle_min')) / self.scanner_info.get('angle_increment')

        angles = np.linspace(0, 2*np.pi, num_beams, endpoint=False)
        measurements = []

        for angle in angles:
            x = particle[0]
            y = particle[1]
            theta = particle[2] + angle  # Adjust for sensor orientation
            max_range = self.scanner_info.get('range_max')
            min_range = self.scanner_info.get('range_min')

            # Cast a ray from the sensor's position
            # Check map dimensions in map callback
            while 0 <= x < map_data.shape[0] and 0 <= y < map_data.shape[1]:
                if map_data[int(x), int(y)] == 1:  # Hit an obstacle
                    distance = np.sqrt((x - particle[0])**2 + (y - particle[1])**2)
                    measurements.append(min(distance, max_range))
                    break
                x += min_range* np.cos(theta)
                y += min_range * np.sin(theta)

            # If no obstacle is hit within the sensor's range, set measurement to max range
            if 0 <= x < map_data.shape[0] and 0 <= y < map_data.shape[1]:
                measurements.append(max_range)
            else:
                measurements.append(0)  # Out of bounds

        return measurements


        

    def localization_loop(self) -> None:

        ''' call for laser scan 
            convert it into cartesion                           
            call sampling method '''
        pass
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
