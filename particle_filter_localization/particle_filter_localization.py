import rclpy
from rclpy.node import Node


class ParticleFilterLocalization(Node):
    def __init__(self):

        # Setup subscribers and publishers
        
        return


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
