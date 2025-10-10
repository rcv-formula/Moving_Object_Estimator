// obstacle_visualizer_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker.hpp>

class ObstacleVisualizer : public rclcpp::Node {
public:
  ObstacleVisualizer() : Node("obstacle_visualizer") {
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/opponent_odom", 20,
      std::bind(&ObstacleVisualizer::odomCallback, this, std::placeholders::_1));
    
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("Obst_tracker_visualization_marker", 20);
  }

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
    visualization_msgs::msg::Marker marker_msg;
    marker_msg.header = odom_msg->header;
    marker_msg.ns = "obstacle_tracker";
    marker_msg.id = 0;
    marker_msg.action = visualization_msgs::msg::Marker::ADD;
    marker_msg.type = visualization_msgs::msg::Marker::CUBE;
    marker_msg.pose = odom_msg->pose.pose;
    marker_msg.scale.x = 0.3;
    marker_msg.scale.y = 0.3;
    marker_msg.scale.z = 0.05;
    marker_msg.color.a = 1.0;
    marker_msg.color.r = 1.0;
    marker_msg.color.g = 0.0;
    marker_msg.color.b = 0.0;
    
    marker_pub_->publish(marker_msg);
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ObstacleVisualizer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
