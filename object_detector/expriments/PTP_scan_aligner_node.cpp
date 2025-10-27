// scan_aligner_node.cpp
// - MapManagerPair가 sensor_msgs::msg::PointCloud2::SharedPtr를 저장/반환한다고 가정
// - 과거 스캔: 오도메 1차 정렬 → ICP 2차 보정
// - /aligned_history_scans, /aligned_history_scans_icp 퍼블리시

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <cmath>

#include "map_manager_pair.hpp"   // 너의 MapManager(Pair) 정의 (ROS PointCloud2 저장)
#include "icp_refiner.hpp"        // header-only ICP

using std::placeholders::_1;
using std::placeholders::_2;

class ScanAligner : public rclcpp::Node {
public:
  using PXYZI = pcl::PointXYZI;
  using Cloud = pcl::PointCloud<PXYZI>;
  using CloudPtr = Cloud::Ptr;
  using CloudConstPtr = Cloud::ConstPtr;

  ScanAligner() : Node("scan_aligner_node"), map_manager_(10)
  {
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::LaserScan, nav_msgs::msg::Odometry>;

    scan_sub_.subscribe(this, "/scan", rmw_qos_profile_sensor_data);
    odom_sub_.subscribe(this, "/odom", rmw_qos_profile_sensor_data);

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(1000), scan_sub_, odom_sub_);
    sync_->registerCallback(std::bind(&ScanAligner::syncCallback, this, _1, _2));

    current_scan_pub_        = this->create_publisher<sensor_msgs::msg::PointCloud2>("/current_scan_pcl", 10);
    aligned_history_pub_     = this->create_publisher<sensor_msgs::msg::PointCloud2>("/aligned_history_scans", 10);
    aligned_history_icp_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/aligned_history_scans_icp", 10);

    // ICP 파라미터(런타임 override 가능)
    this->declare_parameter<int>("icp.max_iterations", 5);
    this->declare_parameter<double>("icp.max_corr_dist", 0.2);
    this->declare_parameter<double>("icp.trans_eps", 1e-6);
    this->declare_parameter<double>("icp.fit_eps", 1e-5);
    this->declare_parameter<double>("icp.voxel_leaf", 0.2);
    this->declare_parameter<bool>("icp.use_downsample", true);
    this->declare_parameter<bool>("icp.reject_far", false);
    this->declare_parameter<double>("icp.reject_radius", 3.0);

    updateIcpFromParams();

    RCLCPP_INFO(this->get_logger(), "scan_aligner_node with ICP (ROS PointCloud2 store) started.");
  }

private:
  // LaserScan -> 로컬 프레임 PCL
  CloudPtr buildLocalCloud(const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg) {
    CloudPtr local(new Cloud);
    local->reserve(scan_msg->ranges.size());
    for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
      const float r = scan_msg->ranges[i];
      if (!std::isfinite(r)) continue;
      const float th = scan_msg->angle_min + i * scan_msg->angle_increment;
      PXYZI pt;
      pt.x = r * std::cos(th);
      pt.y = r * std::sin(th);
      pt.z = 0.f;
      pt.intensity = 255.f;
      local->push_back(pt);
    }
    return local;
  }

  // geometry_msgs::Pose -> 4x4
  static Eigen::Matrix4d poseToMatrix(const geometry_msgs::msg::Pose& p) {
    Eigen::Quaterniond q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
    Eigen::Matrix3d R = q.normalized().toRotationMatrix();
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3,3>() = R;
    T(0,3) = p.position.x;
    T(1,3) = p.position.y;
    T(2,3) = p.position.z;
    return T;
  }

  void updateIcpFromParams() {
    IcpParams ip;
    ip.max_iterations = this->get_parameter("icp.max_iterations").get_parameter_value().get<int>();
    ip.max_corr_dist  = static_cast<float>(this->get_parameter("icp.max_corr_dist").get_parameter_value().get<double>());
    ip.trans_eps      = this->get_parameter("icp.trans_eps").get_parameter_value().get<double>();
    ip.fit_eps        = this->get_parameter("icp.fit_eps").get_parameter_value().get<double>();
    ip.voxel_leaf     = static_cast<float>(this->get_parameter("icp.voxel_leaf").get_parameter_value().get<double>());
    ip.use_downsample = this->get_parameter("icp.use_downsample").get_parameter_value().get<bool>();
    ip.reject_far     = this->get_parameter("icp.reject_far").get_parameter_value().get<bool>();
    ip.reject_radius  = static_cast<float>(this->get_parameter("icp.reject_radius").get_parameter_value().get<double>());
    icp_ = IcpRefiner(ip);
  }

  void syncCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg,
                    const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg)
  {
    // 필요 시 파라미터 로드
    updateIcpFromParams();

    // (A) 현재 프레임 PCL 생성
    CloudPtr curr_cloud = buildLocalCloud(scan_msg);

    // 현재도 퍼블리시 (시각화용)
    sensor_msgs::msg::PointCloud2 curr_msg;
    pcl::toROSMsg(*curr_cloud, curr_msg);
    curr_msg.header = scan_msg->header;
    current_scan_pub_->publish(curr_msg);

    // (B) MapManager에 현재 프레임 저장: ★ ROS PointCloud2로 저장 ★
    map_manager_.addCloudWithPose(
      std::make_shared<sensor_msgs::msg::PointCloud2>(curr_msg),
      odom_msg->pose.pose
    );

    // (C) 스냅샷 불러와 정렬 수행
    auto pairs = map_manager_.snapshot_pairs();
    if (pairs.size() < 2) return;

    const auto& curr_pose = pairs.back().second;
    Eigen::Matrix4d T_map_curr = poseToMatrix(curr_pose);
    Eigen::Matrix4d T_curr_map = T_map_curr.inverse();

    pcl::PointCloud<PXYZI> all_aligned_odom;
    pcl::PointCloud<PXYZI> all_aligned_icp;

    const size_t frames = pairs.size() - 1;
    double total_fitness = 0.0;
    int    fitness_cnt   = 0;

    for (size_t i = 0; i < frames; ++i) {
      // 과거 스캔 로드: ROS PointCloud2 → PCL 변환
      const auto& hist_scan = pairs[i].first;
      const auto& hist_pose = pairs[i].second;

      // (1) 오도메 기반 상대변환: hist -> curr
      Eigen::Matrix4d T_map_hist  = poseToMatrix(hist_pose);
      Eigen::Matrix4d T_curr_hist = T_curr_map * T_map_hist;

      // (2) 1차 정렬(오도메)
      Cloud hist_in_curr;
      pcl::transformPointCloud(*hist_scan, hist_in_curr, T_curr_hist.cast<float>());

      // 프레임별 intensity (과거일수록 어둡게)
      float intensity_value = static_cast<float>((frames - 1) - i);
      for (auto& pt : hist_in_curr.points) pt.intensity = intensity_value;

      all_aligned_odom += hist_in_curr;

      // (3) ICP 2차 보정
      double fitness = 0.0;
      Eigen::Matrix4f icp_delta = icp_.refine(
          CloudConstPtr(new Cloud(hist_in_curr)),   // src: 오도메 정렬된 과거
          CloudConstPtr(new Cloud(*curr_cloud)),     // tgt: 현재
          Eigen::Matrix4f::Identity(),               // init_delta
          &fitness
      );

      Cloud hist_icp;
      pcl::transformPointCloud(hist_in_curr, hist_icp, icp_delta);
      for (auto& pt : hist_icp.points) pt.intensity = intensity_value;

      all_aligned_icp += hist_icp;

      if (fitness >= 0.0) { total_fitness += fitness; fitness_cnt++; }
    }

    // (D) 퍼블리시
    sensor_msgs::msg::PointCloud2 msg_odom, msg_icp;
    pcl::toROSMsg(all_aligned_odom, msg_odom);
    pcl::toROSMsg(all_aligned_icp, msg_icp);

    msg_odom.header.stamp = scan_msg->header.stamp;
    msg_icp.header.stamp  = scan_msg->header.stamp;
    msg_odom.header.frame_id = scan_msg->header.frame_id;
    msg_icp.header.frame_id  = scan_msg->header.frame_id;

    aligned_history_pub_->publish(msg_odom);
    aligned_history_icp_pub_->publish(msg_icp);

    if (fitness_cnt > 0) {
      double mean_fit = total_fitness / fitness_cnt;
      RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "[ICP] mean fitness=%.6f over %d pairs", mean_fit, fitness_cnt
      );
    }
  }

  message_filters::Subscriber<sensor_msgs::msg::LaserScan> scan_sub_;
  message_filters::Subscriber<nav_msgs::msg::Odometry>     odom_sub_;
  std::shared_ptr<message_filters::Synchronizer<
      message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::LaserScan, nav_msgs::msg::Odometry>>> sync_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr current_scan_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_history_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_history_icp_pub_;

  MapManager map_manager_;   // map_manager_pair.hpp의 타입 (ROS PointCloud2 저장)
  IcpRefiner icp_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ScanAligner>());
  rclcpp::shutdown();
  return 0;
}
