#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/header.hpp"

#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Matrix3x3.h"

#include "obstacle_context_msgs/msg/obstacle_cluster.hpp"
#include "obstacle_context_msgs/msg/obstacle_cluster_array.hpp"

class ObstacleClusterContextNode : public rclcpp::Node
{
public:
  ObstacleClusterContextNode()
  : Node("obstacle_cluster_context_node")
  {
    declareParameters();
    loadParameters();

    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&ObstacleClusterContextNode::scanCallback, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_,
      rclcpp::QoS(100),
      std::bind(&ObstacleClusterContextNode::odomCallback, this, std::placeholders::_1));

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&ObstacleClusterContextNode::imuCallback, this, std::placeholders::_1));

    processed_scan_pub_ =
      this->create_publisher<sensor_msgs::msg::LaserScan>(
        processed_scan_topic_,
        rclcpp::SensorDataQoS());

    cluster_pub_ =
      this->create_publisher<obstacle_context_msgs::msg::ObstacleClusterArray>(
        output_topic_,
        rclcpp::SensorDataQoS());

    marker_pub_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(
        marker_topic_,
        rclcpp::QoS(1));

    RCLCPP_INFO(this->get_logger(), "obstacle_cluster_context_node started");
    RCLCPP_INFO(this->get_logger(), "scan_topic: %s", scan_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "odom_topic: %s", odom_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "imu_topic: %s", imu_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "processed_scan_topic: %s", processed_scan_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "output_topic: %s", output_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "marker_topic: %s", marker_topic_.c_str());
  }

private:
  struct OdomPose
  {
    rclcpp::Time stamp;
    double x;
    double y;
    double yaw;
  };

  struct ScanPoint
  {
    size_t index;
    double range;
    double angle;
    double x;
    double y;
  };

  struct TrackState
  {
    double x;
    double y;
    rclcpp::Time stamp;
    OdomPose pose;
  };

  struct Track
  {
    uint32_t id;

    double x;
    double y;
    rclcpp::Time stamp;
    OdomPose pose;

    int observed_count;

    double vx;
    double vy;
    double speed;
    double compensated_displacement;

    uint8_t motion_label;
    int static_hit_count;
    int dynamic_hit_count;

    std::deque<TrackState> history;
  };

  void declareParameters()
  {
    this->declare_parameter<std::string>("scan_topic", "/scan");
    this->declare_parameter<std::string>("odom_topic", "/odom");
    this->declare_parameter<std::string>("imu_topic", "/imu/data");

    this->declare_parameter<std::string>("processed_scan_topic", "/processed_scan");
    this->declare_parameter<std::string>("output_topic", "/obstacle_clusters");
    this->declare_parameter<std::string>("marker_topic", "/obstacle_cluster_markers");

    this->declare_parameter<double>("min_valid_range", 0.15);
    this->declare_parameter<double>("max_valid_range", 10.0);

    this->declare_parameter<bool>("use_roi_filter", true);
    this->declare_parameter<double>("roi_angle_min_deg", -120.0);
    this->declare_parameter<double>("roi_angle_max_deg", 120.0);

    this->declare_parameter<double>("cluster_range_jump", 0.20);
    this->declare_parameter<int>("min_cluster_points", 2);
    this->declare_parameter<double>("min_cluster_width", 0.04);
    this->declare_parameter<double>("max_cluster_width", 0.80);

    this->declare_parameter<double>("association_distance", 0.60);
    this->declare_parameter<double>("track_timeout_sec", 0.50);
    this->declare_parameter<int>("min_observations", 5);
    this->declare_parameter<int>("track_history_size", 7);

    this->declare_parameter<double>("static_speed_threshold", 0.60);
    this->declare_parameter<double>("dynamic_speed_threshold", 1.0);

    this->declare_parameter<bool>("use_imu_motion_gate", true);
    this->declare_parameter<double>("imu_accel_gate_threshold", 1.0);
    this->declare_parameter<double>("imu_yaw_rate_gate_threshold", 0.7);
    this->declare_parameter<double>("imu_motion_gate_hold_sec", 0.25);

    this->declare_parameter<double>("max_odom_age_sec", 0.35);
    this->declare_parameter<double>("max_pose_jump_dist", 1.0);
    this->declare_parameter<double>("max_pose_jump_yaw_deg", 20.0);
    this->declare_parameter<double>("odom_history_sec", 2.0);

    this->declare_parameter<double>("marker_scale", 0.15);
    this->declare_parameter<double>("marker_lifetime_sec", 0.2);
  }

  void loadParameters()
  {
    scan_topic_ = this->get_parameter("scan_topic").as_string();
    odom_topic_ = this->get_parameter("odom_topic").as_string();
    imu_topic_ = this->get_parameter("imu_topic").as_string();

    processed_scan_topic_ = this->get_parameter("processed_scan_topic").as_string();
    output_topic_ = this->get_parameter("output_topic").as_string();
    marker_topic_ = this->get_parameter("marker_topic").as_string();

    min_valid_range_ = this->get_parameter("min_valid_range").as_double();
    max_valid_range_ = this->get_parameter("max_valid_range").as_double();

    use_roi_filter_ = this->get_parameter("use_roi_filter").as_bool();
    roi_angle_min_rad_ = deg2rad(this->get_parameter("roi_angle_min_deg").as_double());
    roi_angle_max_rad_ = deg2rad(this->get_parameter("roi_angle_max_deg").as_double());

    cluster_range_jump_ = this->get_parameter("cluster_range_jump").as_double();
    min_cluster_points_ = this->get_parameter("min_cluster_points").as_int();
    min_cluster_width_ = this->get_parameter("min_cluster_width").as_double();
    max_cluster_width_ = this->get_parameter("max_cluster_width").as_double();

    association_distance_ = this->get_parameter("association_distance").as_double();
    track_timeout_sec_ = this->get_parameter("track_timeout_sec").as_double();
    min_observations_ = this->get_parameter("min_observations").as_int();
    track_history_size_ = this->get_parameter("track_history_size").as_int();

    static_speed_threshold_ = this->get_parameter("static_speed_threshold").as_double();
    dynamic_speed_threshold_ = this->get_parameter("dynamic_speed_threshold").as_double();

    use_imu_motion_gate_ = this->get_parameter("use_imu_motion_gate").as_bool();
    imu_accel_gate_threshold_ = this->get_parameter("imu_accel_gate_threshold").as_double();
    imu_yaw_rate_gate_threshold_ = this->get_parameter("imu_yaw_rate_gate_threshold").as_double();
    imu_motion_gate_hold_sec_ = this->get_parameter("imu_motion_gate_hold_sec").as_double();

    if (static_speed_threshold_ > dynamic_speed_threshold_) {
      RCLCPP_WARN(
        this->get_logger(),
        "static_speed_threshold is greater than dynamic_speed_threshold. swapping thresholds");
      std::swap(static_speed_threshold_, dynamic_speed_threshold_);
    }

    max_odom_age_sec_ = this->get_parameter("max_odom_age_sec").as_double();
    max_pose_jump_dist_ = this->get_parameter("max_pose_jump_dist").as_double();
    max_pose_jump_yaw_rad_ = deg2rad(this->get_parameter("max_pose_jump_yaw_deg").as_double());
    odom_history_sec_ = this->get_parameter("odom_history_sec").as_double();

    marker_scale_ = this->get_parameter("marker_scale").as_double();
    marker_lifetime_sec_ = this->get_parameter("marker_lifetime_sec").as_double();

    if (track_history_size_ < 1) {
      track_history_size_ = 1;
    }

    if (min_observations_ < 1) {
      min_observations_ = 1;
    }
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    OdomPose pose;
    if (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0) {
      pose.stamp = this->get_clock()->now();
      RCLCPP_INFO_ONCE(
        this->get_logger(),
        "odom header stamp is zero. using receive time for odom tracking");
    } else {
      pose.stamp = msg->header.stamp;
    }

    pose.x = msg->pose.pose.position.x;
    pose.y = msg->pose.pose.position.y;
    pose.yaw = yawFromQuaternion(msg->pose.pose.orientation);

    if (!odom_history_.empty()) {
      const OdomPose & prev = odom_history_.back();
      const double dist = std::hypot(pose.x - prev.x, pose.y - prev.y);
      const double dyaw = std::fabs(normalizeAngle(pose.yaw - prev.yaw));

      if (dist > max_pose_jump_dist_ || dyaw > max_pose_jump_yaw_rad_) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          1000,
          "large odom jump detected. dist=%.3f, dyaw=%.3f deg",
          dist,
          dyaw * 180.0 / M_PI);
      }
    }

    odom_history_.push_back(pose);

    while (!odom_history_.empty()) {
      const double age = (pose.stamp - odom_history_.front().stamp).seconds();

      if (age > odom_history_sec_) {
        odom_history_.pop_front();
      } else {
        break;
      }
    }
  }

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan)
  {
    sensor_msgs::msg::LaserScan processed_scan = makeProcessedScan(*scan);
    processed_scan_pub_->publish(processed_scan);

    OdomPose curr_pose;
    const bool has_odom = getInterpolatedOdom(processed_scan.header.stamp, curr_pose);

    if (!has_odom) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        1000,
        "no valid odom for scan time. clusters will be UNKNOWN");
    }

    std::vector<ScanPoint> points = scanToPoints(processed_scan);
    std::vector<std::vector<ScanPoint>> clusters = makeClusters(points);

    obstacle_context_msgs::msg::ObstacleClusterArray out;
    out.header = processed_scan.header;

    for (const auto & cluster : clusters) {
      if (!isValidCluster(cluster)) {
        continue;
      }

      auto msg = makeClusterMsg(processed_scan.header, cluster);
      out.clusters.push_back(msg);
    }

    if (has_odom) {
      updateTracks(out, curr_pose);
    } else {
      setAllClustersUnknown(out);
      removeOldTracks(processed_scan.header.stamp);
    }

    cluster_pub_->publish(out);

    auto marker_array = makeMarkerArray(out);
    marker_pub_->publish(marker_array);
  }

  sensor_msgs::msg::LaserScan makeProcessedScan(
    const sensor_msgs::msg::LaserScan & scan) const
  {
    sensor_msgs::msg::LaserScan out = scan;

    for (size_t i = 0; i < scan.ranges.size(); ++i) {
      const double range = scan.ranges[i];
      const double angle = scan.angle_min + static_cast<double>(i) * scan.angle_increment;

      bool keep = true;

      if (!std::isfinite(range)) {
        keep = false;
      }

      if (range < min_valid_range_ || range > max_valid_range_) {
        keep = false;
      }

      if (range < scan.range_min || range > scan.range_max) {
        keep = false;
      }

      if (use_roi_filter_) {
        if (angle < roi_angle_min_rad_ || angle > roi_angle_max_rad_) {
          keep = false;
        }
      }

      if (!keep) {
        out.ranges[i] = std::numeric_limits<float>::infinity();
      } else {
        out.ranges[i] = static_cast<float>(range);
      }
    }

    return out;
  }

  std::vector<ScanPoint> scanToPoints(
    const sensor_msgs::msg::LaserScan & scan) const
  {
    std::vector<ScanPoint> points;
    points.reserve(scan.ranges.size());

    for (size_t i = 0; i < scan.ranges.size(); ++i) {
      const double range = scan.ranges[i];

      if (!isValidRange(range, scan)) {
        continue;
      }

      const double angle = scan.angle_min + static_cast<double>(i) * scan.angle_increment;

      ScanPoint p;
      p.index = i;
      p.range = range;
      p.angle = angle;
      p.x = range * std::cos(angle);
      p.y = range * std::sin(angle);

      points.push_back(p);
    }

    return points;
  }

  bool isValidRange(
    double range,
    const sensor_msgs::msg::LaserScan & scan) const
  {
    if (!std::isfinite(range)) {
      return false;
    }

    if (range < min_valid_range_ || range > max_valid_range_) {
      return false;
    }

    if (range < scan.range_min || range > scan.range_max) {
      return false;
    }

    return true;
  }

  std::vector<std::vector<ScanPoint>> makeClusters(
    const std::vector<ScanPoint> & points) const
  {
    std::vector<std::vector<ScanPoint>> clusters;

    if (points.empty()) {
      return clusters;
    }

    std::vector<ScanPoint> current;
    current.push_back(points.front());

    for (size_t i = 1; i < points.size(); ++i) {
      const ScanPoint & prev = points[i - 1];
      const ScanPoint & now = points[i];

      const bool adjacent_beam = (now.index == prev.index + 1);
      const double dist = std::hypot(now.x - prev.x, now.y - prev.y);

      if (adjacent_beam && dist < cluster_range_jump_) {
        current.push_back(now);
      } else {
        if (!current.empty()) {
          clusters.push_back(current);
        }

        current.clear();
        current.push_back(now);
      }
    }

    if (!current.empty()) {
      clusters.push_back(current);
    }

    return clusters;
  }

  bool isValidCluster(const std::vector<ScanPoint> & cluster) const
  {
    if (static_cast<int>(cluster.size()) < min_cluster_points_) {
      return false;
    }

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    for (const auto & p : cluster) {
      min_x = std::min(min_x, p.x);
      min_y = std::min(min_y, p.y);
      max_x = std::max(max_x, p.x);
      max_y = std::max(max_y, p.y);
    }

    const double width = std::hypot(max_x - min_x, max_y - min_y);

    if (width < min_cluster_width_) {
      return false;
    }

    if (width > max_cluster_width_) {
      return false;
    }

    return true;
  }

  obstacle_context_msgs::msg::ObstacleCluster makeClusterMsg(
    const std_msgs::msg::Header & header,
    const std::vector<ScanPoint> & cluster) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    ObstacleCluster msg;
    msg.header = header;

    double sum_x = 0.0;
    double sum_y = 0.0;

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();

    double min_range = std::numeric_limits<double>::infinity();
    double max_range = -std::numeric_limits<double>::infinity();

    for (const auto & p : cluster) {
      sum_x += p.x;
      sum_y += p.y;

      min_x = std::min(min_x, p.x);
      min_y = std::min(min_y, p.y);
      max_x = std::max(max_x, p.x);
      max_y = std::max(max_y, p.y);

      min_range = std::min(min_range, p.range);
      max_range = std::max(max_range, p.range);
    }

    const double n = static_cast<double>(cluster.size());

    msg.center_x = static_cast<float>(sum_x / n);
    msg.center_y = static_cast<float>(sum_y / n);

    msg.min_x = static_cast<float>(min_x);
    msg.min_y = static_cast<float>(min_y);
    msg.max_x = static_cast<float>(max_x);
    msg.max_y = static_cast<float>(max_y);

    msg.min_range = static_cast<float>(min_range);
    msg.max_range = static_cast<float>(max_range);

    msg.width = static_cast<float>(std::hypot(max_x - min_x, max_y - min_y));
    msg.point_count = static_cast<uint32_t>(cluster.size());

    msg.track_id = 0;
    msg.motion_label = ObstacleCluster::MOTION_UNKNOWN;

    msg.velocity_x = 0.0f;
    msg.velocity_y = 0.0f;
    msg.speed = 0.0f;
    msg.compensated_displacement = 0.0f;

    msg.risk_weight = 1.2f;

    return msg;
  }

  void updateTracks(
    obstacle_context_msgs::msg::ObstacleClusterArray & clusters_msg,
    const OdomPose & curr_pose)
  {
    const rclcpp::Time now(clusters_msg.header.stamp);

    std::vector<bool> track_matched(tracks_.size(), false);

    for (auto & cluster : clusters_msg.clusters) {
      double best_dist = std::numeric_limits<double>::infinity();
      int best_idx = -1;
      double best_pred_x = 0.0;
      double best_pred_y = 0.0;

      for (size_t i = 0; i < tracks_.size(); ++i) {
        if (track_matched[i]) {
          continue;
        }

        auto pred = transformPrevPointToCurrentFrame(
          tracks_[i].x,
          tracks_[i].y,
          tracks_[i].pose,
          curr_pose);

        const double dist = std::hypot(
          static_cast<double>(cluster.center_x) - pred.first,
          static_cast<double>(cluster.center_y) - pred.second);

        if (dist < best_dist) {
          best_dist = dist;
          best_idx = static_cast<int>(i);
          best_pred_x = pred.first;
          best_pred_y = pred.second;
        }
      }

      if (best_idx >= 0 && best_dist < association_distance_) {
        Track & tr = tracks_[best_idx];

        const double mean_disp = computeMeanCompensatedDisplacement(
          tr,
          static_cast<double>(cluster.center_x),
          static_cast<double>(cluster.center_y),
          curr_pose);

        const double dt = (now - tr.stamp).seconds();

        if (dt > 1e-3) {
          tr.vx = (static_cast<double>(cluster.center_x) - best_pred_x) / dt;
          tr.vy = (static_cast<double>(cluster.center_y) - best_pred_y) / dt;
          tr.speed = std::hypot(tr.vx, tr.vy);
        } else {
          tr.vx = 0.0;
          tr.vy = 0.0;
          tr.speed = 0.0;
        }

        tr.compensated_displacement = mean_disp;
        updateMotionEvidence(tr);
        tr.motion_label = classifyMotionWithHysteresis(tr, now);

        cluster.track_id = tr.id;
        cluster.motion_label = tr.motion_label;
        cluster.velocity_x = static_cast<float>(tr.vx);
        cluster.velocity_y = static_cast<float>(tr.vy);
        cluster.speed = static_cast<float>(tr.speed);
        cluster.compensated_displacement = static_cast<float>(mean_disp);
        cluster.risk_weight = decideRiskWeight(cluster.motion_label);

        tr.x = cluster.center_x;
        tr.y = cluster.center_y;
        tr.stamp = now;
        tr.pose = curr_pose;
        tr.observed_count++;

        pushTrackHistory(
          tr,
          static_cast<double>(cluster.center_x),
          static_cast<double>(cluster.center_y),
          now,
          curr_pose);

        track_matched[best_idx] = true;
      } else {
        Track tr;
        tr.id = next_track_id_++;

        tr.x = cluster.center_x;
        tr.y = cluster.center_y;
        tr.stamp = now;
        tr.pose = curr_pose;

        tr.observed_count = 1;

        tr.vx = 0.0;
        tr.vy = 0.0;
        tr.speed = 0.0;
        tr.compensated_displacement = 0.0;
        tr.motion_label = obstacle_context_msgs::msg::ObstacleCluster::MOTION_UNKNOWN;
        tr.static_hit_count = 0;
        tr.dynamic_hit_count = 0;

        pushTrackHistory(
          tr,
          static_cast<double>(cluster.center_x),
          static_cast<double>(cluster.center_y),
          now,
          curr_pose);

        tracks_.push_back(tr);
        track_matched.push_back(true);

        cluster.track_id = tr.id;
        cluster.motion_label = obstacle_context_msgs::msg::ObstacleCluster::MOTION_UNKNOWN;
        cluster.velocity_x = 0.0f;
        cluster.velocity_y = 0.0f;
        cluster.speed = 0.0f;
        cluster.compensated_displacement = 0.0f;
        cluster.risk_weight = decideRiskWeight(cluster.motion_label);
      }
    }

    removeOldTracks(now);
  }

  void pushTrackHistory(
    Track & tr,
    double x,
    double y,
    const rclcpp::Time & stamp,
    const OdomPose & pose)
  {
    TrackState state;
    state.x = x;
    state.y = y;
    state.stamp = stamp;
    state.pose = pose;

    tr.history.push_back(state);

    while (static_cast<int>(tr.history.size()) > track_history_size_) {
      tr.history.pop_front();
    }
  }

  double computeMeanCompensatedDisplacement(
    const Track & tr,
    double curr_x,
    double curr_y,
    const OdomPose & curr_pose) const
  {
    if (tr.history.empty()) {
      return std::numeric_limits<double>::infinity();
    }

    double sum_dist = 0.0;
    int count = 0;

    for (const auto & h : tr.history) {
      auto prev_in_curr =
        transformPrevPointToCurrentFrame(
          h.x,
          h.y,
          h.pose,
          curr_pose);

      const double dist = std::hypot(
        curr_x - prev_in_curr.first,
        curr_y - prev_in_curr.second);

      sum_dist += dist;
      count++;
    }

    if (count == 0) {
      return std::numeric_limits<double>::infinity();
    }

    return sum_dist / static_cast<double>(count);
  }

  void updateMotionEvidence(Track & tr) const
  {
    if (!std::isfinite(tr.speed)) {
      tr.static_hit_count = 0;
      tr.dynamic_hit_count = 0;
      return;
    }

    if (tr.speed >= dynamic_speed_threshold_) {
      tr.dynamic_hit_count++;
      tr.static_hit_count = 0;
      return;
    }

    if (tr.speed <= static_speed_threshold_) {
      tr.static_hit_count++;
      tr.dynamic_hit_count = 0;
      return;
    }

    tr.static_hit_count = 0;
    tr.dynamic_hit_count = 0;
  }

  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    if (!use_imu_motion_gate_) {
      return;
    }

    const rclcpp::Time stamp(msg->header.stamp);
    const double horizontal_accel =
      std::hypot(msg->linear_acceleration.x, msg->linear_acceleration.y);
    const double yaw_rate = std::fabs(msg->angular_velocity.z);

    if (
      horizontal_accel >= imu_accel_gate_threshold_ ||
      yaw_rate >= imu_yaw_rate_gate_threshold_)
    {
      last_imu_motion_spike_stamp_ = stamp;
      has_imu_motion_spike_ = true;
    }
  }

  uint8_t classifyMotionWithHysteresis(
    const Track & tr,
    const rclcpp::Time & now) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    if (tr.observed_count < min_observations_) {
      return ObstacleCluster::MOTION_UNKNOWN;
    }

    if (static_cast<int>(tr.history.size()) < min_observations_) {
      return ObstacleCluster::MOTION_UNKNOWN;
    }

    if (!std::isfinite(tr.speed)) {
      return ObstacleCluster::MOTION_UNKNOWN;
    }

    const int confirm_count = std::max(1, min_observations_);

    const bool ego_motion_unstable = isEgoMotionUnstable(now);

    if (tr.motion_label == ObstacleCluster::STATIC) {
      if (ego_motion_unstable) {
        return ObstacleCluster::STATIC;
      }

      if (tr.dynamic_hit_count >= confirm_count) {
        return ObstacleCluster::DYNAMIC;
      }

      return ObstacleCluster::STATIC;
    }

    if (tr.motion_label == ObstacleCluster::DYNAMIC) {
      if (tr.static_hit_count >= confirm_count) {
        return ObstacleCluster::STATIC;
      }

      return ObstacleCluster::DYNAMIC;
    }

    if (!ego_motion_unstable && tr.dynamic_hit_count >= confirm_count) {
      return ObstacleCluster::DYNAMIC;
    }

    if (tr.static_hit_count >= confirm_count) {
      return ObstacleCluster::STATIC;
    }

    return ObstacleCluster::MOTION_UNKNOWN;
  }

  bool isEgoMotionUnstable(const rclcpp::Time & now) const
  {
    if (!use_imu_motion_gate_ || !has_imu_motion_spike_) {
      return false;
    }

    const double age = (now - last_imu_motion_spike_stamp_).seconds();
    return age >= 0.0 && age <= imu_motion_gate_hold_sec_;
  }

  float decideRiskWeight(uint8_t motion_label) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    if (motion_label == ObstacleCluster::DYNAMIC) {
      return 1.5f;
    }

    if (motion_label == ObstacleCluster::MOTION_UNKNOWN) {
      return 1.3f;
    }

    return 1.0f;
  }

  void setAllClustersUnknown(
    obstacle_context_msgs::msg::ObstacleClusterArray & clusters_msg) const
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    for (auto & cluster : clusters_msg.clusters) {
      cluster.track_id = 0;
      cluster.motion_label = ObstacleCluster::MOTION_UNKNOWN;
      cluster.velocity_x = 0.0f;
      cluster.velocity_y = 0.0f;
      cluster.speed = 0.0f;
      cluster.compensated_displacement = 0.0f;
      cluster.risk_weight = 1.3f;
    }
  }

  void removeOldTracks(const rclcpp::Time & now)
  {
    std::vector<Track> alive_tracks;
    alive_tracks.reserve(tracks_.size());

    for (const auto & tr : tracks_) {
      const double age = (now - tr.stamp).seconds();

      if (age < track_timeout_sec_) {
        alive_tracks.push_back(tr);
      }
    }

    tracks_ = alive_tracks;
  }

  std::pair<double, double> transformPrevPointToCurrentFrame(
    double px,
    double py,
    const OdomPose & prev_pose,
    const OdomPose & curr_pose) const
  {
    const double cos_prev = std::cos(prev_pose.yaw);
    const double sin_prev = std::sin(prev_pose.yaw);

    const double wx = prev_pose.x + cos_prev * px - sin_prev * py;
    const double wy = prev_pose.y + sin_prev * px + cos_prev * py;

    const double dx = wx - curr_pose.x;
    const double dy = wy - curr_pose.y;

    const double cos_curr = std::cos(curr_pose.yaw);
    const double sin_curr = std::sin(curr_pose.yaw);

    const double cx = cos_curr * dx + sin_curr * dy;
    const double cy = -sin_curr * dx + cos_curr * dy;

    return {cx, cy};
  }

  bool getInterpolatedOdom(
    const rclcpp::Time & target_time,
    OdomPose & out_pose) const
  {
    if (odom_history_.empty()) {
      return false;
    }

    if (odom_history_.size() == 1) {
      const double age = std::fabs((target_time - odom_history_.front().stamp).seconds());

      if (age <= max_odom_age_sec_) {
        out_pose = odom_history_.front();
        return true;
      }

      return false;
    }

    if (target_time < odom_history_.front().stamp) {
      return false;
    }

    if (target_time > odom_history_.back().stamp) {
      const double age = (target_time - odom_history_.back().stamp).seconds();

      if (age <= max_odom_age_sec_) {
        out_pose = odom_history_.back();
        out_pose.stamp = target_time;
        return true;
      }

      return false;
    }

    for (size_t i = 1; i < odom_history_.size(); ++i) {
      const auto & p0 = odom_history_[i - 1];
      const auto & p1 = odom_history_[i];

      if (p0.stamp <= target_time && target_time <= p1.stamp) {
        const double total_dt = (p1.stamp - p0.stamp).seconds();

        if (total_dt < 1e-6) {
          out_pose = p1;
          out_pose.stamp = target_time;
          return true;
        }

        const double ratio = (target_time - p0.stamp).seconds() / total_dt;

        out_pose.stamp = target_time;
        out_pose.x = p0.x + ratio * (p1.x - p0.x);
        out_pose.y = p0.y + ratio * (p1.y - p0.y);
        out_pose.yaw = interpolateYaw(p0.yaw, p1.yaw, ratio);

        return true;
      }
    }

    return false;
  }

  visualization_msgs::msg::MarkerArray makeMarkerArray(
    const obstacle_context_msgs::msg::ObstacleClusterArray & clusters_msg)
  {
    using obstacle_context_msgs::msg::ObstacleCluster;

    visualization_msgs::msg::MarkerArray marker_array;

    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header = clusters_msg.header;
    delete_marker.header.stamp.sec = 0;
    delete_marker.header.stamp.nanosec = 0;
    delete_marker.ns = "obstacle_clusters";
    delete_marker.id = 0;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);

    int id = 1;

    for (const auto & cluster : clusters_msg.clusters) {
      visualization_msgs::msg::Marker marker;

      marker.header = clusters_msg.header;
      marker.header.stamp.sec = 0;
      marker.header.stamp.nanosec = 0;
      marker.ns = "obstacle_clusters";
      marker.id = id++;

      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;

      marker.pose.position.x = cluster.center_x;
      marker.pose.position.y = cluster.center_y;
      marker.pose.position.z = 0.15;

      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;

      marker.scale.x = marker_scale_;
      marker.scale.y = marker_scale_;
      marker.scale.z = marker_scale_;

      if (cluster.motion_label == ObstacleCluster::STATIC) {
        // static: blue
        marker.color.r = 0.1f;
        marker.color.g = 0.3f;
        marker.color.b = 1.0f;
        marker.color.a = 1.0f;
      } else if (cluster.motion_label == ObstacleCluster::DYNAMIC) {
        // dynamic: red
        marker.color.r = 1.0f;
        marker.color.g = 0.1f;
        marker.color.b = 0.1f;
        marker.color.a = 1.0f;
      } else {
        // unknown: green
        marker.color.r = 0.1f;
        marker.color.g = 1.0f;
        marker.color.b = 0.1f;
        marker.color.a = 1.0f;
      }

      marker.lifetime = rclcpp::Duration::from_seconds(marker_lifetime_sec_);

      marker_array.markers.push_back(marker);
    }

    return marker_array;
  }

  double yawFromQuaternion(
    const geometry_msgs::msg::Quaternion & q_msg) const
  {
    tf2::Quaternion q(
      q_msg.x,
      q_msg.y,
      q_msg.z,
      q_msg.w);

    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;

    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    return yaw;
  }

  double normalizeAngle(double a) const
  {
    while (a > M_PI) {
      a -= 2.0 * M_PI;
    }

    while (a < -M_PI) {
      a += 2.0 * M_PI;
    }

    return a;
  }

  double interpolateYaw(double yaw0, double yaw1, double ratio) const
  {
    const double diff = normalizeAngle(yaw1 - yaw0);
    return normalizeAngle(yaw0 + ratio * diff);
  }

  double deg2rad(double deg) const
  {
    return deg * M_PI / 180.0;
  }

private:
  std::string scan_topic_;
  std::string odom_topic_;
  std::string imu_topic_;

  std::string processed_scan_topic_;
  std::string output_topic_;
  std::string marker_topic_;

  double min_valid_range_;
  double max_valid_range_;

  bool use_roi_filter_;
  double roi_angle_min_rad_;
  double roi_angle_max_rad_;

  double cluster_range_jump_;
  int min_cluster_points_;
  double min_cluster_width_;
  double max_cluster_width_;

  double association_distance_;
  double track_timeout_sec_;
  int min_observations_;
  int track_history_size_;

  double static_speed_threshold_;
  double dynamic_speed_threshold_;

  bool use_imu_motion_gate_;
  double imu_accel_gate_threshold_;
  double imu_yaw_rate_gate_threshold_;
  double imu_motion_gate_hold_sec_;

  double max_odom_age_sec_;
  double max_pose_jump_dist_;
  double max_pose_jump_yaw_rad_;
  double odom_history_sec_;

  double marker_scale_;
  double marker_lifetime_sec_;

  std::deque<OdomPose> odom_history_;

  bool has_imu_motion_spike_ = false;
  rclcpp::Time last_imu_motion_spike_stamp_{0, 0, RCL_ROS_TIME};

  std::vector<Track> tracks_;
  uint32_t next_track_id_ = 1;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr processed_scan_pub_;
  rclcpp::Publisher<obstacle_context_msgs::msg::ObstacleClusterArray>::SharedPtr cluster_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ObstacleClusterContextNode>());
  rclcpp::shutdown();
  return 0;
}
