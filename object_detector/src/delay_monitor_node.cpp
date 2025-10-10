#include "rclcpp/rclcpp.hpp"
#include <geometry_msgs/msg/point_stamped.hpp>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <cmath>
#include <algorithm>

class TimestampComparatorNode : public rclcpp::Node
{
public:
  TimestampComparatorNode()
      : Node("timestamp_comparator"),
        max_delay_scan_odom_(0.0),
        sum_delay_scan_odom_(0.0),
        count_scan_odom_(0),
        max_delay_scan_opponent_(0.0),
        sum_delay_scan_opponent_(0.0),
        count_scan_opponent_(0),
        max_delay_imu_odom_(0.0),
        sum_delay_imu_odom_(0.0),
        count_imu_odom_(0),
        max_delay_candidate_opponent_odom_(0.0),
        sum_delay_candidate_opponent_odom_(0.0),
        count_candidate_opponent_odom_(0),
        max_delay_scan_candidate_(0.0),
        sum_delay_scan_candidate_(0.0),
        count_scan_candidate_(0)
  {
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan", 10,
        std::bind(&TimestampComparatorNode::scanCallback, this, std::placeholders::_1));

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data", 10,
        std::bind(&TimestampComparatorNode::imuCallback, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10,
        std::bind(&TimestampComparatorNode::odomCallback, this, std::placeholders::_1));

    opponent_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/opponent_odom", 10,
        std::bind(&TimestampComparatorNode::opponentOdomCallback, this, std::placeholders::_1));

    candidate_sub_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
        "/obstacle_candidates", 10,
        std::bind(&TimestampComparatorNode::candidateCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Timestamp Comparator Node Initialized");
  }

private:
  // 딜레이 계산 및 통계 업데이트를 담당하는 함수 (로그 출력 여부 선택 가능)
  void updateDelayStats(const std::string &label,
                        const rclcpp::Time &sensor_time,
                        const rclcpp::Time &reference_time,
                        double &sum_delay, int &count, double &max_delay,
                        bool enable_log = true)
  {
    double delay = std::fabs((reference_time - sensor_time).seconds());
    if (delay < 1.0)
    {
      ++count;
      sum_delay += delay;
      max_delay = std::max(max_delay, delay);
      if (enable_log)
      {
        double avg_delay = sum_delay / count;
        //RCLCPP_INFO(this->get_logger(), "%s: delay = %f sec, max = %f sec, avg = %f sec",
                    // label.c_str(), delay, max_delay, avg_delay);
      }
    }
    else if (enable_log)
    {
      // RCLCPP_WARN(this->get_logger(), "%s: delay (%f sec) >= 1 sec. Skipping measurement.", label.c_str(), delay);
    }
  }

  // /scan 메시지 처리: /odom, /opponent_odom과 비교
  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg)
  {
    rclcpp::Time scan_time = msg->header.stamp;
    // 최신 스캔 타임스탬프를 별도의 변수에 저장
    latest_scan_time_ = scan_time;

    
    //if (latest_odom_time_.nanoseconds() > 0) {
    //   updateDelayStats("/scan vs /odom", scan_time, latest_odom_time_,
    //                    sum_delay_scan_odom_, count_scan_odom_, max_delay_scan_odom_);
    //} else {
    //  RCLCPP_WARN(this->get_logger(), "No /odom message received yet for /scan");
    //}

    if (latest_opponent_odom_time_.nanoseconds() > 0) {
       updateDelayStats("/scan vs /opponent_odom", scan_time, latest_opponent_odom_time_,
                        sum_delay_scan_opponent_, count_scan_opponent_, max_delay_scan_opponent_);
    } else {
       //RCLCPP_WARN(this->get_logger(), "No /opponent_odom message received yet for /scan");
    }
  }

  // /imu/data 메시지 처리: /odom과 비교
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    rclcpp::Time imu_time = msg->header.stamp;
    latest_imu_time_ = imu_time;

    // if (latest_odom_time_.nanoseconds() > 0) {
    //   updateDelayStats("/imu/data vs /odom", imu_time, latest_odom_time_,
    //                    sum_delay_imu_odom_, count_imu_odom_, max_delay_imu_odom_, false);
    // }
  }

  // /obstacle_candidates 메시지 처리: /odom과 비교 및 /scan과의 비교
  void candidateCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    latest_candidate_time_ = msg->header.stamp;
  
    if (latest_opponent_odom_time_.nanoseconds() > 0)
    {
      updateDelayStats("/obstacle_candidates vs /opponent_odom", latest_candidate_time_, latest_opponent_odom_time_,
                         sum_delay_candidate_opponent_odom_, count_candidate_opponent_odom_, max_delay_candidate_opponent_odom_);
      // /scan vs /obstacle_candidates 비교: 최신 스캔 타임과 후보 타임을 비교
      if (latest_scan_time_.nanoseconds() > 0)
      {
        updateDelayStats("/scan vs /obstacle_candidates", latest_scan_time_, latest_candidate_time_,
                           sum_delay_scan_candidate_, count_scan_candidate_, max_delay_scan_candidate_);
      }
      else
      {
        //RCLCPP_WARN(this->get_logger(), "No /scan message received yet for comparison with /obstacle_candidates");
      }
    }
    else
    {
      //RCLCPP_WARN(this->get_logger(), "No /opponent_odom message received yet for /obstacle_candidates");
    }
  }
  

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    latest_odom_time_ = msg->header.stamp;
  }

  void opponentOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    latest_opponent_odom_time_ = msg->header.stamp;
  }

  // 구독자 변수
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr opponent_odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr candidate_sub_;

  // 최신 타임스탬프 저장 변수
  rclcpp::Time latest_scan_time_;
  rclcpp::Time latest_odom_time_;
  rclcpp::Time latest_opponent_odom_time_;
  rclcpp::Time latest_imu_time_;
  rclcpp::Time latest_candidate_time_;

  // /scan vs /odom 통계 변수
  double max_delay_scan_odom_;
  double sum_delay_scan_odom_;
  int count_scan_odom_;

  // /scan vs /opponent_odom 통계 변수
  double max_delay_scan_opponent_;
  double sum_delay_scan_opponent_;
  int count_scan_opponent_;

  // /imu/data vs /odom 통계 변수
  double max_delay_imu_odom_;
  double sum_delay_imu_odom_;
  int count_imu_odom_;

  // /obstacle_candidates vs /opponent_odom 통계 변수 (추가)
  double max_delay_candidate_opponent_odom_;
  double sum_delay_candidate_opponent_odom_;
  int count_candidate_opponent_odom_;

  // /scan vs /obstacle_candidates 통계 변수 (추가)
  double max_delay_scan_candidate_;
  double sum_delay_scan_candidate_;
  int count_scan_candidate_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TimestampComparatorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
