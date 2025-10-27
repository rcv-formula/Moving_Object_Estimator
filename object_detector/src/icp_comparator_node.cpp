#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/string.hpp>

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
#include <string>
#include <chrono>
#include <map>

#include "map_manager_pair.hpp"
#include "icp_point_to_point.hpp"
// #include "icp_point_to_plane.hpp"  // 비활성화
#include "icp_point_to_line.hpp"
#include "icp_gicp.hpp"

namespace icp_comparator {

// Dummy Point-to-Plane
class IcpPointToPlaneDummy : public IcpRefinerBase {
public:
    IcpPointToPlaneDummy() {}
    
    std::string getMethodName() const override {
        return "Point-to-Plane (Disabled)";
    }
    
    void setParameters(const IcpParams& params) override {}
    
    Eigen::Matrix4f refine(
        const CloudConstPtr& source,
        const CloudConstPtr& target,
        const Eigen::Matrix4f& initial_guess,
        double* fitness_score = nullptr) override 
    {
        if (fitness_score) {
            *fitness_score = std::numeric_limits<double>::infinity();
        }
        return initial_guess;
    }
};

class IcpComparatorNode : public rclcpp::Node {
public:
    using PointT = pcl::PointXYZI;
    using Cloud = pcl::PointCloud<PointT>;
    using CloudPtr = typename Cloud::Ptr;
    using CloudConstPtr = typename Cloud::ConstPtr;

    IcpComparatorNode() 
        : Node("icp_comparator_node"),
          map_manager_(10),
          frame_counter_(0),
          frame_processed_(0),
          frame_skipped_(0)
    {
        initializeParameters();
        initializeRefiners();
        initializePublishers();
        initializeSubscribers();
        initializeTimer();
        
        RCLCPP_INFO(this->get_logger(), 
            "ICP Comparator Node started");
        RCLCPP_INFO(this->get_logger(), 
            "Active ICP method: %s", active_method_.c_str());
        RCLCPP_INFO(this->get_logger(),
            "Performance mode: Only selected method will run");
        print_parameters();
    }

private:
    void initializeParameters() {
        this->declare_parameter<std::string>("icp_method", "point_to_point");
        this->declare_parameter<int>("max_history_frames", 10);
        this->declare_parameter<bool>("throttle_processing", true);
        this->declare_parameter<int>("process_every_n_frames", 2);
        this->declare_parameter<double>("min_delta_trans", 0.2);
        this->declare_parameter<double>("min_delta_rot", 0.1);
        this->declare_parameter<bool>("enable_icp", true);
        this->declare_parameter<int>("max_history_for_icp", 3);

        this->declare_parameter<int>("icp.max_iterations", 30);
        this->declare_parameter<double>("icp.max_corr_dist", 0.5);
        this->declare_parameter<double>("icp.trans_eps", 1e-6);
        this->declare_parameter<double>("icp.fit_eps", 1e-5);
        this->declare_parameter<double>("icp.voxel_leaf", 0.10);
        this->declare_parameter<bool>("icp.use_downsample", true);
        this->declare_parameter<bool>("icp.reject_far", true);
        this->declare_parameter<double>("icp.reject_radius", 5.0);
        this->declare_parameter<int>("icp.min_points", 20);

        this->declare_parameter<double>("pt2line.line_fitting_dist", 0.05);
        this->declare_parameter<int>("pt2line.min_line_points", 5);
        this->declare_parameter<double>("pt2line.outlier_threshold", 0.1);

        this->declare_parameter<int>("gicp.correspondence_randomness", 20);
        this->declare_parameter<int>("gicp.max_optimizer_iterations", 20);
        this->declare_parameter<double>("gicp.rotation_epsilon", 2e-3);
        this->declare_parameter<bool>("gicp.use_reciprocal", false);

        updateParameters();
    }

    void updateParameters() {
        active_method_ = this->get_parameter("icp_method").as_string();
        max_history_frames_ = this->get_parameter("max_history_frames").as_int();
        throttle_processing_ = this->get_parameter("throttle_processing").as_bool();
        process_every_n_frames_ = this->get_parameter("process_every_n_frames").as_int();
        min_delta_trans_ = this->get_parameter("min_delta_trans").as_double();
        min_delta_rot_ = this->get_parameter("min_delta_rot").as_double();
        enable_icp_ = this->get_parameter("enable_icp").as_bool();
        max_history_for_icp_ = this->get_parameter("max_history_for_icp").as_int();

        map_manager_.SetMaxDequeSize(max_history_frames_);
    }

    void print_parameters(){
        RCLCPP_INFO(this->get_logger(), "=== Parameter Summary ===");
        RCLCPP_INFO(this->get_logger(), "icp_method: %s", active_method_.c_str());
        RCLCPP_INFO(this->get_logger(), "max_history_frames: %d", max_history_frames_);
        RCLCPP_INFO(this->get_logger(), "throttle_processing: %s", throttle_processing_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "process_every_n_frames: %d", process_every_n_frames_);
        RCLCPP_INFO(this->get_logger(), "min_delta_trans: %.3f", min_delta_trans_);
        RCLCPP_INFO(this->get_logger(), "min_delta_rot: %.3f", min_delta_rot_);
        RCLCPP_INFO(this->get_logger(), "enable_icp: %s", enable_icp_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "max_history_for_icp: %d", max_history_for_icp_);

        RCLCPP_INFO(this->get_logger(), "icp.max_iterations: %ld", this->get_parameter("icp.max_iterations").as_int());
        RCLCPP_INFO(this->get_logger(), "icp.max_corr_dist: %.3f", this->get_parameter("icp.max_corr_dist").as_double());
        RCLCPP_INFO(this->get_logger(), "icp.trans_eps: %.6f", this->get_parameter("icp.trans_eps").as_double());
        RCLCPP_INFO(this->get_logger(), "icp.fit_eps: %.6f", this->get_parameter("icp.fit_eps").as_double());
        RCLCPP_INFO(this->get_logger(), "icp.voxel_leaf: %.3f", this->get_parameter("icp.voxel_leaf").as_double());
        RCLCPP_INFO(this->get_logger(), "icp.use_downsample: %s", this->get_parameter("icp.use_downsample").as_bool() ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "icp.reject_far: %s", this->get_parameter("icp.reject_far").as_bool() ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "icp.reject_radius: %.3f", this->get_parameter("icp.reject_radius").as_double());
        RCLCPP_INFO(this->get_logger(), "icp.min_points: %ld", this->get_parameter("icp.min_points").as_int());

        RCLCPP_INFO(this->get_logger(), "pt2line.line_fitting_dist: %.3f", this->get_parameter("pt2line.line_fitting_dist").as_double());
        RCLCPP_INFO(this->get_logger(), "pt2line.min_line_points: %ld", this->get_parameter("pt2line.min_line_points").as_int());
        RCLCPP_INFO(this->get_logger(), "pt2line.outlier_threshold: %.3f", this->get_parameter("pt2line.outlier_threshold").as_double());

        RCLCPP_INFO(this->get_logger(), "gicp.correspondence_randomness: %ld", this->get_parameter("gicp.correspondence_randomness").as_int());
        RCLCPP_INFO(this->get_logger(), "gicp.max_optimizer_iterations: %ld", this->get_parameter("gicp.max_optimizer_iterations").as_int());
        RCLCPP_INFO(this->get_logger(), "gicp.rotation_epsilon: %.6f", this->get_parameter("gicp.rotation_epsilon").as_double());
        RCLCPP_INFO(this->get_logger(), "gicp.use_reciprocal: %s", this->get_parameter("gicp.use_reciprocal").as_bool() ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "=============================");
    }

    void initializeRefiners() {
        IcpParams common_params;
        common_params.max_iterations = this->get_parameter("icp.max_iterations").as_int();
        common_params.max_correspondence_distance = 
            static_cast<float>(this->get_parameter("icp.max_corr_dist").as_double());
        common_params.transformation_epsilon = 
            this->get_parameter("icp.trans_eps").as_double();
        common_params.euclidean_fitness_epsilon = 
            this->get_parameter("icp.fit_eps").as_double();
        common_params.voxel_leaf_size = 
            static_cast<float>(this->get_parameter("icp.voxel_leaf").as_double());
        common_params.use_downsample = this->get_parameter("icp.use_downsample").as_bool();
        common_params.reject_far_points = this->get_parameter("icp.reject_far").as_bool();
        common_params.reject_radius = 
            static_cast<float>(this->get_parameter("icp.reject_radius").as_double());
        common_params.min_points = this->get_parameter("icp.min_points").as_int();

        refiner_pt2pt_ = std::make_shared<IcpPointToPoint>(common_params);
        refiner_pt2pl_ = std::make_shared<IcpPointToPlaneDummy>();

        Pt2LineParams pt2line_params;
        static_cast<IcpParams&>(pt2line_params) = common_params;
        pt2line_params.line_fitting_distance = 
            static_cast<float>(this->get_parameter("pt2line.line_fitting_dist").as_double());
        pt2line_params.min_line_points = 
            this->get_parameter("pt2line.min_line_points").as_int();
        pt2line_params.outlier_threshold = 
            static_cast<float>(this->get_parameter("pt2line.outlier_threshold").as_double());
        refiner_pt2line_ = std::make_shared<IcpPointToLine>(pt2line_params);

        GicpParams gicp_params;
        static_cast<IcpParams&>(gicp_params) = common_params;
        gicp_params.correspondence_randomness = 
            this->get_parameter("gicp.correspondence_randomness").as_int();
        gicp_params.maximum_optimizer_iterations = 
            this->get_parameter("gicp.max_optimizer_iterations").as_int();
        gicp_params.rotation_epsilon = 
            static_cast<float>(this->get_parameter("gicp.rotation_epsilon").as_double());
        gicp_params.use_reciprocal_correspondences = 
            this->get_parameter("gicp.use_reciprocal").as_bool();
        refiner_gicp_ = std::make_shared<IcpGeneralized>(gicp_params);

        refiners_["point_to_point"] = refiner_pt2pt_;
        refiners_["point_to_plane"] = refiner_pt2pl_;
        refiners_["point_to_line"] = refiner_pt2line_;
        refiners_["gicp"] = refiner_gicp_;
    }

    void initializePublishers() {
        pub_current_scan_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/current_scan", 10);
        pub_aligned_odom_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/aligned_history_odom", 10);
        
        // *** 선택된 ICP 결과만 퍼블리시 ***
        pub_aligned_icp_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/aligned_history_icp", 10);
        
        pub_stats_ = this->create_publisher<std_msgs::msg::String>(
            "/icp_statistics", 10);
    }

    void initializeSubscribers() {
        using SyncPolicy = message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::LaserScan, nav_msgs::msg::Odometry>;

        scan_sub_.subscribe(this, "/scan", rmw_qos_profile_sensor_data);
        odom_sub_.subscribe(this, "/odom", rmw_qos_profile_sensor_data);
        
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(1000), scan_sub_, odom_sub_);
        sync_->registerCallback(
            std::bind(&IcpComparatorNode::syncCallback, this, 
                     std::placeholders::_1, std::placeholders::_2));
    }

    void initializeTimer() {
        timer_stats_ = this->create_wall_timer(
            std::chrono::seconds(3),
            std::bind(&IcpComparatorNode::publishStatistics, this));
    }

    CloudPtr buildLocalCloud(
        const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg) 
    {
        CloudPtr cloud(new Cloud);
        cloud->reserve(scan_msg->ranges.size());

        for (size_t i = 0; i < scan_msg->ranges.size(); ++i) {
            const float range = scan_msg->ranges[i];
            if (!std::isfinite(range) || range <= 0.0f || range > 30.0f) {
                continue;
            }

            const float angle = scan_msg->angle_min + 
                               i * scan_msg->angle_increment;
            
            if (!std::isfinite(angle)) {
                continue;
            }
            
            PointT point;
            point.x = range * std::cos(angle);
            point.y = range * std::sin(angle);
            point.z = 0.0f;
            point.intensity = 255.0f;
            
            if (std::isfinite(point.x) && std::isfinite(point.y)) {
                cloud->push_back(point);
            }
        }

        return cloud;
    }

    Eigen::Matrix4d poseToMatrix(const geometry_msgs::msg::Pose& pose) {
        Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x,
                            pose.orientation.y, pose.orientation.z);
        const double norm_sq = q.squaredNorm();
        
        if (!std::isfinite(norm_sq) || norm_sq < 1e-12) {
            q = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
        } else {
            q.normalize();
        }

        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        transform.topLeftCorner<3, 3>() = q.toRotationMatrix();
        transform(0, 3) = std::isfinite(pose.position.x) ? pose.position.x : 0.0;
        transform(1, 3) = std::isfinite(pose.position.y) ? pose.position.y : 0.0;
        transform(2, 3) = std::isfinite(pose.position.z) ? pose.position.z : 0.0;

        return transform;
    }

    double computeMotionDelta(const Eigen::Matrix4d& transform1,
                             const Eigen::Matrix4d& transform2,
                             double* rotation_delta = nullptr) 
    {
        Eigen::Vector3d translation1 = transform1.block<3, 1>(0, 3);
        Eigen::Vector3d translation2 = transform2.block<3, 1>(0, 3);
        double translation_delta = (translation2 - translation1).norm();

        if (rotation_delta) {
            Eigen::Matrix3d rotation1 = transform1.block<3, 3>(0, 0);
            Eigen::Matrix3d rotation2 = transform2.block<3, 3>(0, 0);
            Eigen::Matrix3d relative_rotation = rotation1.transpose() * rotation2;
            Eigen::AngleAxisd angle_axis(relative_rotation);
            *rotation_delta = std::abs(angle_axis.angle());
        }

        return translation_delta;
    }

    void syncCallback(
        const sensor_msgs::msg::LaserScan::ConstSharedPtr& scan_msg,
        const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        frame_counter_++;

        if (throttle_processing_ && 
            (frame_counter_ % process_every_n_frames_ != 0)) {
            frame_skipped_++;
            return;
        }

        CloudPtr current_cloud = buildLocalCloud(scan_msg);
        if (current_cloud->empty() || current_cloud->size() < 10) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 
                5000, "Too few points in current scan: %zu", current_cloud->size());
            return;
        }

        sensor_msgs::msg::PointCloud2 current_msg;
        pcl::toROSMsg(*current_cloud, current_msg);
        current_msg.header = scan_msg->header;
        pub_current_scan_->publish(current_msg);

        map_manager_.AddCloudWithPose(
            std::make_shared<sensor_msgs::msg::PointCloud2>(current_msg),
            odom_msg->pose.pose);

        const Eigen::Matrix4d current_transform = 
            poseToMatrix(odom_msg->pose.pose);

        bool should_process = false;
        if (!last_processed_pose_set_) {
            should_process = true;
            last_processed_pose_set_ = true;
        } else {
            double rotation_delta;
            double translation_delta = computeMotionDelta(
                last_processed_pose_, current_transform, &rotation_delta);
            
            if (translation_delta >= min_delta_trans_ || 
                rotation_delta >= min_delta_rot_) {
                should_process = true;
            }
        }

        if (!should_process) {
            return;
        }

        last_processed_pose_ = current_transform;
        frame_processed_++;

        processHistory(scan_msg->header, current_cloud, current_transform);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);
        processing_times_[active_method_].push_back(duration.count());
        
        if (processing_times_[active_method_].size() > 100) {
            processing_times_[active_method_].erase(
                processing_times_[active_method_].begin());
        }
    }

    void processHistory(
        const std_msgs::msg::Header& header,
        const CloudPtr& current_cloud,
        const Eigen::Matrix4d& current_transform)
    {
        auto pairs = map_manager_.GetCloudPoseSnapshot();
        if (pairs.size() < 2) {
            return;
        }

        const Eigen::Matrix4d current_transform_inv = current_transform.inverse();
        const size_t history_size = pairs.size() - 1;
        const size_t icp_start_idx = (history_size > static_cast<size_t>(max_history_for_icp_))
            ? (history_size - max_history_for_icp_) : 0;

        Cloud cloud_odom, cloud_icp;
        double fitness_sum = 0.0;
        int fitness_count = 0;

        for (size_t i = 0; i < history_size; ++i) {
            //for test
            // if(i != 0) continue;

            const auto& history_cloud_ptr = pairs[i].first;
            const auto& history_pose = pairs[i].second;

            if (!history_cloud_ptr || history_cloud_ptr->empty()) {
                continue;
            }

            const Eigen::Matrix4d history_transform = poseToMatrix(history_pose);
            const Eigen::Matrix4d relative_transform = 
                current_transform_inv * history_transform;

            Cloud history_aligned_odom;
            pcl::transformPointCloud(*history_cloud_ptr, history_aligned_odom,
                                    relative_transform.cast<float>());

            if (history_aligned_odom.size() < 20) {
                continue;
            }

            const float intensity = static_cast<float>((history_size - 1) - i);
            for (auto& point : history_aligned_odom.points) {
                point.intensity = intensity;
            }

            cloud_odom += history_aligned_odom;

            const bool apply_icp = enable_icp_ && (i >= icp_start_idx);

            if (apply_icp) {
                Eigen::Matrix4f init = relative_transform.cast<float>();
                processWithSelectedICP(history_aligned_odom, current_cloud, intensity,
                                      init, cloud_icp, fitness_sum, fitness_count);
            } else {
                RCLCPP_WARN(this->get_logger(), "ICP not available!!");
                cloud_icp += history_aligned_odom;
            }
        }

        publishClouds(header, cloud_odom, cloud_icp);

        if (fitness_count > 0) {
            fitness_scores_[active_method_] = fitness_sum / fitness_count;
        }
    }
    
    void processWithSelectedICP(
        const Cloud& history_odom,
        const CloudPtr& current_cloud,
        float intensity,
        const Eigen::Matrix4f& init_guess,
        Cloud& out_icp,
        double& fitness_sum,
        int& fitness_count)
    {
        CloudConstPtr history_ptr(new Cloud(history_odom));
        
        // 선택된 refiner 가져오기
        auto refiner_it = refiners_.find(active_method_);
        if (refiner_it == refiners_.end()) {
            RCLCPP_ERROR_ONCE(this->get_logger(),
                "Invalid ICP method: %s. Using odom only.", active_method_.c_str());
            out_icp += history_odom;
            return;
        }
        
        auto& refiner = refiner_it->second;
        
        // ICP 실행 (선택된 방법만!)
        
        double fitness = std::numeric_limits<double>::infinity();
        Eigen::Matrix4f transform = refiner->refine(
            history_ptr, current_cloud, Eigen::Matrix4f::Identity(), &fitness);
        
        // 변환 적용
        Cloud cloud_icp;
        pcl::transformPointCloud(history_odom, cloud_icp, transform);
        setIntensity(cloud_icp, intensity);
        out_icp += cloud_icp;
        
        // Fitness 기록
        if (std::isfinite(fitness)) {
            fitness_sum += fitness;
            fitness_count++;
        }
    }

    void setIntensity(Cloud& cloud, float intensity) {
        for (auto& point : cloud.points) {
            point.intensity = intensity;
        }
    }

    void publishClouds(
        const std_msgs::msg::Header& header,
        const Cloud& cloud_odom,
        const Cloud& cloud_icp)
    {
        sensor_msgs::msg::PointCloud2 msg;

        // 오도메트리 정렬
        pcl::toROSMsg(cloud_odom, msg);
        msg.header = header;
        pub_aligned_odom_->publish(msg);

        // 선택된 ICP 결과
        pcl::toROSMsg(cloud_icp, msg);
        msg.header = header;
        pub_aligned_icp_->publish(msg);
    }

    void publishStatistics() {
        if (processing_times_.empty()) {
            return;
        }

        std::stringstream ss;
        ss << "\n=== ICP Performance Statistics ===\n";
        ss << "Input: " << (frame_processed_ + frame_skipped_) / 3.0 << " Hz\n";
        ss << "Processing: " << frame_processed_ / 3.0 << " Hz\n";
        ss << "Skipped: " << frame_skipped_ << " frames\n\n";

        ss << "Active method: " << active_method_ << "\n";
        
        if (processing_times_.find(active_method_) != processing_times_.end() &&
            !processing_times_[active_method_].empty()) {
            
            const auto& times = processing_times_[active_method_];
            double sum = 0.0;
            double max_time = 0.0;
            for (const auto& t : times) {
                sum += t;
                max_time = std::max(max_time, static_cast<double>(t));
            }
            double avg = sum / times.size();

            ss << "  Avg: " << avg << " ms\n";
            ss << "  Max: " << max_time << " ms\n";
            
            if (fitness_scores_.find(active_method_) != fitness_scores_.end()) {
                ss << "  Fitness: " << fitness_scores_[active_method_] << "\n";
            }
        }

        std_msgs::msg::String msg;
        msg.data = ss.str();
        pub_stats_->publish(msg);

        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());

        frame_processed_ = 0;
        frame_skipped_ = 0;
    }

    MapManager map_manager_;
    
    message_filters::Subscriber<sensor_msgs::msg::LaserScan> scan_sub_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
    std::shared_ptr<message_filters::Synchronizer<
        message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::LaserScan, nav_msgs::msg::Odometry>>> sync_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_current_scan_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_aligned_odom_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_aligned_icp_;  // 단일 토픽!
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_stats_;
    rclcpp::TimerBase::SharedPtr timer_stats_;

    std::shared_ptr<IcpPointToPoint> refiner_pt2pt_;
    std::shared_ptr<IcpRefinerBase> refiner_pt2pl_;
    std::shared_ptr<IcpPointToLine> refiner_pt2line_;
    std::shared_ptr<IcpGeneralized> refiner_gicp_;
    std::map<std::string, std::shared_ptr<IcpRefinerBase>> refiners_;

    std::string active_method_;
    int max_history_frames_;
    bool throttle_processing_;
    int process_every_n_frames_;
    double min_delta_trans_;
    double min_delta_rot_;
    bool enable_icp_;
    int max_history_for_icp_;

    Eigen::Matrix4d last_processed_pose_;
    bool last_processed_pose_set_{false};
    size_t frame_counter_;
    size_t frame_processed_;
    size_t frame_skipped_;

    std::map<std::string, std::vector<int64_t>> processing_times_;
    std::map<std::string, double> fitness_scores_;
};

} // namespace icp_comparator

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<icp_comparator::IcpComparatorNode>());
    rclcpp::shutdown();
    return 0;
}