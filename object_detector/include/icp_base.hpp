#ifndef ICP_BASE_HPP
#define ICP_BASE_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <memory>
#include <string>

namespace icp_comparator {

/**
 * @brief Common parameters for all ICP methods
 */
struct IcpParams {
    int max_iterations;
    float max_correspondence_distance;
    double transformation_epsilon;
    double euclidean_fitness_epsilon;
    float voxel_leaf_size;
    bool use_downsample;
    bool reject_far_points;
    float reject_radius;
    int min_points;

    IcpParams()
        : max_iterations(30),
          max_correspondence_distance(0.5f),
          transformation_epsilon(1e-6),
          euclidean_fitness_epsilon(1e-5),
          voxel_leaf_size(0.10f),
          use_downsample(true),
          reject_far_points(true),
          reject_radius(5.0f),
          min_points(20) {}
};

/**
 * @brief Base class for ICP refiners
 */
class IcpRefinerBase {
public:
    using PointT = pcl::PointXYZI;
    using Cloud = pcl::PointCloud<PointT>;
    using CloudPtr = typename Cloud::Ptr;
    using CloudConstPtr = typename Cloud::ConstPtr;

    virtual ~IcpRefinerBase() = default;

    /**
     * @brief Refine alignment between source and target clouds
     * @param source Source point cloud (already roughly aligned)
     * @param target Target point cloud (current frame)
     * @param initial_guess Initial transformation guess
     * @param fitness_score Output fitness score (optional)
     * @return Refined transformation matrix
     */
    virtual Eigen::Matrix4f refine(
        const CloudConstPtr& source,
        const CloudConstPtr& target,
        const Eigen::Matrix4f& initial_guess,
        double* fitness_score = nullptr) = 0;

    /**
     * @brief Get method name
     */
    virtual std::string getMethodName() const = 0;

    /**
     * @brief Set parameters
     */
    virtual void setParameters(const IcpParams& params) = 0;

protected:
    IcpParams params_;

    /**
     * @brief Downsample point cloud using voxel grid
     */
    CloudPtr downsample(const CloudConstPtr& input) const;

    /**
     * @brief Remove points outside radius
     */
    CloudPtr rejectFarPoints(const CloudConstPtr& input, float radius) const;

    /**
     * @brief Remove NaN/Inf points
     */
    CloudPtr sanitizeCloud(const CloudConstPtr& input) const;

    /**
     * @brief Preprocess cloud (downsample + reject + sanitize)
     */
    CloudPtr preprocessCloud(const CloudConstPtr& input) const;
};

} // namespace icp_comparator

#endif // ICP_BASE_HPP