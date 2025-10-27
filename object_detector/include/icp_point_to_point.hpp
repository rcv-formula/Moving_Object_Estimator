#ifndef ICP_POINT_TO_POINT_HPP
#define ICP_POINT_TO_POINT_HPP

#include "icp_base.hpp"
#include "icp_base_impl.hpp"
#include <pcl/registration/icp.h>

namespace icp_comparator {

/**
 * @brief Point-to-Point ICP implementation
 * Classic ICP minimizing point-to-point distances
 */
class IcpPointToPoint : public IcpRefinerBase {
public:
    IcpPointToPoint() {
        params_ = IcpParams();
    }

    explicit IcpPointToPoint(const IcpParams& params) {
        params_ = params;
    }

    std::string getMethodName() const override {
        return "Point-to-Point";
    }

    void setParameters(const IcpParams& params) override {
        params_ = params;
    }

    Eigen::Matrix4f refine(
        const CloudConstPtr& source,
        const CloudConstPtr& target,
        const Eigen::Matrix4f& initial_guess,
        double* fitness_score = nullptr) override 
    {
        if (!source || !target || 
            source->size() < static_cast<size_t>(params_.min_points) ||
            target->size() < static_cast<size_t>(params_.min_points)) {
            if (fitness_score) {
                *fitness_score = std::numeric_limits<double>::infinity();
            }
            return initial_guess;
        }

        // Preprocess clouds
        CloudPtr source_processed = preprocessCloud(source);
        CloudPtr target_processed = preprocessCloud(target);

        if (source_processed->size() < static_cast<size_t>(params_.min_points) ||
            target_processed->size() < static_cast<size_t>(params_.min_points)) {
            if (fitness_score) {
                *fitness_score = std::numeric_limits<double>::infinity();
            }
            return initial_guess;
        }

        // Configure ICP
        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setMaxCorrespondenceDistance(params_.max_correspondence_distance);
        icp.setMaximumIterations(params_.max_iterations);
        icp.setTransformationEpsilon(params_.transformation_epsilon);
        icp.setEuclideanFitnessEpsilon(params_.euclidean_fitness_epsilon);
        
        icp.setInputSource(source_processed);
        icp.setInputTarget(target_processed);

        // Perform alignment
        Cloud aligned;
        icp.align(aligned, initial_guess);

        // Get results
        if (fitness_score) {
            *fitness_score = icp.getFitnessScore();
        }

        if (icp.hasConverged()) {
            return icp.getFinalTransformation();
        }

        return initial_guess;
    }
};

} // namespace icp_comparator

#endif // ICP_POINT_TO_POINT_HPP