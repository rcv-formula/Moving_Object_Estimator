#ifndef ICP_GICP_HPP
#define ICP_GICP_HPP

#include "icp_base.hpp"
#include "icp_base_impl.hpp"
#include <pcl/registration/gicp.h>

namespace icp_comparator {

/**
 * @brief Additional parameters for GICP
 */
struct GicpParams : public IcpParams {
    int correspondence_randomness;
    int maximum_optimizer_iterations;
    float rotation_epsilon;
    bool use_reciprocal_correspondences;

    GicpParams() : IcpParams(),
        correspondence_randomness(20),
        maximum_optimizer_iterations(20),
        rotation_epsilon(2e-3f),
        use_reciprocal_correspondences(false) {}
};

/**
 * @brief Generalized ICP implementation
 * Probabilistic ICP using surface covariances
 */
class IcpGeneralized : public IcpRefinerBase {
public:
    IcpGeneralized() {
        gicp_params_ = GicpParams();
        params_ = gicp_params_;
    }

    explicit IcpGeneralized(const GicpParams& params) {
        gicp_params_ = params;
        params_ = params;
    }

    std::string getMethodName() const override {
        return "GICP";
    }

    void setParameters(const IcpParams& params) override {
        params_ = params;
    }

    void setParametersGicp(const GicpParams& params) {
        gicp_params_ = params;
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

        // Configure GICP
        pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
        
        // Basic ICP parameters
        gicp.setMaxCorrespondenceDistance(params_.max_correspondence_distance);
        gicp.setMaximumIterations(params_.max_iterations);
        gicp.setTransformationEpsilon(params_.transformation_epsilon);
        gicp.setEuclideanFitnessEpsilon(params_.euclidean_fitness_epsilon);
        
        // GICP-specific parameters
        gicp.setCorrespondenceRandomness(gicp_params_.correspondence_randomness);
        gicp.setMaximumOptimizerIterations(gicp_params_.maximum_optimizer_iterations);
        gicp.setRotationEpsilon(gicp_params_.rotation_epsilon);
        gicp.setUseReciprocalCorrespondences(gicp_params_.use_reciprocal_correspondences);
        
        gicp.setInputSource(source_processed);
        gicp.setInputTarget(target_processed);

        // Perform alignment
        Cloud aligned;
        gicp.align(aligned, initial_guess);

        // Get results
        if (fitness_score) {
            *fitness_score = gicp.getFitnessScore();
        }

        if (gicp.hasConverged()) {
            return gicp.getFinalTransformation();
        }

        return initial_guess;
    }

private:
    GicpParams gicp_params_;
};

} // namespace icp_comparator

#endif // ICP_GICP_HPP