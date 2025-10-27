#ifndef ICP_BASE_IMPL_HPP
#define ICP_BASE_IMPL_HPP

#include "icp_base.hpp"
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <cmath>

namespace icp_comparator {

inline IcpRefinerBase::CloudPtr IcpRefinerBase::downsample(
    const CloudConstPtr& input) const 
{
    if (!input || !params_.use_downsample || params_.voxel_leaf_size <= 0.0f) {
        return CloudPtr(new Cloud(*input));
    }

    pcl::VoxelGrid<PointT> voxel_grid;
    voxel_grid.setLeafSize(params_.voxel_leaf_size, 
                           params_.voxel_leaf_size, 
                           params_.voxel_leaf_size);
    voxel_grid.setInputCloud(input);

    CloudPtr output(new Cloud);
    voxel_grid.filter(*output);
    return output;
}

inline IcpRefinerBase::CloudPtr IcpRefinerBase::rejectFarPoints(
    const CloudConstPtr& input, float radius) const 
{
    if (!input || !params_.reject_far_points) {
        return CloudPtr(new Cloud(*input));
    }

    CloudPtr output(new Cloud);
    output->reserve(input->size());

    const float radius_sq = radius * radius;
    for (const auto& point : input->points) {
        const float dist_sq = point.x * point.x + 
                             point.y * point.y + 
                             point.z * point.z;
        if (dist_sq <= radius_sq) {
            output->push_back(point);
        }
    }

    return output;
}

inline IcpRefinerBase::CloudPtr IcpRefinerBase::sanitizeCloud(
    const CloudConstPtr& input) const 
{
    if (!input) {
        return CloudPtr(new Cloud);
    }

    CloudPtr output(new Cloud);
    output->reserve(input->size());

    for (const auto& point : input->points) {
        if (std::isfinite(point.x) && 
            std::isfinite(point.y) && 
            std::isfinite(point.z)) {
            output->push_back(point);
        }
    }

    return output;
}

inline IcpRefinerBase::CloudPtr IcpRefinerBase::preprocessCloud(
    const CloudConstPtr& input) const 
{
    CloudPtr cloud = sanitizeCloud(input);
    cloud = rejectFarPoints(cloud, params_.reject_radius);
    cloud = downsample(cloud);
    return cloud;
}

} // namespace icp_comparator

#endif // ICP_BASE_IMPL_HPP