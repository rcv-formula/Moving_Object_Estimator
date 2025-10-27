#ifndef ICP_POINT_TO_PLANE_HPP
#define ICP_POINT_TO_PLANE_HPP

#include "icp_base.hpp"
#include "icp_base_impl.hpp"
#include <pcl/registration/icp_nl.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>

namespace icp_comparator {

struct Pt2PlParams : public IcpParams {
    bool normal_use_radius;
    float normal_radius;
    int normal_k;
    bool assume_planar;

    Pt2PlParams() : IcpParams(),
        normal_use_radius(false),
        normal_radius(0.3f),
        normal_k(10),
        assume_planar(true) {}
};

class IcpPointToPlane : public IcpRefinerBase {
public:
    using PointNormal = pcl::PointXYZINormal;
    using CloudNormal = pcl::PointCloud<PointNormal>;
    using CloudNormalPtr = typename CloudNormal::Ptr;

    IcpPointToPlane() {
        pt2pl_params_ = Pt2PlParams();
        params_ = pt2pl_params_;
    }

    explicit IcpPointToPlane(const Pt2PlParams& params) {
        pt2pl_params_ = params;
        params_ = params;
    }

    std::string getMethodName() const override {
        return "Point-to-Plane";
    }

    void setParameters(const IcpParams& params) override {
        params_ = params;
    }

    void setParametersPt2Pl(const Pt2PlParams& params) {
        pt2pl_params_ = params;
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

        if (!source_processed || !target_processed ||
            source_processed->size() < static_cast<size_t>(params_.min_points) ||
            target_processed->size() < static_cast<size_t>(params_.min_points)) {
            if (fitness_score) {
                *fitness_score = std::numeric_limits<double>::infinity();
            }
            return initial_guess;
        }

        // Compute normals with strict validation
        CloudNormalPtr source_with_normals = computeNormals(source_processed);
        CloudNormalPtr target_with_normals = computeNormals(target_processed);

        // *** 핵심 수정: 철저한 검증 ***
        if (!source_with_normals || !target_with_normals) {
            if (fitness_score) {
                *fitness_score = std::numeric_limits<double>::infinity();
            }
            return initial_guess;
        }

        // NaN/Inf 제거 (법선 포함)
        source_with_normals = sanitizeCloudWithNormals(source_with_normals);
        target_with_normals = sanitizeCloudWithNormals(target_with_normals);

        if (!source_with_normals || !target_with_normals ||
            source_with_normals->size() < static_cast<size_t>(params_.min_points) ||
            target_with_normals->size() < static_cast<size_t>(params_.min_points)) {
            if (fitness_score) {
                *fitness_score = std::numeric_limits<double>::infinity();
            }
            return initial_guess;
        }

        // Configure ICP with normals
        pcl::IterativeClosestPointWithNormals<PointNormal, PointNormal> icp;
        icp.setMaxCorrespondenceDistance(params_.max_correspondence_distance);
        icp.setMaximumIterations(params_.max_iterations);
        icp.setTransformationEpsilon(params_.transformation_epsilon);
        icp.setEuclideanFitnessEpsilon(params_.euclidean_fitness_epsilon);
        
        icp.setInputSource(source_with_normals);
        icp.setInputTarget(target_with_normals);

        // Perform alignment
        CloudNormal aligned;
        try {
            icp.align(aligned, initial_guess);
        } catch (const std::exception& e) {
            // ICP 실패 시 초기 추정치 반환
            if (fitness_score) {
                *fitness_score = std::numeric_limits<double>::infinity();
            }
            return initial_guess;
        }

        // Get results
        if (fitness_score) {
            *fitness_score = icp.getFitnessScore();
        }

        if (icp.hasConverged()) {
            return icp.getFinalTransformation();
        }

        return initial_guess;
    }

private:
    Pt2PlParams pt2pl_params_;

    // *** 새로운 함수: 법선을 포함한 포인트 검증 ***
    CloudNormalPtr sanitizeCloudWithNormals(const CloudNormalPtr& cloud) const {
        if (!cloud) return CloudNormalPtr(new CloudNormal);

        CloudNormalPtr clean(new CloudNormal);
        clean->reserve(cloud->size());

        for (const auto& point : cloud->points) {
            // 좌표 검증
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                continue;
            }

            // 법선 검증 (중요!)
            if (!std::isfinite(point.normal_x) || 
                !std::isfinite(point.normal_y) || 
                !std::isfinite(point.normal_z)) {
                continue;
            }

            // 법선 크기 검증
            const float normal_norm = std::sqrt(
                point.normal_x * point.normal_x +
                point.normal_y * point.normal_y +
                point.normal_z * point.normal_z
            );

            if (!std::isfinite(normal_norm) || normal_norm < 1e-6f) {
                continue;
            }

            // intensity 검증
            if (!std::isfinite(point.intensity)) {
                PointNormal clean_point = point;
                clean_point.intensity = 0.0f;
                clean->push_back(clean_point);
            } else {
                clean->push_back(point);
            }
        }

        return clean;
    }

    CloudNormalPtr computeNormals(const CloudPtr& cloud) const {
        if (pt2pl_params_.assume_planar) {
            // For 2D laser scans: fix normals to z-axis
            return createFixedNormals(cloud);
        }

        // Estimate normals using neighbors
        return estimateNormals(cloud);
    }

    CloudNormalPtr createFixedNormals(const CloudPtr& cloud) const {
        CloudNormalPtr cloud_with_normals(new CloudNormal);
        if (!cloud) return cloud_with_normals;

        cloud_with_normals->reserve(cloud->size());

        for (const auto& point : cloud->points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || 
                !std::isfinite(point.z)) {
                continue;
            }

            PointNormal pn;
            pn.x = point.x;
            pn.y = point.y;
            pn.z = point.z;
            pn.intensity = std::isfinite(point.intensity) ? point.intensity : 0.0f;
            pn.normal_x = 0.0f;
            pn.normal_y = 0.0f;
            pn.normal_z = 1.0f;  // Fixed normal for 2D
            pn.curvature = 0.0f;

            cloud_with_normals->push_back(pn);
        }

        return cloud_with_normals;
    }

    CloudNormalPtr estimateNormals(const CloudPtr& cloud) const {
        CloudNormalPtr cloud_with_normals(new CloudNormal);
        if (!cloud) return cloud_with_normals;

        CloudPtr clean = sanitizeCloud(cloud);
        if (!clean || clean->size() < static_cast<size_t>(params_.min_points)) {
            return cloud_with_normals;
        }

        // *** 수정: 더 안전한 법선 추정 ***
        pcl::NormalEstimationOMP<PointT, pcl::Normal> normal_estimator;
        normal_estimator.setInputCloud(clean);

        typename pcl::search::KdTree<PointT>::Ptr tree(
            new pcl::search::KdTree<PointT>());
        
        // KdTree 설정 실패 시 early return
        try {
            normal_estimator.setSearchMethod(tree);
        } catch (...) {
            return createFixedNormals(cloud);  // Fallback
        }

        if (pt2pl_params_.normal_use_radius) {
            normal_estimator.setRadiusSearch(pt2pl_params_.normal_radius);
        } else {
            // KNN을 충분히 크게 설정
            normal_estimator.setKSearch(std::max(5, pt2pl_params_.normal_k));
        }

        pcl::PointCloud<pcl::Normal>::Ptr normals(
            new pcl::PointCloud<pcl::Normal>());
        
        try {
            normal_estimator.compute(*normals);
        } catch (...) {
            // 법선 추정 실패 시 고정 법선 사용
            return createFixedNormals(cloud);
        }

        // 법선이 계산되지 않은 경우
        if (!normals || normals->size() != clean->size()) {
            return createFixedNormals(cloud);
        }

        // Combine points and normals with strict validation
        cloud_with_normals->reserve(clean->size());
        for (size_t i = 0; i < clean->size(); ++i) {
            const auto& point = clean->points[i];
            const auto& normal = normals->points[i];

            // 좌표 검증
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || 
                !std::isfinite(point.z)) {
                continue;
            }

            PointNormal pn;
            pn.x = point.x;
            pn.y = point.y;
            pn.z = point.z;
            pn.intensity = std::isfinite(point.intensity) ? point.intensity : 0.0f;

            float nx = normal.normal_x;
            float ny = normal.normal_y;
            float nz = normal.normal_z;

            // *** 핵심: 법선 철저히 검증 ***
            if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz)) {
                // 잘못된 법선 -> z축으로 대체
                nx = 0.0f; ny = 0.0f; nz = 1.0f;
            } else {
                const float norm = std::sqrt(nx*nx + ny*ny + nz*nz);
                if (!std::isfinite(norm) || norm < 1e-6f) {
                    // 0에 가까운 법선 -> z축으로 대체
                    nx = 0.0f; ny = 0.0f; nz = 1.0f;
                } else {
                    // 정규화
                    nx /= norm;
                    ny /= norm;
                    nz /= norm;

                    // 정규화 후 재검증
                    if (!std::isfinite(nx) || !std::isfinite(ny) || !std::isfinite(nz)) {
                        nx = 0.0f; ny = 0.0f; nz = 1.0f;
                    }
                }
            }

            pn.normal_x = nx;
            pn.normal_y = ny;
            pn.normal_z = nz;
            pn.curvature = std::isfinite(normal.curvature) ? normal.curvature : 0.0f;

            cloud_with_normals->push_back(pn);
        }

        // 최종 검증: 충분한 포인트가 남았는지 확인
        if (cloud_with_normals->size() < static_cast<size_t>(params_.min_points)) {
            // 너무 적으면 고정 법선 사용
            return createFixedNormals(cloud);
        }

        return cloud_with_normals;
    }
};

} // namespace icp_comparator

#endif // ICP_POINT_TO_PLANE_HPP