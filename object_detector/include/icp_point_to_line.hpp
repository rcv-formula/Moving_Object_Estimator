#ifndef ICP_POINT_TO_LINE_HPP
#define ICP_POINT_TO_LINE_HPP

#include "icp_base.hpp"
#include "icp_base_impl.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <Eigen/Eigenvalues>

namespace icp_comparator {

struct Pt2LineParams : public IcpParams {
    float line_fitting_distance;
    int min_line_points;
    float outlier_threshold;

    Pt2LineParams() : IcpParams(),
        line_fitting_distance(0.05f),
        min_line_points(5),
        outlier_threshold(0.1f) {}
};

class IcpPointToLine : public IcpRefinerBase {
public:
    struct Line2D {
        Eigen::Vector2f point;
        Eigen::Vector2f direction;
        Eigen::Vector2f normal;
        std::vector<int> indices;
    };

    struct Correspondence {
        Eigen::Vector2f source_point;
        Eigen::Vector2f target_projection;
        Eigen::Vector2f line_normal;
        float error;
    };

    IcpPointToLine() {
        pt2line_params_ = Pt2LineParams();
        params_ = pt2line_params_;
    }

    explicit IcpPointToLine(const Pt2LineParams& params) {
        pt2line_params_ = params;
        params_ = params;
    }

    std::string getMethodName() const override {
        return "Point-to-Line";
    }

    void setParameters(const IcpParams& params) override {
        params_ = params;
    }

    void setParametersPt2Line(const Pt2LineParams& params) {
        pt2line_params_ = params;
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

        // Extract lines from target
        std::vector<Line2D> target_lines;
        try {
            target_lines = extractLines(target_processed);
        } catch (const std::exception& e) {
            // 선분 추출 실패
            if (fitness_score) {
                *fitness_score = std::numeric_limits<double>::infinity();
            }
            return initial_guess;
        }

        if (target_lines.empty()) {
            if (fitness_score) {
                *fitness_score = std::numeric_limits<double>::infinity();
            }
            return initial_guess;
        }

        // Iterative refinement
        Eigen::Matrix4f transform = initial_guess;
        CloudPtr source_transformed(new Cloud);

        for (int iter = 0; iter < params_.max_iterations; ++iter) {
            // Transform source
            try {
                pcl::transformPointCloud(*source_processed, *source_transformed, transform);
            } catch (...) {
                break;
            }

            // Find correspondences
            std::vector<Correspondence> correspondences;
            findCorrespondences(source_transformed, target_lines, correspondences);

            if (correspondences.size() < static_cast<size_t>(params_.min_points)) {
                break;
            }

            // Compute transformation update
            Eigen::Matrix4f delta = computeTransformation(correspondences);

            // 변환 행렬 검증
            if (!isValidTransform(delta)) {
                break;
            }

            // Check convergence
            Eigen::Vector2f translation_delta(delta(0, 3), delta(1, 3));
            if (translation_delta.norm() < params_.transformation_epsilon) {
                transform = delta * transform;
                break;
            }

            // Update transformation
            transform = delta * transform;
        }

        // Compute final fitness score
        if (fitness_score) {
            *fitness_score = computeFitness(source_processed, target_lines, transform);
        }

        return transform;
    }

private:
    Pt2LineParams pt2line_params_;

    // *** 새로운 함수: 변환 행렬 유효성 검사 ***
    bool isValidTransform(const Eigen::Matrix4f& T) const {
        // NaN/Inf 체크
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (!std::isfinite(T(i, j))) {
                    return false;
                }
            }
        }

        // 회전 행렬 검증 (determinant should be close to 1)
        float det = T.block<2, 2>(0, 0).determinant();
        if (!std::isfinite(det) || std::abs(det - 1.0f) > 0.1f) {
            return false;
        }

        return true;
    }

    std::vector<Line2D> extractLines(const CloudPtr& cloud) const {
        std::vector<Line2D> lines;
        std::vector<bool> used(cloud->size(), false);

        pcl::KdTreeFLANN<PointT> kdtree;
        
        // KdTree 설정 시 예외 처리
        try {
            kdtree.setInputCloud(cloud);
        } catch (const std::exception& e) {
            return lines;  // 빈 벡터 반환
        }

        for (size_t i = 0; i < cloud->size(); ++i) {
            if (used[i]) continue;

            // 포인트 유효성 검사
            const auto& pt = cloud->points[i];
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
                used[i] = true;
                continue;
            }

            std::vector<int> indices;
            std::vector<float> distances;
            
            try {
                kdtree.radiusSearch(cloud->points[i], 
                                   pt2line_params_.line_fitting_distance,
                                   indices, distances);
            } catch (const std::exception& e) {
                used[i] = true;
                continue;
            }

            if (indices.size() < static_cast<size_t>(pt2line_params_.min_line_points)) {
                continue;
            }

            // Compute mean
            Eigen::Vector2f mean(0.0f, 0.0f);
            int valid_count = 0;
            for (int idx : indices) {
                const auto& p = cloud->points[idx];
                if (std::isfinite(p.x) && std::isfinite(p.y)) {
                    mean.x() += p.x;
                    mean.y() += p.y;
                    valid_count++;
                }
            }

            if (valid_count < pt2line_params_.min_line_points) {
                continue;
            }

            mean /= static_cast<float>(valid_count);

            // Compute covariance matrix
            Eigen::Matrix2f covariance = Eigen::Matrix2f::Zero();
            for (int idx : indices) {
                const auto& p = cloud->points[idx];
                if (!std::isfinite(p.x) || !std::isfinite(p.y)) continue;

                Eigen::Vector2f diff(p.x - mean.x(), p.y - mean.y());
                covariance += diff * diff.transpose();
            }
            covariance /= static_cast<float>(valid_count);

            // Eigenvalue decomposition
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> solver(covariance);
            
            if (solver.info() != Eigen::Success) {
                continue;
            }

            Eigen::Vector2f direction = solver.eigenvectors().col(1);
            
            // 방향 벡터 검증
            if (!std::isfinite(direction.x()) || !std::isfinite(direction.y())) {
                continue;
            }

            const float dir_norm = direction.norm();
            if (!std::isfinite(dir_norm) || dir_norm < 1e-6f) {
                continue;
            }

            direction.normalize();

            // 재검증
            if (!std::isfinite(direction.x()) || !std::isfinite(direction.y())) {
                continue;
            }

            // Perpendicular normal
            Eigen::Vector2f normal(-direction.y(), direction.x());

            if (!std::isfinite(normal.x()) || !std::isfinite(normal.y())) {
                continue;
            }

            // Create line
            Line2D line;
            line.point = mean;
            line.direction = direction;
            line.normal = normal;
            line.indices = indices;
            lines.push_back(line);

            // Mark points as used
            for (int idx : indices) {
                used[idx] = true;
            }
        }

        return lines;
    }

    void findCorrespondences(
        const CloudPtr& source,
        const std::vector<Line2D>& target_lines,
        std::vector<Correspondence>& correspondences) const 
    {
        correspondences.clear();
        correspondences.reserve(source->size());

        for (const auto& point : source->points) {
            // 포인트 유효성 검사
            if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
                continue;
            }

            Eigen::Vector2f src_pt(point.x, point.y);

            // Find closest line
            float min_distance = std::numeric_limits<float>::max();
            int best_line_idx = -1;
            Eigen::Vector2f best_projection;

            for (size_t j = 0; j < target_lines.size(); ++j) {
                const auto& line = target_lines[j];
                Eigen::Vector2f projection = projectPointToLine(src_pt, line);
                
                // 투영 결과 검증
                if (!std::isfinite(projection.x()) || !std::isfinite(projection.y())) {
                    continue;
                }

                float distance = (src_pt - projection).norm();

                if (!std::isfinite(distance)) {
                    continue;
                }

                if (distance < min_distance && 
                    distance < params_.max_correspondence_distance) {
                    min_distance = distance;
                    best_line_idx = static_cast<int>(j);
                    best_projection = projection;
                }
            }

            if (best_line_idx >= 0) {
                Correspondence corr;
                corr.source_point = src_pt;
                corr.target_projection = best_projection;
                corr.line_normal = target_lines[best_line_idx].normal;
                corr.error = min_distance;
                correspondences.push_back(corr);
            }
        }
    }

    Eigen::Vector2f projectPointToLine(
        const Eigen::Vector2f& point,
        const Line2D& line) const 
    {
        Eigen::Vector2f diff = point - line.point;
        float projection_length = diff.dot(line.direction);
        return line.point + projection_length * line.direction;
    }

    Eigen::Matrix4f computeTransformation(
        const std::vector<Correspondence>& correspondences) const 
    {
        if (correspondences.empty()) {
            return Eigen::Matrix4f::Identity();
        }

        // Compute centroids
        Eigen::Vector2f source_centroid(0.0f, 0.0f);
        Eigen::Vector2f target_centroid(0.0f, 0.0f);

        for (const auto& corr : correspondences) {
            source_centroid += corr.source_point;
            target_centroid += corr.target_projection;
        }

        const float n = static_cast<float>(correspondences.size());
        source_centroid /= n;
        target_centroid /= n;

        // Compute cross-covariance matrix
        Eigen::Matrix2f H = Eigen::Matrix2f::Zero();
        for (const auto& corr : correspondences) {
            Eigen::Vector2f p = corr.source_point - source_centroid;
            Eigen::Vector2f q = corr.target_projection - target_centroid;
            H += p * q.transpose();
        }

        // SVD to find rotation
        Eigen::JacobiSVD<Eigen::Matrix2f> svd(
            H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        
        if (svd.info() != Eigen::Success) {
            return Eigen::Matrix4f::Identity();
        }

        Eigen::Matrix2f rotation = svd.matrixV() * svd.matrixU().transpose();

        // Ensure proper rotation (det = 1)
        if (rotation.determinant() < 0.0f) {
            Eigen::Matrix2f V = svd.matrixV();
            V.col(1) *= -1.0f;
            rotation = V * svd.matrixU().transpose();
        }

        // Compute translation
        Eigen::Vector2f translation = target_centroid - rotation * source_centroid;

        // Build 4x4 transformation matrix
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        transform.block<2, 2>(0, 0) = rotation;
        transform(0, 3) = translation.x();
        transform(1, 3) = translation.y();

        return transform;
    }

    double computeFitness(
        const CloudPtr& source,
        const std::vector<Line2D>& lines,
        const Eigen::Matrix4f& transform) const 
    {
        CloudPtr transformed(new Cloud);
        
        try {
            pcl::transformPointCloud(*source, *transformed, transform);
        } catch (...) {
            return std::numeric_limits<double>::infinity();
        }

        double total_error = 0.0;
        int count = 0;

        for (const auto& point : transformed->points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
                continue;
            }

            Eigen::Vector2f pt(point.x, point.y);
            float min_distance = std::numeric_limits<float>::max();

            for (const auto& line : lines) {
                Eigen::Vector2f projection = projectPointToLine(pt, line);
                
                if (!std::isfinite(projection.x()) || !std::isfinite(projection.y())) {
                    continue;
                }

                float distance = (pt - projection).norm();
                
                if (std::isfinite(distance)) {
                    min_distance = std::min(min_distance, distance);
                }
            }

            if (min_distance < params_.max_correspondence_distance) {
                total_error += min_distance * min_distance;
                count++;
            }
        }

        return (count > 0) ? (total_error / count) : 
               std::numeric_limits<double>::infinity();
    }
};

} // namespace icp_comparator

#endif // ICP_POINT_TO_LINE_HPP