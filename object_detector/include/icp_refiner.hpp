#pragma once
// icp_refiner.hpp (header-only, GCC14-safe)
// - 디폴트 인자 없음
// - 파라미터 struct를 클래스 밖으로 이동
// - PCL PointXYZI 기준 Point-to-Point ICP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>

struct IcpParams {
  int   max_iterations;
  float max_corr_dist;
  double trans_eps;
  double fit_eps;
  float voxel_leaf;
  bool  use_downsample;
  bool  reject_far;
  float reject_radius;

  IcpParams()
    : max_iterations(20),
      max_corr_dist(0.1f),
      trans_eps(1e-6),
      fit_eps(1e-5),
      voxel_leaf(0.2f),
      use_downsample(true),
      reject_far(false),
      reject_radius(3.0f){}
};

class IcpRefiner {
public:
  using PXYZI = pcl::PointXYZI;
  using Cloud = pcl::PointCloud<PXYZI>;
  using CloudPtr = typename Cloud::Ptr;
  using CloudConstPtr = typename Cloud::ConstPtr;

  explicit IcpRefiner(const IcpParams& p) : p_(p) {}
  IcpRefiner() : p_(IcpParams()) {}

  // src_in_curr: (오도메 1차 정렬된) 과거 스캔, 현재 프레임 좌표계
  // tgt_curr   : 현재 스캔 (타겟)
  // init_delta : 초기 델타(보통 Identity)
  inline Eigen::Matrix4f refine(const CloudConstPtr& src_in_curr,
                                const CloudConstPtr& tgt_curr,
                                const Eigen::Matrix4f& init_delta,
                                double* out_fitness = nullptr)
  {
    CloudPtr src_filt = radiusFilter(src_in_curr, p_.reject_radius);
    CloudPtr tgt_filt = radiusFilter(tgt_curr,   p_.reject_radius);
    CloudPtr src_ds   = downsample(src_filt);
    CloudPtr tgt_ds   = downsample(tgt_filt);

    pcl::IterativeClosestPoint<PXYZI, PXYZI> icp;
    icp.setMaxCorrespondenceDistance(p_.max_corr_dist);
    icp.setMaximumIterations(p_.max_iterations);
    icp.setTransformationEpsilon(p_.trans_eps);
    icp.setEuclideanFitnessEpsilon(p_.fit_eps);
    icp.setInputSource(src_ds);
    icp.setInputTarget(tgt_ds);

    // 초기 델타 적용
    Cloud src_init;
    pcl::transformPointCloud(*src_ds, src_init, init_delta);

    // 정합 실행
    Cloud aligned;
    icp.align(aligned);

    if (out_fitness) *out_fitness = icp.getFitnessScore();

    Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
    if (icp.hasConverged()) {
      // 전체 보정: init_delta * icp_delta
      delta = init_delta * icp.getFinalTransformation();
    } else {
      // 수렴 실패 시 초기 델타 유지
      delta = init_delta;
    }
    return delta;
  }

private:
  IcpParams p_;

  inline CloudPtr downsample(const CloudConstPtr& in) const {
    if (!p_.use_downsample || p_.voxel_leaf <= 0.f)
      return CloudPtr(new Cloud(*in));
    pcl::VoxelGrid<PXYZI> vg;
    vg.setLeafSize(p_.voxel_leaf, p_.voxel_leaf, p_.voxel_leaf);
    vg.setInputCloud(in);
    CloudPtr out(new Cloud);
    vg.filter(*out);
    return out;
  }

  inline CloudPtr radiusFilter(const CloudConstPtr& in, float radius) const {
    if (!p_.reject_far)
      return CloudPtr(new Cloud(*in));
    CloudPtr out(new Cloud);
    out->reserve(in->size());
    const float r2 = radius * radius;
    for (const auto& pt : in->points) {
      if ((pt.x*pt.x + pt.y*pt.y + pt.z*pt.z) <= r2)
        out->push_back(pt);
    }
    return out;
  }
};
