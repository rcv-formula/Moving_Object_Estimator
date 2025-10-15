#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <limits>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "map_manager_pair.hpp"

class DynamicObjectDetector {
public:

  DynamicObjectDetector() = default;

  void set_threshold1(double v)        { threshold1_ = std::max(0.0, v); }
  void set_threshold2(double v)        { threshold2_ = std::max(0.0, v); }
  void set_target_frame_idx(int idx)   { target_frame_idx_ = idx; }

  double threshold1() const            { return threshold1_; }
  double threshold2() const            { return threshold2_; }
  int    target_frame_idx() const      { return target_frame_idx_; }

  inline Eigen::Matrix4d poseToMatrix(const geometry_msgs::msg::Pose& p) {
    Eigen::Quaterniond q(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = q.normalized().toRotationMatrix();
    T(0,3) = p.position.x;
    T(1,3) = p.position.y;
    T(2,3) = p.position.z;
    return T;
  }

  inline std::vector<std::vector<geometry_msgs::msg::PointStamped>>
  transformObjHistoryToCurrentFrames(
      const std::vector<std::pair<std::vector<geometry_msgs::msg::PointStamped>, geometry_msgs::msg::Pose>>& deque_snapshot_obj_pairs,
      const geometry_msgs::msg::Pose& current_pose)
  {
    std::vector<std::vector<geometry_msgs::msg::PointStamped>> out;
    out.reserve(deque_snapshot_obj_pairs.size());

    // current(map) 변환: map->current, so current->map is inverse
    const Eigen::Matrix4d T_map_curr = poseToMatrix(current_pose);
    const Eigen::Matrix4d T_curr_map = T_map_curr.inverse();

    for (const auto& pr : deque_snapshot_obj_pairs) {
      const auto& frame_pts  = pr.first;   // 과거 프레임의 로컬 점들
      const auto& hist_pose  = pr.second;  // 그 프레임이 찍힐 당시의 pose (map 기준)

      // hist(local) -> map
      const Eigen::Matrix4d T_map_hist  = poseToMatrix(hist_pose);
      // hist(local) -> current
      const Eigen::Matrix4d T_curr_hist = T_curr_map * T_map_hist;

      std::vector<geometry_msgs::msg::PointStamped> frame_out;
      frame_out.reserve(frame_pts.size());

      for (const auto& ps : frame_pts) {
        // ps.point 는 "그때의 로컬 프레임" 좌표라고 가정
        Eigen::Vector4d p_hist(ps.point.x, ps.point.y, ps.point.z, 1.0);
        Eigen::Vector4d p_curr = T_curr_hist * p_hist;

        geometry_msgs::msg::PointStamped ps_out;
        ps_out.header.stamp = ps.header.stamp;     // 타임스탬프는 원본 유지(원하면 현재로 바꿔도 무방)
        // ps_out.header.frame_id 는 여기서 비워두고, 시각화 시 Marker의 frame_id로 지정하는 걸 권장
        ps_out.point.x = p_curr.x();
        ps_out.point.y = p_curr.y();
        ps_out.point.z = p_curr.z();

        frame_out.emplace_back(std::move(ps_out));
      }
      out.emplace_back(std::move(frame_out));
    }
    return out;
  }

    inline int isDynamic(
      const std::vector<std::vector<geometry_msgs::msg::PointStamped>>& aligned_frames,
      const geometry_msgs::msg::Point& obj_point  // ← const& 권장
  ) const
  {
    using PXYZ = pcl::PointXYZ;

    // 프레임 최소 2장 필요(타깃 + 현재)
    if (aligned_frames.size() < 2) return -1;

    const int curr_idx = static_cast<int>(aligned_frames.size()) - 1;

    // 타깃 프레임 인덱스 정규화: 유효하지 않거나 현재 프레임이면 바로 이전 프레임 사용
    int tgt = target_frame_idx_;
    if (tgt < 0 || tgt >= static_cast<int>(aligned_frames.size()) || tgt == curr_idx) {
      tgt = curr_idx - 1;
      if (tgt < 0) return -1;
    }

    const auto& tgt_pts = aligned_frames[tgt];
    if (tgt_pts.empty()) return -1;

    // 타깃 프레임 KD-Tree
    pcl::PointCloud<PXYZ>::Ptr tgt_cloud(new pcl::PointCloud<PXYZ>());
    tgt_cloud->reserve(tgt_pts.size());
    for (const auto& ps : tgt_pts) {
      PXYZ p;
      p.x = static_cast<float>(ps.point.x);
      p.y = static_cast<float>(ps.point.y);
      p.z = static_cast<float>(ps.point.z);
      tgt_cloud->push_back(p);
    }
    if (tgt_cloud->empty()) return -1;

    pcl::KdTreeFLANN<PXYZ> kdtree;
    kdtree.setInputCloud(tgt_cloud);

    // 현재 장애물의 대표점(obj_point)만을 사용하여 반경 판정
    PXYZ q;
    q.x = static_cast<float>(obj_point.x);
    q.y = static_cast<float>(obj_point.y);
    q.z = static_cast<float>(obj_point.z);

    const float t1 = static_cast<float>(threshold1_);
    const float t2 = static_cast<float>(threshold2_);

    // 1) threshold1 반경 내 존재 여부 → 정적(0)
    {
      std::vector<int> idx;
      std::vector<float> sq;
      int found = kdtree.radiusSearch(q, t1, idx, sq);
      if (found > 0) return 0;  // static
    }

    // 2) (t1, t2] 반경 내 존재 여부 → 동적(1)
    {
      std::vector<int> idx;
      std::vector<float> sq;
      int found = kdtree.radiusSearch(q, t2, idx, sq);
      if (found > 0) return 1;  // dynamic
    }

    // 3) t2 초과 → unknown(-1)
    return -1;
  }

  inline void publishAlignedFramesMarkers(
      const std::vector<std::vector<geometry_msgs::msg::PointStamped>> &aligned_frames,
      const std::string &frame_id,
      const rclcpp::Time &stamp,
      const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr &pub,
      double point_scale = 0.06,
      double lifetime_sec = 0.25) const
  {
    if (!pub || aligned_frames.empty()) return;

    visualization_msgs::msg::MarkerArray arr;

    // 기존 마커 지우기
    {
      visualization_msgs::msg::Marker del;
      del.header.frame_id = frame_id;
      del.header.stamp    = stamp;
      del.ns              = "aligned_frames";
      del.id              = 0;
      del.action          = visualization_msgs::msg::Marker::DELETEALL;
      arr.markers.push_back(del);
    }

    const int N = static_cast<int>(aligned_frames.size());
    int next_id = 1;

    for (int i = 0; i < N; ++i) {
      const auto &frame_pts = aligned_frames[i];
      if (frame_pts.empty()) continue;

      visualization_msgs::msg::Marker m;
      m.header.frame_id = frame_id;
      m.header.stamp    = stamp;
      m.ns              = "aligned_frames";
      m.id              = next_id++;
      m.type            = visualization_msgs::msg::Marker::SPHERE_LIST;
      m.action          = visualization_msgs::msg::Marker::ADD;
      m.scale.x = m.scale.y = m.scale.z = point_scale;

      // 오래된 프레임일수록 어둡게: hue는 고정하고 value만 낮춤(또는 hue를 프레임별로 바꿔도 됨)
      // 여기서는 hue를 프레임별로 조금씩 변화 + 밝기는 (i/N)로 점점 어둡게
      float r=1, g=1, b=1;
      double hue = ( (N > 1) ? (static_cast<double>(i) / (N-1)) : 0.0 ); // 0~1
      double val = 0.95 * (0.35 + 0.65 * (1.0 - static_cast<double>(i)/std::max(1, N-1))); // 최신이 더 밝게
      hsvToRgb(hue, 0.9, val, r, g, b);
      m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.95f;

      // 점들 채우기
      m.points.reserve(frame_pts.size());
      for (const auto &ps : frame_pts) {
        geometry_msgs::msg::Point p;
        p.x = ps.point.x; p.y = ps.point.y; p.z = ps.point.z;
        m.points.push_back(p);
      }

      m.lifetime = rclcpp::Duration::from_seconds(lifetime_sec);
      arr.markers.push_back(m);

      // 프레임 인덱스 텍스트(선택)
      visualization_msgs::msg::Marker t;
      t.header.frame_id = frame_id;
      t.header.stamp    = stamp;
      t.ns              = "aligned_frames_text";
      t.id              = next_id++;
      t.type            = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      t.action          = visualization_msgs::msg::Marker::ADD;
      t.scale.z         = point_scale * 3.0;
      t.color.r = t.color.g = t.color.b = 1.0f; t.color.a = 0.9f;

      // 레이블은 프레임 중앙 근처에 표시
      geometry_msgs::msg::Point center{};
      for (const auto &ps : frame_pts) { center.x += ps.point.x; center.y += ps.point.y; center.z += ps.point.z; }
      center.x /= frame_pts.size(); center.y /= frame_pts.size(); center.z /= frame_pts.size();
      center.z += point_scale * 2.0; // 살짝 위
      t.pose.position = center;

      char buf[64];
      std::snprintf(buf, sizeof(buf), "frame %d/%d", i, N-1);
      t.text = buf;
      t.lifetime = rclcpp::Duration::from_seconds(lifetime_sec);
      arr.markers.push_back(t);
    }

    pub->publish(arr);
  }
  inline std::vector<geometry_msgs::msg::Point>
  extractNearestCluster(
      const geometry_msgs::msg::Point& obj,
      const std::vector<std::vector<geometry_msgs::msg::PointStamped>>& aligned_frames,
      double eps = 0.12,          // DBSCAN half-width
      int    minPts = 3,          // DBSCAN min points
      double search_radius = 0.60,// 후보 탐색 반경(성능/오류 완화)
      bool   exclude_current = true
  ) const
  {
    struct P2 { double x, y; };
    std::vector<P2> pool; pool.reserve(1024);
    if (aligned_frames.empty()) return {};

    // 1) obj 주변 search_radius 안의 포인트만 모아서 후보 풀 구성
    const double R2 = search_radius * search_radius;
    const int last = static_cast<int>(aligned_frames.size()) - 1;
    const int end_idx = exclude_current ? last : last + 1; // 현재 프레임 제외 여부

    for (int i = 0; i < end_idx; ++i) {
      for (const auto& ps : aligned_frames[i]) {
        const double dx = ps.point.x - obj.x;
        const double dy = ps.point.y - obj.y;
        if (!std::isfinite(dx) || !std::isfinite(dy)) continue;
        if (dx*dx + dy*dy <= R2) pool.push_back({ps.point.x, ps.point.y});
      }
    }
    if (pool.size() < static_cast<size_t>(minPts)) return {};

    // 2) DBSCAN (2D)
    const double eps2 = eps * eps;
    const int N = static_cast<int>(pool.size());
    std::vector<int> labels(N, -1);  // -1: unvisited, -2: noise, >=0: cluster id

    auto regionQuery = [&](int i){
      std::vector<int> nb; nb.reserve(32);
      for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        const double dx = pool[i].x - pool[j].x;
        const double dy = pool[i].y - pool[j].y;
        if (dx*dx + dy*dy <= eps2) nb.push_back(j);
      }
      return nb;
    };

    int cid = 0;
    for (int i = 0; i < N; ++i) {
      if (labels[i] != -1) continue;
      auto neigh = regionQuery(i);
      if (static_cast<int>(neigh.size()) + 1 < minPts) { labels[i] = -2; continue; } // noise
      labels[i] = cid;
      std::vector<int> seeds = neigh;
      for (size_t k = 0; k < seeds.size(); ++k) {
        int j = seeds[k];
        if (labels[j] == -2) labels[j] = cid;     // border → core
        if (labels[j] != -1) continue;            // visited
        labels[j] = cid;
        auto neigh_j = regionQuery(j);
        if (static_cast<int>(neigh_j.size()) + 1 >= minPts) {
          seeds.insert(seeds.end(), neigh_j.begin(), neigh_j.end());
        }
      }
      cid++;
    }

    if (cid == 0) return {};

    // 3) obj와 가장 가까운 클러스터 선택 (센트로이드 기준)
    auto dist = [](double ax, double ay, double bx, double by){
      const double dx = ax - bx, dy = ay - by;
      return std::sqrt(dx*dx + dy*dy);
    };

    int best_c = -1;
    double best_d = std::numeric_limits<double>::infinity();

    for (int c = 0; c < cid; ++c) {
      double sx=0.0, sy=0.0; int cnt=0;
      for (int i = 0; i < N; ++i) if (labels[i] == c) { sx += pool[i].x; sy += pool[i].y; cnt++; }
      if (cnt == 0) continue;
      const double cx = sx / cnt, cy = sy / cnt;
      const double d  = dist(cx, cy, obj.x, obj.y);
      if (d < best_d) { best_d = d; best_c = c; }
    }

    if (best_c < 0) return {};

    // 4) 선택된 클러스터의 포인트들을 geometry_msgs::msg::Point 로 반환
    std::vector<geometry_msgs::msg::Point> cluster;
    for (int i = 0; i < N; ++i) {
      if (labels[i] != best_c) continue;
      geometry_msgs::msg::Point p;
      p.x = pool[i].x; p.y = pool[i].y; p.z = 0.0;
      cluster.push_back(p);
    }
    return cluster;
  }

  static inline double computeFootprintSpan(const std::vector<geometry_msgs::msg::Point>& cluster)
  {
    if (cluster.size() < 2) return std::numeric_limits<double>::quiet_NaN();

    auto dist = [](const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b){
      const double dx=a.x-b.x, dy=a.y-b.y, dz=a.z-b.z;
      return std::sqrt(dx*dx + dy*dy + dz*dz);
    };

    // seed: 임의(0)
    const size_t n = cluster.size();
    size_t seed = 0;

    // 1-pass: seed에서 가장 먼 점 A
    size_t ia = seed; double da = -1.0;
    for (size_t i=0;i<n;++i){
      double d = dist(cluster[i], cluster[seed]);
      if (d > da) { da = d; ia = i; }
    }
    // 2-pass: A에서 가장 먼 거리 = 근사 지름
    double span = 0.0;
    for (size_t i=0;i<n;++i){
      double d = dist(cluster[i], cluster[ia]);
      if (d > span) span = d;
    }
    return span;
  }

  inline int classifyDynamicByFootprint(
      const geometry_msgs::msg::Point& obj,
      const std::vector<std::vector<geometry_msgs::msg::PointStamped>>& aligned_frames,
      double eps = 0.12,           // DBSCAN 반경
      int    minPts = 3,           // DBSCAN 최소 포인트 수
      double search_radius = 0.60, // obj 주변 후보 수집 반경
      bool   exclude_current = true,
      double motion_thresh = 0.10, // 10cm
      std::vector<geometry_msgs::msg::Point>* out_cluster = nullptr,
      double* out_span = nullptr
  ) const
  {
    // 1) obj와 가장 가까운 과거 클러스터 추출
    auto cluster = extractNearestCluster(obj, aligned_frames, eps, minPts, search_radius, exclude_current);
    if (out_cluster) *out_cluster = cluster;
    if (cluster.size() < static_cast<size_t>(minPts)) return -1;
    
    // 2) 발자취 span 계산
    const double span = computeFootprintSpan(cluster);
    if (out_span) *out_span = span;
    if (!std::isfinite(span)) return -1;

    // 3) 판정
    return (span > motion_thresh) ? 1 : 0; // 1=dynamic, 0=static
  }
  
private:

  static inline void hsvToRgb(double h, double s, double v, float &r, float &g, float &b) {
    double i = std::floor(h * 6.0);
    double f = h * 6.0 - i;
    double p = v * (1.0 - s);
    double q = v * (1.0 - f * s);
    double t = v * (1.0 - (1.0 - f) * s);
    switch (static_cast<int>(i) % 6) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
    }
  }

  double threshold1_{0.05};  // [m] 예: 아주 근접 (정적/정합으로 간주)
  double threshold2_{0.1};  // [m] 예: 경계 구간 상한 
  int    target_frame_idx_{5}; //default는 5 이전의 frame
};
