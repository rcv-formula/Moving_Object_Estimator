#pragma once
/**
 * @file dynamic_obstacle_detector.hpp
 * @brief 로컬-맵 정렬된 과거 프레임을 관리/시각화하고, footprint 기반 동적/정적 판정을 수행하는 보조 클래스
 *
 * 제공 기능:
 *  - transformObjHistoryToCurrentFrames(obj_pairs, current_pose):
 *      과거 오브젝트 점들을 현재 프레임(=현재 포즈 기준 로컬 좌표계)로 변환하여 프레임별 점 리스트 반환
 *  - publishAlignedFramesMarkers(aligned_frames, frame_id, stamp, pub, point_scale, alpha):
 *      정렬된 과거 오브젝트 프레임들을 MarkerArray로 시각화
 *  - classifyDynamicByFootprint(target, aligned_frames, eps, minPts, search_radius, exclude_current, motion_thresh, out_footprint, out_span):
 *      현재 타깃과 **반경 내에서 수집된 과거 프레임 점들** 사이의 최대 거리로 동적(1)/정적(0) 분류.
 *      단, 반경 내에서 실제로 수집된 **서로 다른 과거 프레임 수 ≤ 5**이면 UNKNOWN(-1).
 *  - visualizeFootprint(footprint, label, frame_id, id, stamp, pub):
 *      footprint를 MarkerArray로 시각화
 */

#include <functional>
#include <vector>
#include <cmath>
#include <limits>
#include <memory>
#include <algorithm>
#include <unordered_set>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

class DynamicObjectDetector {
public:
  // 라벨 정의: UNKNOWN=0, STATIC=1, DYNAMIC=2
  enum Label { UNKNOWN=0, STATIC=1, DYNAMIC=2 };

  DynamicObjectDetector() = default;

  // === 과거 obj (map 좌표) → 현재 프레임(현재 포즈 기준 로컬 좌표)로 정렬 ===
  std::vector<std::vector<geometry_msgs::msg::Point>>
  transformObjHistoryToCurrentFrames(
      const std::vector<std::pair<std::vector<geometry_msgs::msg::PointStamped>, geometry_msgs::msg::Pose>>& pairs_obj,
      const geometry_msgs::msg::Pose& current_pose_map) const
  {
    std::vector<std::vector<geometry_msgs::msg::Point>> out;
    out.reserve(pairs_obj.size());

    const Eigen::Matrix4d T_map_curr = poseToT(current_pose_map); // map->curr
    const Eigen::Matrix4d T_curr_map = T_map_curr.inverse();      // curr->map

    for (const auto& pair : pairs_obj) {
      const auto& pts  = pair.first;  // map 좌표에 있는 과거 점들

      std::vector<geometry_msgs::msg::Point> transformed;
      transformed.reserve(pts.size());

      for (const auto& ps : pts) {
        Eigen::Vector4d pw(ps.point.x, ps.point.y, ps.point.z, 1.0); // map
        Eigen::Vector4d pc = T_curr_map * pw;                        // curr(frame)
        geometry_msgs::msg::Point p; p.x = pc.x(); p.y = pc.y(); p.z = pc.z();
        transformed.push_back(p);
      }
      out.push_back(std::move(transformed));
    }
    return out;
  }

  // === 정렬된 과거 프레임 점들을 RViz Marker로 시각화 ===
  void publishAlignedFramesMarkers(
      const std::vector<std::vector<geometry_msgs::msg::Point>>& aligned_frames,
      const std::string& frame_id,
      const builtin_interfaces::msg::Time& stamp,
      const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& pub,
      double point_scale = 0.06,
      double alpha = 0.15) const
  {
    if (!pub) return;

    visualization_msgs::msg::MarkerArray arr;

    // DELETEALL
    {
      visualization_msgs::msg::Marker del;
      del.header.frame_id = frame_id;
      del.header.stamp    = stamp;
      del.ns   = "aligned_obj_history";
      del.id   = 0;
      del.action = visualization_msgs::msg::Marker::DELETEALL;
      arr.markers.push_back(del);
    }

    // SPHERE_LIST per frame
    for (size_t i = 0; i < aligned_frames.size(); ++i) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = frame_id;
      m.header.stamp    = stamp;
      m.ns   = "aligned_obj_history";
      m.id   = static_cast<int>(i + 1);
      m.type = visualization_msgs::msg::Marker::SPHERE_LIST;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.scale.x = point_scale; m.scale.y = point_scale; m.scale.z = point_scale;

      float r, g, b;
      hsvToRgb(static_cast<float>((i % 12) / 12.0), 0.8f, 0.9f, r, g, b);
      m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = static_cast<float>(alpha);

      for (const auto& p : aligned_frames[i]) m.points.push_back(p);
      arr.markers.push_back(m);
    }

    pub->publish(arr);
  }

  /**
   * @brief footprint 기반 분류 (UNKNOWN=-1, STATIC=0, DYNAMIC=1)
   *
   * 절차(현재↔과거 최대거리 + 반경 내 과거 프레임 수 검증):
   *  1) `aligned_frames`의 마지막 인덱스를 현재 프레임으로 가정.
   *  2) target(현재 프레임 좌표계) 주변 `search_radius` 내 footprint 점 수집.
   *  3) footprint 중 **서로 다른 과거 프레임 인덱스**의 개수를 센다(현재 프레임 제외).
   *     - 이 개수가 **5 이하**이면 UNKNOWN(-1) 반환.
   *  4) 과거 프레임 점들과 target 사이의 **최대 거리(move_max)** 계산.
   *     - `move_max >= motion_thresh` → DYNAMIC(1), 아니면 STATIC(0).
   *  5) `out_span`은 footprint의 x범위(max−min)를 반환(분류에는 미사용).
   */
  int classifyDynamicByFootprint(
      const geometry_msgs::msg::Point& target_in_current,
      const std::vector<std::vector<geometry_msgs::msg::Point>>& aligned_frames,
      double eps,                 // (미사용)
      int /*minPts*/,             // (미사용)
      double search_radius,
      bool   /*exclude_current*/, // (미사용)
      double motion_thresh,
      std::vector<geometry_msgs::msg::Point>* out_footprint,
      double* out_span) const
  {
    (void)eps;
    if (aligned_frames.empty()) return UNKNOWN;

    const int current_idx = static_cast<int>(aligned_frames.size()) - 1;

    // (B) 반경 내 footprint 수집 (프레임 인덱스 포함)
    std::vector<std::pair<geometry_msgs::msg::Point, int>> footprint_idx; // (점, 프레임인덱스)
    const double R2 = search_radius * search_radius;
    for (size_t f = 0; f < aligned_frames.size(); ++f) {
      for (const auto& p : aligned_frames[f]) {
        const double dx = p.x - target_in_current.x;
        const double dy = p.y - target_in_current.y;
        const double dz = p.z - target_in_current.z;
        const double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 <= R2) footprint_idx.emplace_back(p, static_cast<int>(f));
      }
    }

    if (out_footprint) {
      out_footprint->clear();
      out_footprint->reserve(footprint_idx.size());
      for (const auto& kv : footprint_idx) out_footprint->push_back(kv.first);
    }

    if (out_span) {
      if (footprint_idx.empty()) *out_span = 0.0;
      else {
        double xmin=+1e9, xmax=-1e9;
        for (const auto& kv : footprint_idx) {
          xmin = std::min(xmin, kv.first.x);
          xmax = std::max(xmax, kv.first.x);
        }
        *out_span = (xmax - xmin);
      }
    }

    // (C) 반경 내 footprint가 포함하는 **서로 다른 과거 프레임 수** 계산
    std::unordered_set<int> past_frames_in_footprint;
    for (const auto& kv : footprint_idx) {
      const int f = kv.second;
      if (f != current_idx) past_frames_in_footprint.insert(f);
    }
    // 과거 프레임 수가 2개 이하라면 UNKNOWN
    if (static_cast<int>(past_frames_in_footprint.size()) <= 3) {
      return UNKNOWN;
    }

    // (D) 현재 타깃 ↔ 과거 프레임 점들 간 최대 거리
    double move_sum = 0.0;
    double move_avg = 0.0;
    bool   found_past = false;
    std::vector<double> dist_vec;
    
    for (const auto& kv : footprint_idx) {
      const int f = kv.second;
      if (f == current_idx) continue;  // 과거만 확인
      const auto& p = kv.first;
      const double d = hypot3(p, target_in_current);
      dist_vec.emplace_back(d);
      found_past = true;
    }
    
    if (!found_past) {
      // 반경 내 점이 모두 현재 프레임뿐이면 과거 대비가 불가 → UNKNOWN이 타당
      return UNKNOWN;
    }

    std::sort(dist_vec.begin(),dist_vec.end(),std::greater<double>());

    if(dist_vec.size() >= 3 && dist_vec[0] < motion_thresh && dist_vec[1] < motion_thresh) return STATIC;

    return DYNAMIC;
  }

  // === footprint 시각화 ===
  void visualizeFootprint(
      const std::vector<geometry_msgs::msg::Point>& footprint,
      int dyn_label,                         // -1=unknown, 0=static, 1=dynamic
      const std::string& frame_id,
      int id_base,
      const builtin_interfaces::msg::Time& stamp,
      const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& pub) const
  {
    if (!pub) return;

    visualization_msgs::msg::MarkerArray arr;

    // 포인트 구름
    {
      visualization_msgs::msg::Marker pts;
      pts.header.frame_id = frame_id;
      pts.header.stamp    = stamp;
      pts.ns   = "footprint_points";
      pts.id   = id_base;
      pts.type = visualization_msgs::msg::Marker::SPHERE_LIST;
      pts.action = visualization_msgs::msg::Marker::ADD;
      pts.scale.x = 0.06; pts.scale.y = 0.06; pts.scale.z = 0.06;

      if (dyn_label == DYNAMIC) {         // 빨강
        pts.color.r = 1.0f; pts.color.g = 0.1f; pts.color.b = 0.1f; pts.color.a = 0.95f;
      } else if (dyn_label == STATIC) {   // 파랑
        pts.color.r = 0.1f; pts.color.g = 0.4f; pts.color.b = 1.0f; pts.color.a = 0.95f;
      } else {                            // UNKNOWN = 노랑
        pts.color.r = 1.0f; pts.color.g = 0.9f; pts.color.b = 0.1f; pts.color.a = 0.95f;
      }

      for (const auto& p : footprint) pts.points.push_back(p);
      pts.lifetime = rclcpp::Duration::from_seconds(0.25);
      arr.markers.push_back(pts);
    }

    // 외곽선(Convex Hull 유사 라이트 버전: x 정렬 후 polyline)
    if (footprint.size() >= 3) {
      std::vector<geometry_msgs::msg::Point> hull = polylineByX(footprint);

      visualization_msgs::msg::Marker line;
      line.header.frame_id = frame_id;
      line.header.stamp    = stamp;
      line.ns   = "footprint_outline";
      line.id   = id_base + 10000;
      line.type = visualization_msgs::msg::Marker::LINE_STRIP;
      line.action = visualization_msgs::msg::Marker::ADD;
      line.scale.x = 0.025;

      // UNKNOWN은 회백색, 나머지는 흰색
      if (dyn_label == UNKNOWN) { line.color.r = 0.8f; line.color.g = 0.8f; line.color.b = 0.8f; line.color.a = 0.8f; }
      else                      { line.color.r = 0.95f; line.color.g = 0.95f; line.color.b = 0.95f; line.color.a = 0.8f; }

      for (const auto& p : hull) line.points.push_back(p);
      line.points.push_back(hull.front()); // 닫기
      line.lifetime = rclcpp::Duration::from_seconds(0.25);
      arr.markers.push_back(line);
    }

    pub->publish(arr);
  }

private:
  static Eigen::Matrix4d poseToT(const geometry_msgs::msg::Pose &pose)
  {
    Eigen::Quaterniond q(pose.orientation.w,
                         pose.orientation.x,
                         pose.orientation.y,
                         pose.orientation.z);
    q.normalize();

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = q.toRotationMatrix();
    T(0,3) = pose.position.x;
    T(1,3) = pose.position.y;
    T(2,3) = pose.position.z;
    return T;
  }

  static void hsvToRgb(float h, float s, float v, float& r, float& g, float& b)
  {
    float i = std::floor(h * 6.0f);
    float f = h * 6.0f - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);
    switch (static_cast<int>(i) % 6) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
    }
  }

  static double hypot3(const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b)
  {
    const double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
  }

  static std::vector<geometry_msgs::msg::Point>
  polylineByX(const std::vector<geometry_msgs::msg::Point>& pts)
  {
    std::vector<geometry_msgs::msg::Point> v = pts;
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b){ return a.x < b.x; });
    if (v.size() < 3) return v;

    // 상단 껍질
    std::vector<geometry_msgs::msg::Point> up;
    for (const auto& p : v) {
      while (up.size() >= 2 && cross(up[up.size()-2], up.back(), p) <= 0.0) up.pop_back();
      up.push_back(p);
    }
    // 하단 껍질
    std::vector<geometry_msgs::msg::Point> lo;
    for (int i = static_cast<int>(v.size()) - 1; i >= 0; --i) {
      const auto& p = v[i];
      while (lo.size() >= 2 && cross(lo[lo.size()-2], lo.back(), p) <= 0.0) lo.pop_back();
      lo.push_back(p);
    }
    // 병합(마지막 중복 제거)
    up.pop_back();
    lo.pop_back();
    up.insert(up.end(), lo.begin(), lo.end());
    return up;
  }

  static double cross(const geometry_msgs::msg::Point& a,
                      const geometry_msgs::msg::Point& b,
                      const geometry_msgs::msg::Point& c)
  {
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
  }
};
