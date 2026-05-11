#pragma once

#include <geometry_msgs/msg/quaternion.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace wall_map
{

class StaticWallMap
{
public:
  StaticWallMap(bool unknown_occupied = true, int occupied_threshold = 50);

  bool load_from_yaml(const std::string & yaml_path);
  bool load_from_config(const std::string & config_path);
  void set_map(const nav_msgs::msg::OccupancyGrid & map_msg);

  bool has_map() const;
  const nav_msgs::msg::OccupancyGrid & map() const;

  std::optional<std::pair<int, int>> world_to_map(double x, double y) const;
  std::optional<int8_t> value_at_world(double x, double y) const;
  std::optional<bool> is_wall_at(double x, double y) const;
  bool is_wall_value(int8_t value) const;

private:
  struct map_yaml
  {
    std::string image;
    double resolution = 0.0;
    double origin_x = 0.0;
    double origin_y = 0.0;
    double origin_yaw = 0.0;
    int negate = 0;
    double occupied_thresh = 0.65;
    double free_thresh = 0.196;
  };

  static bool parse_yaml(const std::string & yaml_path, map_yaml & out);
  static bool read_pgm(
    const std::string & pgm_path, int & width, int & height, std::vector<uint8_t> & pixels);
  static std::string resolve_config_path(const std::string & config_path);
  static std::string resolve_yaml_path(const std::string & yaml_path);
  static std::string resolve_image_path(const std::string & yaml_path, const std::string & image_field);
  static double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & q);
  static size_t cell_index(int mx, int my, int width);

  bool load_grid_from_yaml(const std::string & yaml_path, nav_msgs::msg::OccupancyGrid & out) const;
  void cache_map();

  bool unknown_occupied{true};
  int occupied_threshold{50};
  bool loaded{false};

  nav_msgs::msg::OccupancyGrid map_msg;

  double resolution{0.0};
  int width{0};
  int height{0};
  double origin_x{0.0};
  double origin_y{0.0};
  double cos_yaw{1.0};
  double sin_yaw{0.0};
};

}  // namespace wall_map
