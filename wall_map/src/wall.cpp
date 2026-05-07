#include "wall_map/wall.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <fstream>

namespace wall_map
{

StaticWallMap::StaticWallMap(bool unknown_occupied_in, int occupied_threshold_in)
: unknown_occupied(unknown_occupied_in),
  occupied_threshold(occupied_threshold_in)
{
}

bool StaticWallMap::load_from_yaml(const std::string & yaml_path)
{
  nav_msgs::msg::OccupancyGrid loaded_map;
  if (!load_grid_from_yaml(yaml_path, loaded_map)) {
    return false;
  }

  set_map(loaded_map);
  return true;
}

bool StaticWallMap::load_from_config(const std::string & config_path)
{
  const std::string resolved_config_path = resolve_yaml_path(config_path);
  
  try {
    YAML::Node config = YAML::LoadFile(resolved_config_path);
    
    if (!config["map_file"]) {
      return false;
    }
    
    std::string map_file = config["map_file"].as<std::string>();
    
    // Resolve map file path relative to config file location
    const std::string config_dir = resolved_config_path.substr(0, resolved_config_path.find_last_of("/\\"));
    std::string full_map_path = config_dir + "/" + map_file;
    
    return load_from_yaml(full_map_path);
  } catch (const std::exception &) {
    return false;
  }
}

void StaticWallMap::set_map(const nav_msgs::msg::OccupancyGrid & map_msg_in)
{
  map_msg = map_msg_in;
  loaded = true;
  cache_map();
}

bool StaticWallMap::has_map() const
{
  return loaded;
}

const nav_msgs::msg::OccupancyGrid & StaticWallMap::map() const
{
  return map_msg;
}

std::optional<std::pair<int, int>> StaticWallMap::world_to_map(double x, double y) const
{
  if (!loaded || resolution <= 0.0 || width <= 0 || height <= 0) {
    return std::nullopt;
  }

  const double dx = x - origin_x;
  const double dy = y - origin_y;

  const double mx_float = (dx * cos_yaw + dy * sin_yaw) / resolution;
  const double my_float = (-dx * sin_yaw + dy * cos_yaw) / resolution;

  const int mx = static_cast<int>(std::floor(mx_float));
  const int my = static_cast<int>(std::floor(my_float));

  if (mx < 0 || my < 0 || mx >= width || my >= height) {
    return std::nullopt;
  }

  return std::make_pair(mx, my);
}

std::optional<int8_t> StaticWallMap::value_at_world(double x, double y) const
{
  const auto cell = world_to_map(x, y);
  if (!cell.has_value()) {
    return std::nullopt;
  }

  const auto [mx, my] = cell.value();
  const size_t index = cell_index(mx, my, width);
  if (index >= map_msg.data.size()) {
    return std::nullopt;
  }

  return map_msg.data[index];
}

std::optional<bool> StaticWallMap::is_wall_at(double x, double y) const
{
  const auto value = value_at_world(x, y);
  if (!value.has_value()) {
    return std::nullopt;
  }

  return is_wall_value(value.value());
}

bool StaticWallMap::is_wall_value(int8_t value) const
{
  if (value < 0) {
    return unknown_occupied;
  }

  return value >= occupied_threshold;
}

bool StaticWallMap::parse_yaml(const std::string & yaml_path, map_yaml & out)
{
  YAML::Node doc = YAML::LoadFile(yaml_path);

  if (!doc["image"] || !doc["resolution"] || !doc["origin"]) {
    return false;
  }

  out.image = doc["image"].as<std::string>();
  out.resolution = doc["resolution"].as<double>();

  const auto origin = doc["origin"];
  out.origin_x = origin[0].as<double>();
  out.origin_y = origin[1].as<double>();
  out.origin_yaw = origin[2].as<double>();

  if (doc["negate"]) {
    out.negate = doc["negate"].as<int>();
  }
  if (doc["occupied_thresh"]) {
    out.occupied_thresh = doc["occupied_thresh"].as<double>();
  }
  if (doc["free_thresh"]) {
    out.free_thresh = doc["free_thresh"].as<double>();
  }

  return true;
}

bool StaticWallMap::read_pgm(
  const std::string & pgm_path, int & width_out, int & height_out, std::vector<uint8_t> & pixels)
{
  std::ifstream input(pgm_path, std::ios::binary);
  if (!input) {
    return false;
  }

  std::string magic;
  input >> magic;
  if (magic != "P5" && magic != "P2") {
    return false;
  }

  auto skip_comments = [&input]() {
    while (input.peek() == '#') {
      std::string line;
      std::getline(input, line);
    }
  };

  skip_comments();
  input >> width_out;
  skip_comments();
  input >> height_out;
  skip_comments();

  int max_value = 0;
  input >> max_value;
  input.get();

  if (width_out <= 0 || height_out <= 0) {
    return false;
  }

  pixels.resize(static_cast<size_t>(width_out) * static_cast<size_t>(height_out));

  if (magic == "P5") {
    input.read(reinterpret_cast<char *>(pixels.data()), pixels.size());
    return static_cast<size_t>(input.gcount()) == pixels.size();
  }

  for (size_t i = 0; i < pixels.size(); ++i) {
    int value = 0;
    input >> value;
    if (!input.good()) {
      return false;
    }
    pixels[i] = static_cast<uint8_t>(std::clamp(value, 0, 255));
  }

  return true;
}

std::string StaticWallMap::resolve_image_path(
  const std::string & yaml_path, const std::string & image_field)
{
  if (!image_field.empty() && image_field.front() == '/') {
    return image_field;
  }

  const auto pos = yaml_path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return image_field;
  }

  return yaml_path.substr(0, pos + 1) + image_field;
}

std::string StaticWallMap::resolve_yaml_path(const std::string & yaml_path)
{
  if (!yaml_path.empty() && yaml_path.front() == '/') {
    return yaml_path;
  }

  const std::string package_share_dir = ament_index_cpp::get_package_share_directory("wall_map");
  return package_share_dir + "/maps/" + yaml_path;
}

double StaticWallMap::yaw_from_quaternion(const geometry_msgs::msg::Quaternion & q)
{
  const double siny = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny, cosy);
}

size_t StaticWallMap::cell_index(int mx, int my, int width_in)
{
  return static_cast<size_t>(my) * static_cast<size_t>(width_in) + static_cast<size_t>(mx);
}

bool StaticWallMap::load_grid_from_yaml(
  const std::string & yaml_path, nav_msgs::msg::OccupancyGrid & out) const
{
  const std::string resolved_yaml_path = resolve_yaml_path(yaml_path);

  map_yaml yaml_data;
  if (!parse_yaml(resolved_yaml_path, yaml_data)) {
    return false;
  }

  const std::string image_path = resolve_image_path(resolved_yaml_path, yaml_data.image);

  int map_width = 0;
  int map_height = 0;
  std::vector<uint8_t> pixels;
  if (!read_pgm(image_path, map_width, map_height, pixels)) {
    return false;
  }

  out.info.resolution = static_cast<float>(yaml_data.resolution);
  out.info.width = static_cast<uint32_t>(map_width);
  out.info.height = static_cast<uint32_t>(map_height);
  out.info.origin.position.x = yaml_data.origin_x;
  out.info.origin.position.y = yaml_data.origin_y;
  out.info.origin.position.z = 0.0;

  const double half_yaw = yaml_data.origin_yaw * 0.5;
  out.info.origin.orientation.w = std::cos(half_yaw);
  out.info.origin.orientation.x = 0.0;
  out.info.origin.orientation.y = 0.0;
  out.info.origin.orientation.z = std::sin(half_yaw);

  out.data.resize(static_cast<size_t>(map_width) * static_cast<size_t>(map_height));

  for (int row = 0; row < map_height; ++row) {
    for (int col = 0; col < map_width; ++col) {
      const size_t image_index = cell_index(col, row, map_width);
      const size_t grid_index = cell_index(col, map_height - 1 - row, map_width);

      const double occupancy_probability = (yaml_data.negate == 0) ?
        (1.0 - static_cast<double>(pixels[image_index]) / 255.0) :
        (static_cast<double>(pixels[image_index]) / 255.0);

      if (occupancy_probability > yaml_data.occupied_thresh) {
        out.data[grid_index] = 100;
      } else if (occupancy_probability < yaml_data.free_thresh) {
        out.data[grid_index] = 0;
      } else {
        out.data[grid_index] = -1;
      }
    }
  }

  return true;
}

void StaticWallMap::cache_map()
{
  resolution = static_cast<double>(map_msg.info.resolution);
  width = static_cast<int>(map_msg.info.width);
  height = static_cast<int>(map_msg.info.height);
  origin_x = map_msg.info.origin.position.x;
  origin_y = map_msg.info.origin.position.y;

  const double yaw = yaw_from_quaternion(map_msg.info.origin.orientation);
  cos_yaw = std::cos(yaw);
  sin_yaw = std::sin(yaw);
}

}  // namespace wall_map
