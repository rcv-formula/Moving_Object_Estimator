## Usage

```cpp
#include "wall_map/wall.hpp"
using wall_map::StaticWallMap;

StaticWallMap wall_map;

if (!wall_map.load_from_yaml("0120.yaml")) {
    return;
}

// 또는 설치된 config/config.yaml 사용
if (!wall_map.load_from_config("config.yaml")) {
    return;
}

// x,y는 obstacle 체크 후보
auto is_wall = wall_map.is_wall_at(x, y);
if (is_wall && *is_wall) {
}

auto value = wall_map.value_at_world(x, y);
if (value) {
    std::cout << "Occupancy value: " << static_cast<int>(*value) << std::endl;
}
```
