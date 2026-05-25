uint8 MOTION_UNKNOWN=0
uint8 STATIC=1
uint8 DYNAMIC=2

std_msgs/Header header

float32 center_x
float32 center_y

float32 min_x
float32 min_y
float32 max_x
float32 max_y

float32 min_range
float32 max_range

float32 width
uint32 point_count

uint32 track_id
uint8 motion_label

float32 velocity_x
float32 velocity_y
float32 speed
float32 compensated_displacement

float32 risk_weight
