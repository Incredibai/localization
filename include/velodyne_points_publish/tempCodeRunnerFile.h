//// Velodyne datastructures
//// one laser, 3 Bytes
//typedef struct hdl_laser {
//  u_int16_t distance;                   // 0-65536*2mm
//  unsigned char intensity;              // 0-255, the greater, the brighter
//} __attribute__((packed)) hdl_laser_t;

//// one shot, 100 Bytes
//typedef struct hdl_shot {
//  u_int16_t lower_upper;
//  u_int16_t rotational_angle;
//  hdl_laser_t lasers[HDL32_NUM_LASERS];
//} __attribute__((packed)) hdl_shot_t;

//// one packet 1206 Bytes
//typedef struct hdl_packet {
//    hdl_shot_t shots[HDL32_NUM_SHOTS];
//    u_int8_t GPS_time_stamp[4];
//    u_int8_t Factory[2];
//}  __attribute__((packed)) hdl_packet_t;