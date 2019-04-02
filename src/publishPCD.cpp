#include <ros/ros.h>
#include <iostream>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/io/pcd_io.h>

int main(int argc, char **argv) {
    int pubVel = 1;
    ros::init (argc, argv, "publishPCD");
    ros::NodeHandle nh; 
    // Create a ROS publisher for the output point cloud
    ros::Publisher pubCloudCorner = nh.advertise<sensor_msgs::PointCloud2> ("my_laser_cloud_corner_surround", 5); 
    ros::Publisher pubCloudSurf = nh.advertise<sensor_msgs::PointCloud2> ("my_laser_cloud_surf_surround", 5);   

    pcl::PCLPointCloud2::Ptr cloud2_Corner(new pcl::PCLPointCloud2); 
    pcl::PCLPointCloud2::Ptr cloud2_Surf(new pcl::PCLPointCloud2); 
    pcl::io::loadPCDFile ("/home/william/data/LeGO-LOAM/saved/cornerMap.pcd", *cloud2_Corner);
    pcl::io::loadPCDFile ("/home/william/data/LeGO-LOAM/saved/surfMap.pcd", *cloud2_Surf);
    // Convert to ROS data type
    sensor_msgs::PointCloud2 output_Corner;
    sensor_msgs::PointCloud2 output_Surf;
    pcl_conversions::fromPCL(*cloud2_Corner, output_Corner);
    pcl_conversions::fromPCL(*cloud2_Surf, output_Surf);
    output_Corner.header.frame_id = "/camera_init";
    output_Surf.header.frame_id = "/camera_init";

    ros::Rate rate(pubVel);
    int count = 1;
    while(ros::ok()){
        std::cout << "publishing...count = " << count << std::endl;
        pubCloudCorner.publish (output_Corner);
        pubCloudSurf.publish (output_Surf);
        rate.sleep();
        count++;
    }
 
    return 0;
}