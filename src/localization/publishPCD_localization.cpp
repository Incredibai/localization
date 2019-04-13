#include <ros/ros.h>
#include <iostream>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/io/pcd_io.h>
#include <std_msgs/Bool.h>


bool GetMap = false;

void GetMapFlagHandler(const std_msgs::Bool msg){
    GetMap = true;
}

int main(int argc, char **argv) {
    int pubVel = 1;
    ros::init (argc, argv, "publishPCD");
    ros::NodeHandle nh; 

    ros::Subscriber subGetMapFlag = nh.subscribe<std_msgs::Bool>("/GetMapFlag", 1, &GetMapFlagHandler);
    // Create a ROS publisher for the output point cloud
    ros::Publisher pubCloudCorner = nh.advertise<sensor_msgs::PointCloud2> ("my_laser_cloud_corner_surround", 5); 
    ros::Publisher pubCloudSurf = nh.advertise<sensor_msgs::PointCloud2> ("my_laser_cloud_surf_surround", 5);   

    pcl::PCLPointCloud2::Ptr cloud2_Corner(new pcl::PCLPointCloud2); 
    pcl::PCLPointCloud2::Ptr cloud2_Surf(new pcl::PCLPointCloud2); 
    pcl::io::loadPCDFile ("/home/william/data/LeGO-LOAM/saved/KITTI/cornerMap.pcd", *cloud2_Corner);
    pcl::io::loadPCDFile ("/home/william/data/LeGO-LOAM/saved/KITTI/surfMap.pcd", *cloud2_Surf);
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
        ros::spinOnce();
        if(!GetMap){
            std::cout << "-*-*- Publishing Map " << std::endl;
            pubCloudCorner.publish (output_Corner);
            pubCloudSurf.publish (output_Surf);
        }
        
        if(GetMap){
            std::cout << "-*-*- Map Published " << std::endl;
        }
        rate.sleep();
    }
 
    return 0;
}