#include <ros/ros.h>
#include <tf/transform_listener.h>

int main(int argc, char** argv){
  ros::init(argc, argv, "my_tf_listener");

  ros::NodeHandle node;
  tf::TransformListener listener;

  ros::Rate rate(10.0);
  while (node.ok()){
    tf::StampedTransform transform;
    try{
      listener.lookupTransform("/tf_end", "/tf_begin", // transform from the latter to the former  
                               ros::Time(0), transform);
    }
    catch (tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
      ros::Duration(1.0).sleep();
    }

    std::cout << "transform stamp: " << transform.stamp_ << std::endl;    
    std::cout << "transform getOrigin: " << transform.getOrigin().x() << ", " << transform.getOrigin().y()
    << ", " << transform.getOrigin().z() << std::endl;
    std::cout << "transform getRotation: " << -transform.getRotation().getY() << ", "
    << -transform.getRotation().getZ() << ", " << transform.getRotation().getX() << ", "
    << transform.getRotation().getW() << std::endl;

    rate.sleep();
  }
  return 0;
};