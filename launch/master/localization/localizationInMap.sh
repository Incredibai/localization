#!/bin/bash
roslaunch localizationInMap.launch & 
sleep 12
echo "localizationInMap started success!" 
roslaunch velodynePointsPublish.launch & 
sleep 0.1 
wait 
exit 0

