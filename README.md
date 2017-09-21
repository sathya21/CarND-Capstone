## CarND Capstone Project
_____________________________________________________
## Team:
  1. Jana Punuru ( jpunuru@gmail.com )
  2. Sathya Narayanan Nagarajan  ( sathya21@gmail.com )
  3. Friedrich Erbs ( friedricherbs@gmail.com )
  4. Marcus Erbar ( marcus.erbar@gmail.com )
  5. James Jackson ( fiberhog@gmail.com )


## Overview:
Main goal of this project is to implement modules in ros for self driving system to follow waypoints , detect traffic lights, and control the vehicle based on the status of the traffic light. As described in the project, it is divided into three components described below.

### 1. Computing Waypoints to follow:
The provided system supplies an array of waypoints covering the entire track via ``/base_waypoints``. 

``waypoint_update.py`` determines the closest waypoint in travel direction and builds a new array, looking 50 waypoints down the road if there is no red traffic light ahead. In the case of a red light, it only extracts new waypoints up to the traffic light's stopping line.

### 2. Traffic Light Detection and Classification:
Once we make the car able to follow the waypoints consistently on the simulator, next we spent on how it detects traffic light status. For this, initially, we implemented heuristic by just count number red pixels, then we tried with simpler two layer feedforward net, and finally we ended up implementing much more sophisticated model as described below.

The traffic light module localizes and classifies traffic lights within the camera image stream. The chosen approach relies on two phases. The first phase uses deep learning to generate candidate bounding boxes for the traffic lights. The second phase validates the bounding boxes, and uses image thresholding to deterministically classify the traffic lights (RED, YELLOW, GREEN, UNKNOWN).

**Phase 1: Localization/Object Detection**

187 real world images and 99 simulator images are randomly sampled from the full datasets. The [RectLabel](https://rectlabel.com/) application is used to manually draw bounding boxes and assign the "traffic light" labels. Given the constraints on external dependencies, the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection) is chosen, and a custom script is used to create TFrecords from the images and their associated annotations ([PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) format). Real world and simulator datasets are kept separate, and each is split 70/30 into training and validation sets. Training/validation is performed on an AWS GPU instance (g3.4xlarge), the jobs are monitored via [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard), and select checkpoints are exported to Tensorflow graph protos. A comparison of the Tensorflow detection models pre-trained on the [COCO](http://mscoco.org) dataset is shown [here](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md). A model based on Faster R-CNN with Resnet-101 performs very well (highly accurate bounding boxes) but takes ~330ms for detection on GPU (Tesla M60). SSD MobileNet v1 brings the detection time all the way down to ~20ms, but accuracy takes a noticeable hit (poorly fit bounding boxes). SSD Inception v2 brings the right balance with ~32ms detection time, and good bounding boxes.

**Phase 2: Bounding Box Validation and Color Classification**

A bounding box with a score over 0.5 is chosen for color classification. In the case of multiple bounding boxes, the one with the highest score is chosen. Both the simulator and real-world environments are restricted to considering only a single traffic light at a time. A simple check is performed on the ratio of height to width for the bounding box. The bounding box image is grayscaled, and thresholded to black and white, at 190 for the real-world images, and at 100 for the simulator images. This results in the active light appearing as a white circle on a mostly black background. Segmenting the bounding box image into 3 sections, and comparing the non-zero pixel densities allows the color to be determined. Note that the determination is based on the relative position of the light, rather than the actual color. For the real-world images, there is very little color variation between the red and yellow lights.

**Accuracy**

For real world images, the detection/classification failure rate is 1% for red light images, 0% for yellow light images, 0% for green light images, and 0% for images with no lights. The red light failures are missed detections (some due to occlusions), where processing subsequent images succeeds.

**IMPORTANT NOTE ON TESTING**

There are separate detection/classification models for the simulator and the real world. To test with the simulator, set "SIMULATOR_MODEL = True" in tl_classifier.py. To test in the real world, set "SIMULATOR_MODEL = False".

### 3. Updating Waypoints:

Once the traffic light state is detected from the subscribed image using the light classifier, if it is detected as red, corresponding waypoint computed using data from `/traffic_light_config`.  Here, we identify closest waypoint associated with traffic light from the car's current position. After some experimentation, we decided to use a threshold of 2. It means, traffic light needs to be detected as red at least two times before publishing corresponding waypoint ``/traffic_waypoint`` topic.  

In the waypoint\_updater,  we subscribe to the ``/traffic_waypoint`` to receive the index of the waypoint at which car has to complete stop. Once we get waypoint index at which the cars is expected stop, we extracted map coordinates, along x axis, for next waypoint and traffic lights waypoint.  As shown in ``waypoint_updater.py``, we compute interpolation with current velocity for next_waypoint and 0.0 for waypoint at traffic light. Corresponding code is as shown below:

```python
sp_wp_i = [to_red_wp_count, 0]
next_wp_velocity = self.get_waypoint_velocity(self.base_waypoints.waypoints[next_wp_i])
if next_wp_velocity > SPEED_LIMIT:
     next_wp_velocity = SPEED_LIMIT
sp_v = [next_wp_velocity, 0.0]
self.f_sp = interp1d(sp_wp_i, sp_v)

```
Computed spline is used set the velocities for the following waypoints until waypoint for the traffic light.  if -1 in is received from `/traffic_waypoint` topic, indicating traffic light status is changed. Next set of waypoint are set maximum speed limit i.e. 10mph.
 
## Vehicle Actuation
We control inputs to the vehicle's throttle, braking and steering subsytems in ``dbw_node.py``. 

All three are controlled with simple PID Controllers whose outputs are further smoothed via separate low-pass filters. Input errors for the controllers are derived from the current velocity of the vehicle in ``/current_velocity`` and suggested steering angles in ``/twist_cmd``. 

In case of a manual override by the safety driver, the PID controllers and filters reset to zero.

## Conclusion

 After implementing the steps, mapping waypoint for the follow, publishing throttle, brake, steering only dbw_enabled, implementing traffic_light detection module,  implementing updating velocities for waypoint based traffic light status, we tested with  on simulator multiples. Also tested with rosbag files as well. Based on our implementation working well with the simulator and rosbag files.

