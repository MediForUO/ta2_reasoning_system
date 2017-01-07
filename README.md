### ta2_reasoning_system

###Short Description:

Medifor, University of Oregon

###Full Description:

Medifor Project. Update GitHub for the Docker Image to auto build on the Docker Cloud.

###GitHub page:

https://github.com/MediForUO/ta2_reasoning_system

Our Docker Page:

https://hub.docker.com/r/mediforuo/mediforuota2/

###Pull current image from Docker:

docker pull mediforuo/mediforuota2

###Run current image from docker:

Step1: Pull latest docker image: docker pull mediforuo/mediforuota2

Step2: Run docker image:  docker run -ti --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY mediforuo/mediforuota2

###Build and Run current project from GitHub 
(uses docker to build new image):

./build

./run

###To Run the program locally:

Step1: Install all dependencies: https://github.com/MediForUO/ta2_reasoning_system/blob/master/install_dependencies.txt

Step2: python3 MediForUI.py
