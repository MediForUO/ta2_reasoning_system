# ta2_reasoning_system

Short Description:

Medifor, University of Oregon

Full Description:

Medifor Project. Update GitHub for the Docker Image to auto build on the Docker Cloud.

GitHub page:

https://github.com/MediForUO/ta2_reasoning_system

To view our Docker Page:

https://hub.docker.com/r/mediforuo/mediforuota2/

To Pull current project from Docker:

docker pull mediforuo/mediforuota2

To Build current project from Docker:

You shouldn't need to build the proect image, docker will do this once GitHub is updated.

To Build and run current project from GitHub:

./build

./run

To Run current project from docker, after pull:

docker run -ti --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY mediforuo/mediforuota2


