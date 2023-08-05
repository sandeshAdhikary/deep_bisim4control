#!/bin/bash
echo "Setting up X11 authentication..."
# Set SDL library to use the X11 driver
export SDL_VIDEODRIVER=X11
# Set location of the X11 socket
export XSOCK=/tmp/.X11-unix
# Set location of the X11 authentication file
export XAUTH=/tmp/.docker.xauth
# Copy the X11 authentication cookie to the container (so that the container can access the X11 display)
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | sudo xauth -f $XAUTH nmerge -
# Change permission on xauth file so container can access it
sudo chmod 777 $XAUTH