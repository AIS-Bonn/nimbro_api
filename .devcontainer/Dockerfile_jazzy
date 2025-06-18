FROM osrf/ros:jazzy-desktop-full

SHELL ["/bin/bash", "-c"]

# Install system dependencies and Python packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gdb git wget autoconf automake nano vim tmux curl \
    python3-dev python3-pip python3-scipy python3-matplotlib

RUN pip3 install --break-system-packages jupyter

WORKDIR /colcon_ws/src/nimbro_api

# Copy requirements file and install Python dependencies without upgrading pip
COPY nimbro_api/requirements.txt /colcon_ws/src/nimbro_api/requirements.txt
RUN pip3 install --break-system-packages -r requirements.txt

# Copy project files and build workspace
COPY nimbro_api /colcon_ws/src/nimbro_api
COPY nimbro_api_interfaces /colcon_ws/src/nimbro_api_interfaces

WORKDIR /colcon_ws
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source /colcon_ws/install/local_setup.bash" >> /root/.bashrc

# Entrypoint to source ROS and workspace environments
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/jazzy/setup.bash && source /colcon_ws/install/setup.bash && exec bash"]
