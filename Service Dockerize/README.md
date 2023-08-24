# Cats vs Dog Service Dockerized

The task involved creating a FastAPI-based image classification service for distinguishing between cats and dogs. I have used the two EfficientNetV2B2 models (scratched and pretrained) from previous task for the classification task. This service was then Dockerized to ensure easy deployment and consistent behavior across different environments and operating systems. 

## Workflow

**1. Model Conversion to ONNX:** I have converted the two pytorch models (that I have trained in previous task) to onnx format to use these models with different frameworks and platforms. ONNX (Open Neural Network Exchange) is an open-source standard/ open ecosystem for representing deep learning models. ONNX prevents developers from getting locked into any particular machine learning framework by providing tools that make it easy to move from one to the other. ONNX Runtime can be very useful since we can use models in inference with a single framework no matter what hardware we are going to use. So without having to actually rewrite the code depending on whether we want to use a CPU, GPU, FPGA or whatever.

**2. FastAPI Implementation:** I have created a FastAPI application (app.py) that serves as the backend for the image classification service. This application includes index, version, and two other routes, one for the scratch-trained model and another for the pretrained model. The defined routes handled image uploads and classification predictions. I utilized ONNX models for image classification that enables easy model interchangeability.

**3. Dockerization of the FastAPI Service:** I dockerized my FastAPI application using a Dockerfile (in the same directory where app.py is present). This Dockerfile specifies the environment and dependencies required to run the FastAPI service within a Docker container. In this Dockerfile, I set up the Docker image to use an official Python runtime as the parent image (I used official Python 3.9 slim image as the base image for the Docker container). The 'WORKDIR' instruction sets the working directory to '/app' inside the Docker container. So, when I run commands or copy files in subsequent instructions in the Dockerfile, they will be relative to the /app directory inside the container. According to that, I copied application code (app.py), ONNX models, and the requirements file into the container. The --no-cache-dir option tells pip to not save the downloaded packages locally, as that is only if pip was going to be run again to install the same packages, but that's not the case when working with containers. Because the previous step copying the file could be detected by the Docker cache, this step will also use the Docker cache when available. Using the cache in this step will save a lot of time when building the image again and again during development, instead of downloading and installing all the dependencies every time. After installing required Python packages specified in the requirements.txt file, I exposed port 80 to allow external access to the FastAPI application and configured the image to run the FastAPI application using Uvicorn.

**4. Building the Docker Image:** I built a docker image from the Dockerfile using the docker build command. This image contains the FastAPI application, the ONNX models, and the required Python dependencies. I run the following command on terminal being on same directory:

```json
docker build -t fastapi-cat_v_dog:1.0 .
```
This command built a Docker image tagged as fastapi-cat_v_dog with version 1.0. The . at the end of the command specifies the current directory as the build context.

**5. Running the Docker Container:** I tested the Docker image locally using the docker run command to ensure that your FastAPI service runs within a container. I have mapped port 80 of the container to port 8000 on the host machine to access the service and verified that the service was accessible through a web browser or API client on the host. I run the following command on terminal being on same directory:

```json
docker run -p 8000:80 fastapi-cat_v_dog:1.0
```
**6. Docker Compose:** Docker Compose is a tool for defining and running multi-container applications. It is particularly useful when the application consists of multiple services or containers that need to work together. It provides a structured and reproducible way to define and run application's components. I have prepared a Docker Compose configuration (docker-compose.yml) to simplify container management. The Docker Compose file included the necessary instructions to build and run the FastAPI service. We don't necessarily need to add volumes unless your application requires specific volume mounts. We would use volumes if we want to share files or directories between host machine and the Docker container. For example, if the FastAPI application needs to access a configuration file or a database stored on host machine, one can use volumes to make these files accessible to the container. To add a volume to a service, one can include a volumes section under the service definition in your docker-compose.yml file.

### Summary insights

**Docker**

Docker is a versatile platform designed for simplifying the development, packaging, and deployment of applications. It operates using containers, which are lightweight, isolated environments that encapsulate an application and its dependencies. Containers ensure consistency across various environments, making it easier to ship and run applications.

Within Docker, applications are packaged into images, read-only templates that define the application environment and code. These images can be deployed into containers, allowing multiple instances of an application to run concurrently from the same image.

Developers create Docker images using Dockerfiles, which are scripts specifying the base image, application code, dependencies, and runtime configurations. Docker Hub, a cloud-based registry, serves as a repository for sharing and distributing Docker images among developers, simplifying the deployment process.

**Docker run vs compose**

Docker run is used to run a single container based on an image. It's a simple and straightforward way to start a single container instance. For example, if you want to run a specific container with custom settings or options, you can use docker run. When you run docker run, it starts a new container each time unless you specify additional options to control container naming or reuse.

Docker-compose, on the other hand, is a tool for defining and running multi-container Docker applications. It uses a YAML file (docker-compose.yml) to define the services, networks, and volumes required for your application. You can define multiple services in the docker-compose.yml file, each with its own configuration. Running docker-compose up creates and starts all the containers defined in the docker-compose.yml file as a group. This is especially useful for complex applications with multiple interconnected containers, such as microservices architectures.

**WSL (Windows Subsystem for Linux)** 

It is a compatibility layer for running Linux applications and tools on Windows. It allows Windows users to work with a Linux distribution alongside their existing Windows environment. WSL supports multiple Linux distributions, such as Ubuntu and Debian, which can be installed either through the Microsoft Store or via command-line tools. It seamlessly integrates Windows and Linux file systems, offering easy access and manipulation of files from both sides. This environment is especially popular among developers for running Linux-based development tools, web servers, and containers. With the introduction of WSL 2, which includes a full Linux kernel, the performance and compatibility of Linux-based tools like Docker have significantly improved, making it an ideal choice for containerized development on Windows machines.

## Conclusion

The Dockerized FastAPI-based image classification service offers a practical solution for deploying machine learning models in a user-friendly and consistent manner. It simplifies the deployment process and enhances portability across various platforms. This containerization approach facilitates rapid deployment and scaling of the service while maintaining reliability and stability. 
