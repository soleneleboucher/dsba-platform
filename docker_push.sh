#!/bin/bash

DOCKER_USERNAME="solene25"
REPO_NAME="dsba-platform-adclick"
TAG="v1004_2"

# Build the Docker image 
docker build -t $DOCKER_USERNAME/$REPO_NAME:$TAG .

docker login

# Push the image to Docker Hub 
docker push $DOCKER_USERNAME/$REPO_NAME:$TAG