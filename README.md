# Human-robot interaction pipeline

Human-robot interaction pipeline for humanoid robot in a simple household environment.
Implemented in ROS, uses docker containers.
Motor control is designed for the Tiago robot but can be adapted to other robots.

## Installation

Simply clone the repository.
Then, download the data (weights for yolov8 and image datasets) using the following command:

```bash
bash download_data.sh
```

## Configuration

The pipeline is configured using environment variables in the `.env` file.

## Usage

Run docker compose:

```bash
docker compose up
```

## Contents

- `docker-compose.yml`: docker compose file for the pipeline.  
- `[compose]`: directory for docker files and helper scripts.  
- `[lib]`: external repositories.  
- `[data]`: data (weights) for the pipeline.  
- `[config]`: configuration files for the pipeline.  
- `[src]`: (additional) source code for the pipeline; used by the containers.  
