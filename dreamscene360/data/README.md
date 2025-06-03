Directory of output data that mount on docker container

```bash
docker run --gpus all -it \
  -v ~/dreamscene360_docker/data:/workspace/DreamScene360/data \
  -v ~/dreamscene360_docker/output:/workspace/DreamScene360/output \
  -v ~/dreamscene360_docker/checkpoints:/workspace/DreamScene360/pre_checkpoints \
```
