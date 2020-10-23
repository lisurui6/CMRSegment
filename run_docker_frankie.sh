nvidia-docker run -it -e CUDA_VISIBLE_DEVICES=0,1 \
-v /mnt/storage/home/suruli/cardiac:/mnt/storage/home/suruli/cardiac \
-v /mnt/storage/home/suruli/suruli/projects/CMRSegment:/workspace/CMRSegment \
docker.io/lisurui6/cmr-segment:latest
