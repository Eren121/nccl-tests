.PHONY: image

IMAGE = my-nccl

DEVS = $(wildcard /dev/infiniband/uverbs*)
DEV_OPT = $(foreach f,$(DEVS),--device=$(f)) --device=/dev/infiniband/rdma_cm

BASIC_OPT = --net=host --privileged --gpus=all -it --rm -w $(HOME) --ulimit memlock=819200000:819200000
HOSTS_OPT = $(shell awk '{print "--add-host="$$1":"$$2}' $(HOME)/hosts.txt)
MOUNT_OPT = --mount type=bind,src=$(HOME),dst=$(HOME)

image_hpe:
	docker build \
		--build-arg NCCL_IB_HCA='=mlx5_2:1' \
		--build-arg NCCL_IB_GID_INDEX=3 \
		-t $(IMAGE) .

image_smartedge:
	docker build \
		--build-arg NCCL_IB_HCA='=mlx5_0:1' \
		--build-arg NCCL_IB_GID_INDEX=3 \
		-t $(IMAGE) .

.PHONY: run
run:
	docker run $(BASIC_OPT) $(HOSTS_OPT) $(DEV_OPT) $(MOUNT_OPT) \
		$(IMAGE)
