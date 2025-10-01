.PHONY: image

DEVS = $(wildcard /dev/infiniband/uverbs*)
DEV_OPT = $(foreach f,$(DEVS),--device=$(f)) --device=/dev/infiniband/rdma_cm

BASIC_OPT = --net=host --privileged --gpus=all -it --rm -w $(HOME) --ulimit memlock=819200000:819200000
HOSTS_OPT = $(shell awk '{print "--add-host="$$1":"$$2}' hosts.txt)
MOUNT_OPT = --mount type=bind,src=$(HOME),dst=$(HOME)

image:
	docker build -t my_nccl .

.PHONY: run
run:
	docker run $(BASIC_OPT) $(HOSTS_OPT) $(DEV_OPT) $(MOUNT_OPT) \
		my_nccl
