.PHONY: image

DEVS = $(wildcard /dev/infiniband/uverbs*)
DEV_OPT = $(foreach f,$(DEVS),--device=$(f))

BASIC_OPT = --net=host --gpus=all -it --rm -w $(HOME) 
HOSTS_OPT = $(shell awk '{print "--add-host="$$1":"$$2}' hosts.txt)
MOUNT_OPT = --mount type=bind,src=$(HOME),dst=$(HOME)

image:
	docker build -t my_nccl .

.PHONY: run
run:
	docker run $(BASIC_OPT) $(HOSTS_OPT) \
		$(DEV_OPT) --device=/dev/infiniband/rdma_cm \
		$(MOUNT_OPT) my_nccl
