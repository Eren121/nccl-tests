.PHONY: image

DEVS = $(wildcard /dev/infiniband/uverbs*)
DEV_OPT = $(foreach f,$(DEVS),--device=$(f))

image:
	docker build -t my_nccl .

.PHONY: run
run:
	docker run --net=host --gpus=all -it --rm  -w $(HOME) \
                $(DEV_OPT) --device=/dev/infiniband/rdma_cm \
                --mount type=bind,src=$(HOME),dst=$(HOME) my_nccl
