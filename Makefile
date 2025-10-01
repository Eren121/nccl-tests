.PHONY: image

DEVS = $(wildcard /dev/infiniband/uverbs*)
DEV_OPT = $(foreach f,$(DEVS),--device=$(f))

image:
	docker build -t my_nccl .

.PHONY: run
run:
	docker run --net=host --gpus=all -it --rm  -w $(HOME) \
		--add-host nccl-hpe:192.168.120.1 \
		--add-host nccl-smartedge:192.168.120.2 \
			$(DEV_OPT) --device=/dev/infiniband/rdma_cm \
                --mount type=bind,src=$(HOME),dst=$(HOME) my_nccl
