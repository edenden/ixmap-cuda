#define _GNU_SOURCE
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <endian.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <net/ethernet.h>
#include <signal.h>
#include <pthread.h>

#include <driver_functions.h>
#include <driver_types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "ixmap.h"
#include "memory.h"

static int ixmap_dma_unmap(struct ixmap_handle *ih, unsigned long addr_dma);
static int ixmap_dma_map(struct ixmap_handle *ih, void *addr_virt,
	unsigned long *addr_dma, unsigned long size);
static int ixmap_dma_unmap_direct(struct nvmap_handle *nh, unsigned long addr_dma);
static int ixmap_dma_map_direct(struct nvmap_handle *nh, void *addr_virt,
	unsigned long *addr_dma, unsigned long size);

struct nvmap_handle *nvmap_open(void)
{
	struct nvmap_handle *nh;
	char filename[FILENAME_SIZE];

	nh = malloc(sizeof(struct nvmap_handle));
	if (!nh)
		goto err_alloc_nh;
	memset(nh, 0, sizeof(struct nvmap_handle));

	snprintf(filename, sizeof(filename), "/dev/nvmap");
	nh->fd = open(filename, O_RDWR);
	if (nh->fd < 0)
		goto err_open;

	return nh;

err_open:
	free(nh);
err_alloc_nh:
	return NULL;
}

void nvmap_close(struct nvmap_handle *nh)
{
	close(nh->fd);
	free(nh);

	return;
}

struct ixmap_desc *ixmap_desc_alloc_cuda(struct ixmap_handle **ih_list,
	int ih_num, int queue_index)
{
	struct ixmap_desc *desc;
	unsigned long size_tx_desc, size_rx_desc, size_mem;
	void *addr_virt, *addr_mem;
	int i, ret;
	int desc_assigned = 0;
	cudaError_t ret_cuda;

	desc = malloc(sizeof(struct ixmap_desc));
	if(!desc)
		goto err_alloc_desc;

	/*
	 * XXX: We don't support NUMA-aware memory allocation in userspace.
	 * To support, mbind() or set_mempolicy() will be useful.
	 */
	desc->addr_virt = mmap(NULL, SIZE_1GB, PROT_READ | PROT_WRITE,
		MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, 0, 0);
	if(desc->addr_virt == MAP_FAILED){
		goto err_mmap;
	}

	ret_cuda = cudaHostRegister(desc->addr_virt, SIZE_256MB, cudaHostRegisterMapped);
	if(ret_cuda != cudaSuccess){
		goto err_cuda;
	}

	addr_virt = desc->addr_virt;

	for(i = 0; i < ih_num; i++, desc_assigned++){
		int *slot_index;
		struct ixmap_handle *ih;
		unsigned long addr_dma;

		ih = ih_list[i];

		size_rx_desc = sizeof(union ixmap_adv_rx_desc) * ih->num_rx_desc;
		size_rx_desc = ALIGN(size_rx_desc, 128); /* needs 128-byte alignment */
		size_tx_desc = sizeof(union ixmap_adv_tx_desc) * ih->num_tx_desc;
		size_tx_desc = ALIGN(size_tx_desc, 128); /* needs 128-byte alignment */

		/* Rx descripter ring allocation */
		ret = ixmap_dma_map(ih, addr_virt, &addr_dma, size_rx_desc);
		if(ret < 0){
			goto err_rx_dma_map;
		}

		ih->rx_ring[queue_index].addr_dma = addr_dma;
		ih->rx_ring[queue_index].addr_virt = addr_virt;

		slot_index = malloc(sizeof(int32_t) * ih->num_rx_desc);
		if(!slot_index){
			goto err_rx_assign;
		}

		ih->rx_ring[queue_index].next_to_use = 0;
		ih->rx_ring[queue_index].next_to_clean = 0;
		ih->rx_ring[queue_index].slot_index = slot_index;

		addr_virt += size_rx_desc;

		/* Tx descripter ring allocation */
		ret = ixmap_dma_map(ih, addr_virt, &addr_dma, size_tx_desc);
		if(ret < 0){
			goto err_tx_dma_map;
		}

		ih->tx_ring[queue_index].addr_dma = addr_dma;
		ih->tx_ring[queue_index].addr_virt = addr_virt;

		slot_index = malloc(sizeof(int32_t) * ih->num_tx_desc);
		if(!slot_index){
			goto err_tx_assign;
		}

		ih->tx_ring[queue_index].next_to_use = 0;
		ih->tx_ring[queue_index].next_to_clean = 0;
		ih->tx_ring[queue_index].slot_index = slot_index;

		addr_virt += size_rx_desc;

		continue;

err_tx_assign:
		ixmap_dma_unmap(ih, ih->tx_ring[queue_index].addr_dma);
err_tx_dma_map:
		free(ih->rx_ring[queue_index].slot_index);
err_rx_assign:
		ixmap_dma_unmap(ih, ih->rx_ring[queue_index].addr_dma);
err_rx_dma_map:
		goto err_desc_assign;
	}

	addr_mem = (void *)ALIGN((unsigned long)addr_virt, L1_CACHE_BYTES);
	size_mem = SIZE_1GB - (addr_mem - desc->addr_virt);
	desc->node = ixmap_mem_init(addr_mem, size_mem);
	if(!desc->node)
		goto err_mem_init;

	return desc;

err_mem_init:
err_desc_assign:
	for(i = 0; i < desc_assigned; i++){
		struct ixmap_handle *ih;

		ih = ih_list[i];
		free(ih->tx_ring[queue_index].slot_index);
		ixmap_dma_unmap(ih, ih->tx_ring[queue_index].addr_dma);
		free(ih->rx_ring[queue_index].slot_index);
		ixmap_dma_unmap(ih, ih->rx_ring[queue_index].addr_dma);
	}
	cudaHostUnregister(desc->addr_virt);
err_cuda:
	munmap(desc->addr_virt, SIZE_1GB);
err_mmap:
	free(desc);
err_alloc_desc:
	return NULL;
}

void ixmap_desc_release_cuda(struct ixmap_handle **ih_list, int ih_num,
	int queue_index, struct ixmap_desc *desc)
{
	int i;

	ixmap_mem_destroy(desc->node);

	for(i = 0; i < ih_num; i++){
		struct ixmap_handle *ih;

		ih = ih_list[i];
		free(ih->tx_ring[queue_index].slot_index);
		ixmap_dma_unmap(ih, ih->tx_ring[queue_index].addr_dma);
		free(ih->rx_ring[queue_index].slot_index);
		ixmap_dma_unmap(ih, ih->rx_ring[queue_index].addr_dma);
	}

	cudaHostUnregister(desc->addr_virt);
	munmap(desc->addr_virt, SIZE_1GB);
	free(desc);
	return;
}

struct ixmap_buf *ixmap_buf_alloc_cuda(struct ixmap_handle **ih_list,
	int ih_num, uint32_t count, uint32_t buf_size)
{
	struct ixmap_buf *buf;
	void	*addr_virt;
	unsigned long addr_dma, size;
	int *slots;
	int ret, i, mapped_ports = 0;
	cudaError_t ret_cuda;

	buf = malloc(sizeof(struct ixmap_buf));
	if(!buf)
		goto err_alloc_buf;

	buf->addr_dma = malloc(sizeof(unsigned long) * ih_num);
	if(!buf->addr_dma)
		goto err_alloc_buf_addr_dma;

	/*
	 * XXX: Should we add buffer padding for memory interleaving?
	 * DPDK does so in rte_mempool.c/optimize_object_size().
	 */
	size = buf_size * (ih_num * count);

	/*
	 * XXX: We don't support NUMA-aware memory allocation in userspace.
	 * To support, mbind() or set_mempolicy() will be useful.
	 */
	addr_virt = mmap(NULL, size, PROT_READ | PROT_WRITE,
		MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, 0, 0);
	if(addr_virt == MAP_FAILED)
		goto err_mmap;

	ret_cuda = cudaHostRegister(addr_virt, size, cudaHostRegisterMapped);
	if(ret_cuda != cudaSuccess){
		goto err_cuda;
	}

	for(i = 0; i < ih_num; i++, mapped_ports++){
		ret = ixmap_dma_map(ih_list[i], addr_virt, &addr_dma, size);
		if(ret < 0)
			goto err_ixmap_dma_map;

		buf->addr_dma[i] = addr_dma;
	}

	slots = malloc(sizeof(int32_t) * (count * ih_num));
	if(!slots)
		goto err_alloc_slots;

	buf->addr_virt = addr_virt;
	buf->buf_size = buf_size;
	buf->count = count;
	buf->slots = slots;

	for(i = 0; i < buf->count * ih_num; i++){
		buf->slots[i] = 0;
	}

	return buf;

err_alloc_slots:
err_ixmap_dma_map:
	for(i = 0; i < mapped_ports; i++){
		ixmap_dma_unmap(ih_list[i], buf->addr_dma[i]);
	}
	cudaHostUnregister(addr_virt);
err_cuda:
	munmap(addr_virt, size);
err_mmap:
	free(buf->addr_dma);
err_alloc_buf_addr_dma:
	free(buf);
err_alloc_buf:
	return NULL;
}

void ixmap_buf_release_cuda(struct ixmap_buf *buf,
	struct ixmap_handle **ih_list, int ih_num)
{
	int i, ret;
	unsigned long size;

	free(buf->slots);

	for(i = 0; i < ih_num; i++){
		ret = ixmap_dma_unmap(ih_list[i], buf->addr_dma[i]);
		if(ret < 0)
			perror("failed to unmap buf");
	}

	cudaHostUnregister(buf->addr_virt);

	size = buf->buf_size * buf->count;
	munmap(buf->addr_virt, size);
	free(buf->addr_dma);
	free(buf);

	return;
}

struct ixmap_buf *ixmap_buf_alloc_cuda_direct(struct nvmap_handle *nh,
	int ih_num, uint32_t count, uint32_t buf_size)
{
	struct ixmap_buf *buf;
	void	*addr_virt;
	unsigned long addr_dma, size;
	int *slots;
	int ret, i;
	unsigned int cuda_flag = 1;
	cudaError_t cuda_ret;
	CUresult cuda_stat;

	buf = malloc(sizeof(struct ixmap_buf));
	if(!buf)
		goto err_alloc_buf;

	buf->addr_dma = malloc(sizeof(unsigned long) * ih_num);
	if(!buf->addr_dma)
		goto err_alloc_buf_addr_dma;

	/*
	 * XXX: Should we add buffer padding for memory interleaving?
	 * DPDK does so in rte_mempool.c/optimize_object_size().
	 */
	size = buf_size * (ih_num * count);

	cuda_ret = cudaMalloc(&addr_virt, size);
	if(cuda_ret != cudaSuccess)
		goto err_cuda_malloc;

	cuda_stat = cuPointerSetAttribute(&cuda_flag,
		CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)addr_virt);
	if(cuda_stat != CUDA_SUCCESS)
		goto err_cuda_setattr;

	ret = ixmap_dma_map_direct(nh, addr_virt, &addr_dma, size);
	if(ret < 0)
		goto err_ixmap_dma_map;

	for(i = 0; i < ih_num; i++){
		buf->addr_dma[i] = addr_dma;
	}

	slots = malloc(sizeof(int32_t) * (count * ih_num));
	if(!slots)
		goto err_alloc_slots;

	buf->addr_virt = addr_virt;
	buf->buf_size = buf_size;
	buf->count = count;
	buf->slots = slots;

	for(i = 0; i < buf->count * ih_num; i++){
		buf->slots[i] = 0;
	}

	return buf;

err_alloc_slots:
	ixmap_dma_unmap_direct(nh, addr_dma);
err_ixmap_dma_map:
err_cuda_setattr:
	cudaFree(addr_virt);
err_cuda_malloc:
	free(buf->addr_dma);
err_alloc_buf_addr_dma:
	free(buf);
err_alloc_buf:
	return NULL;
}

void ixmap_buf_release_cuda_direct(struct ixmap_buf *buf,
	struct nvmap_handle *nh, int ih_num)
{
	int i, ret;

	free(buf->slots);

	for(i = 0; i < ih_num; i++){
		if(i + 1 < ih_num
		&& buf->addr_dma[i] != buf->addr_dma[i + 1]){
			/* If we faced with unexpected situation */
			perror("DMA addr differs between ixgbe ports");
			return;
		}
	}

	ret = ixmap_dma_unmap_direct(nh, buf->addr_dma[0]);
	if(ret < 0)
		perror("failed to unmap buf");

	cudaFree(buf->addr_virt);
	free(buf->addr_dma);
	free(buf);

	return;
}

static int ixmap_dma_map(struct ixmap_handle *ih, void *addr_virt,
	unsigned long *addr_dma, unsigned long size)
{
	struct ixmap_map_req req_map;

	req_map.addr_virt = (unsigned long)addr_virt;
	req_map.addr_dma = 0;
	req_map.size = size;
	req_map.cache = IXGBE_DMA_CACHE_DISABLE;

	if(ioctl(ih->fd, IXMAP_MAP, (unsigned long)&req_map) < 0)
		return -1;

	*addr_dma = req_map.addr_dma;
	return 0;
}

static int ixmap_dma_unmap(struct ixmap_handle *ih, unsigned long addr_dma)
{
	struct ixmap_unmap_req req_unmap;

	req_unmap.addr_dma = addr_dma;

	if(ioctl(ih->fd, IXMAP_UNMAP, (unsigned long)&req_unmap) < 0)
		return -1;

	return 0;
}

static int ixmap_dma_map_direct(struct nvmap_handle *nh, void *addr_virt,
	unsigned long *addr_dma, unsigned long size)
{
	struct nvmap_map_req req_map;

	req_map.addr_virt = (unsigned long)addr_virt;
	req_map.addr_dma = 0;
	req_map.size = size;

	if(ioctl(nh->fd, NVMAP_MAP, (unsigned long)&req_map) < 0)
		return -1;

	*addr_dma = req_map.addr_dma;
	return 0;
}

static int ixmap_dma_unmap_direct(struct nvmap_handle *nh, unsigned long addr_dma)
{
	struct nvmap_unmap_req req_unmap;

	req_unmap.addr_dma = addr_dma;

	if(ioctl(nh->fd, NVMAP_UNMAP, (unsigned long)&req_unmap) < 0)
		return -1;

	return 0;
}
