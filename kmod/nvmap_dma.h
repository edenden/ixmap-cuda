#ifndef _NVMAP_DMA_H
#define _NVMAP_DMA_H

#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (_AC(1,UL) << GPU_PAGE_SHIFT)
#define GPU_PAGE_MASK (~(GPU_PAGE_SIZE-1))
#define GPU_PAGE_ALIGN(addr) ALIGN(addr, GPU_PAGE_SIZE)

struct nvmap_cb_data {
	struct nvmap_info	*info;
	dma_addr_t		addr_dma;
};

struct nvmap_dma_area {
	struct list_head	list;
	unsigned long		size;
	dma_addr_t		addr_dma;

	unsigned long		user_start;
	struct nvidia_p2p_page_table
				*page_table;
	struct nvmap_cb_data	*cb_data;
};

void nvmap_dma_callback(void *data);
dma_addr_t nvmap_dma_map(struct nvmap_info *info,
	unsigned long addr_virtual, unsigned long size);
int nvmap_dma_unmap(struct nvmap_info *info, unsigned long addr_dma);
void nvmap_dma_unmap_all(struct nvmap_info *info);

#endif /* _NVMAP_DMA_H */
