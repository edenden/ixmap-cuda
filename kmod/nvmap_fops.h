#ifndef _NVMAP_FOPS_H
#define _NVMAP_FOPS_H

#define MISCDEV_NAME_SIZE	32

#define NVMAP_MAP		_IOW('U', 210, int)
struct nvmap_map_req {
	unsigned long		addr_virtual;
	unsigned long		addr_dma;
	unsigned long		size;
};

#define NVMAP_UNMAP		_IOW('U', 211, int)
struct nvmap_unmap_req {
	unsigned long		addr_dma;
};

int nvmap_miscdev_register(struct nvmap_info *info);
void nvmap_miscdev_deregister(struct nvmap_info *info);

#endif /* _NVMAP_FOPS_H */
