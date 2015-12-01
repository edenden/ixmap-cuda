#ifndef _NVMAP_MAIN_H
#define _NVMAP_MAIN_H

#include <linux/if_ether.h>
#include <linux/types.h>
#include <asm/page.h>

/* common prefix used by pr_<> macros */
#undef pr_fmt
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

struct nvmap_info {
	struct list_head	areas;
	struct miscdevice	miscdev;
	struct semaphore	sem;
	atomic_t		refcount;
};

int nvmap_info_inuse(struct nvmap_info *info);
void nvmap_info_get(struct nvmap_info *info);
void nvmap_info_put(struct nvmap_info *info);

#endif /* _NVMAP_MAIN_H */

