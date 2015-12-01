#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/ioport.h>
#include <linux/init.h>
#include <linux/poll.h>
#include <linux/proc_fs.h>
#include <linux/spinlock.h>
#include <linux/sysctl.h>
#include <linux/wait.h>
#include <linux/miscdevice.h>
#include <linux/ioport.h>
#include <linux/pci.h>
#include <linux/file.h>
#include <linux/scatterlist.h>
#include <linux/sched.h>
#include <asm/io.h>
#include <linux/dma_remapping.h>
#include <nv-p2p.h>

#include "nvmap_main.h"
#include "nvmap_dma.h"

static struct nvmap_dma_area *nvmap_dma_area_lookup(struct nvmap_info *info,
	unsigned long addr_dma);
static struct list_head *nvmap_dma_area_whereto(struct nvmap_info *info,
	unsigned long addr_dma, unsigned long size);
static void nvmap_dma_area_free(struct nvmap_info *info,
	struct nvmap_dma_area *area);

void nvmap_dma_callback(void *data)
{
	struct nvmap_cb_data *cb_data;
	struct nvmap_info *info;
	dma_addr_t addr_dma;
	struct nvmap_dma_area *area;
	struct nvidia_p2p_page_table *page_table;

	cb_data = (struct nvmap_cb_data *)data;
	if(!cb_data->info || !cb_data->addr_dma){
		pr_err("ERR: invalid GPU memory region\n");
		return;
	}
	info = cb_data->info;
	addr_dma = cb_data->addr_dma;

	down(&info->sem);

	area = nvmap_dma_area_lookup(info, addr_dma);
	if (!area){
		pr_err("ERR: failed to release GPU memory region\n");
		goto out;
	}

	list_del(&area->list);
	page_table = area->page_table;
	nvidia_p2p_free_page_table(page_table);

	kfree(area->cb_data);
	kfree(area);

out:
	up(&info->sem);
	return;

}

dma_addr_t nvmap_dma_map(struct nvmap_info *info,
	unsigned long addr_virtual, unsigned long size)
{
	struct nvmap_dma_area *area;
	struct list_head *where;
	struct nvidia_p2p_page_table *page_table;
	unsigned long user_start, user_end, user_length;
	unsigned int i, page_size;
	int ret;
	dma_addr_t addr_dma;
	struct nvmap_cb_data *cb_data;

	user_start = addr_virtual & GPU_PAGE_MASK;
	user_end = GPU_PAGE_ALIGN(addr_virtual + size);
	user_length = user_end - user_start;

	cb_data = kzalloc(sizeof(struct nvmap_cb_data), GFP_KERNEL);
	if(!cb_data)
		goto err_alloc_cb_data;

	pr_info("get pages start = %p, length = %lu\n", (void *)user_start, user_length);
	ret = nvidia_p2p_get_pages(0, 0, user_start, user_length, &page_table,
		nvmap_dma_callback, cb_data);
	if(ret < 0){
		pr_err("ERR: failed to get pages, ret = %d\n", ret);
		goto err_get_user_pages;
	}

	switch(page_table->page_size){
	case NVIDIA_P2P_PAGE_SIZE_4KB:
		page_size = 4 * 1024;
		break;
	case NVIDIA_P2P_PAGE_SIZE_64KB:
		page_size = 64 * 1024;
		break;
	case NVIDIA_P2P_PAGE_SIZE_128KB:
		page_size = 128 * 1024;
		break;
	default:
		goto err_invalid_page_size;
	}

	for(i = 0; i < page_table->entries; i++){
		if(i + 1 < page_table->entries &&
		page_table->pages[i]->physical_address + page_size
		!= page_table->pages[i + 1]->physical_address){
			pr_err("ERR: non-contiguous dma area\n");
			goto err_get_user_pages_not_contiguous;
		}
	}

	addr_dma = page_table->pages[0]->physical_address;
        where = nvmap_dma_area_whereto(info, addr_dma, user_length);
        if (!where)
		goto err_area_whereto;

	area = kzalloc(sizeof(struct nvmap_dma_area), GFP_KERNEL);
	if (!area)
		goto err_alloc_area;

	area->size = user_length;
	area->addr_dma = addr_dma;
	area->user_start = user_start;
	area->page_table = page_table;
	area->cb_data = cb_data;
	list_add(&area->list, where);

	cb_data->info = info;
	cb_data->addr_dma = addr_dma;
	
	return addr_dma;

err_alloc_area:
err_area_whereto:
err_get_user_pages_not_contiguous:
err_invalid_page_size:
	nvidia_p2p_put_pages(0, 0, user_start, page_table);
err_get_user_pages:
	kfree(cb_data);
err_alloc_cb_data:
	return 0;
}

int nvmap_dma_unmap(struct nvmap_info *info, unsigned long addr_dma)
{
	struct nvmap_dma_area *area;

	area = nvmap_dma_area_lookup(info, addr_dma);
	if (!area)
		return -ENOENT;

	list_del(&area->list);
	nvmap_dma_area_free(info, area);

	return 0;
}

void nvmap_dma_unmap_all(struct nvmap_info *info)
{
	struct nvmap_dma_area *area, *temp;

	list_for_each_entry_safe(area, temp, &info->areas, list) {
		list_del(&area->list);
		nvmap_dma_area_free(info, area);
	}

	return;
}

static struct nvmap_dma_area *nvmap_dma_area_lookup(struct nvmap_info *info,
	unsigned long addr_dma)
{
	struct nvmap_dma_area *area;

	list_for_each_entry(area, &info->areas, list) {
		if (area->addr_dma == addr_dma)
			return area;
	}

	return NULL;
}

static struct list_head *nvmap_dma_area_whereto(struct nvmap_info *info,
	unsigned long addr_dma, unsigned long size)
{
	unsigned long start_new, end_new;
	unsigned long start_area, end_area;
	struct nvmap_dma_area *area;
	struct list_head *last;

	pr_info("add area: start = %p end = %p size = %lu\n",
		(void *)addr_dma, (void *)(addr_dma + size), size);

	start_new = addr_dma;
	end_new   = start_new + size;
	last  = &info->areas;

	list_for_each_entry(area, &info->areas, list) {
		start_area = area->addr_dma;
		end_area   = start_area + area->size;

		/* Since the list is sorted we know at this point that
		 * new area goes before this one. */
		if (end_new <= start_area)
			break;

		last = &area->list;

		if ((start_new >= start_area && start_new < end_area) ||
				(end_new > start_area && end_new <= end_area)) {
			/* Found overlap. Set start to the end of the current
			 * area and keep looking. */
			last = NULL;
			break;
		}
	}

	return last;
}

static void nvmap_dma_area_free(struct nvmap_info *info,
	struct nvmap_dma_area *area)
{
	unsigned long user_start;
	struct nvidia_p2p_page_table *page_table;

	pr_info("delete area: start = %p end = %p size = %lu\n",
		(void *)area->addr_dma, (void *)(area->addr_dma + area->size), area->size);

	user_start = area->user_start;
	page_table = area->page_table;

	nvidia_p2p_put_pages(0, 0, user_start, page_table);

	kfree(area->cb_data);
	kfree(area);
	return;
}

