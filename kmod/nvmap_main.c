#include <linux/interrupt.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/init.h>
#include <linux/wait.h>
#include <linux/miscdevice.h>
#include <linux/ioport.h>
#include <linux/pci.h>
#include <linux/sched.h>
#include <linux/semaphore.h>
#include <asm/io.h>

#include "nvmap_main.h"
#include "nvmap_fops.h"
#include "nvmap_dma.h"

static struct nvmap_info *nvmap_info_alloc(void);
static void nvmap_info_dealloc(struct nvmap_info *info);

const char nvmap_driver_name[]	= "nvmap";
const char nvmap_driver_desc[]	= "Direct access to ixgbe device register";
const char nvmap_driver_ver[]	= "1.0";
const char *nvmap_copyright[]	= {
	"Copyright (c) 2015 by Yukito Ueno <eden@sfc.wide.ad.jp>.",
};
struct nvmap_info *info;

static struct nvmap_info *nvmap_info_alloc(void)
{
	struct nvmap_info *info;

	info = kzalloc(sizeof(struct nvmap_info), GFP_KERNEL);
	if (!info){
		return NULL;
	}

	atomic_set(&info->refcount, 1);
	sema_init(&info->sem, 1);

	INIT_LIST_HEAD(&info->areas);

	return info;
}

static void nvmap_info_dealloc(struct nvmap_info *info)
{
	kfree(info);
	return;
}

int nvmap_info_inuse(struct nvmap_info *info)
{
	unsigned ref = atomic_read(&info->refcount);
	if (ref == 1)
		return 0;
	return 1;
}

void nvmap_info_get(struct nvmap_info *info)
{
	atomic_inc(&info->refcount);
	return;
}

void nvmap_info_put(struct nvmap_info *info)
{
	atomic_dec(&info->refcount);
	return;
}

static int __init nvmap_module_init(void)
{
	int err;

	pr_info("%s - version %s\n",
		nvmap_driver_desc, nvmap_driver_ver);
	pr_info("%s\n", nvmap_copyright[0]);

	info = nvmap_info_alloc();
	if(info == NULL){
		err = -ENOMEM;
		goto err_alloc;
	}

        err = nvmap_miscdev_register(info);
	if(err < 0)
		goto err_miscdev_register;

	return 0;

err_miscdev_register:
	nvmap_info_dealloc(info);
err_alloc:
	return err;
}

static void __exit nvmap_module_exit(void)
{
	nvmap_miscdev_deregister(info);

	down(&info->sem);
	nvmap_dma_unmap_all(info);
	up(&info->sem);

	nvmap_info_dealloc(info);

	return;
}

module_init(nvmap_module_init);
module_exit(nvmap_module_exit);
MODULE_AUTHOR("Yukito Ueno <eden@sfc.wide.ad.jp>");
MODULE_DESCRIPTION("Direct access to ixgbe device register");
MODULE_LICENSE("GPL");
MODULE_VERSION("1.0");
