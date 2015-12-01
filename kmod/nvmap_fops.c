#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/ioport.h>
#include <linux/poll.h>
#include <linux/sysctl.h>
#include <linux/wait.h>
#include <linux/miscdevice.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/pci.h>

#include "nvmap_main.h"
#include "nvmap_fops.h"
#include "nvmap_dma.h"

static int nvmap_cmd_map(struct nvmap_info *info,
	void __user *argp);
static int nvmap_cmd_unmap(struct nvmap_info *info,
	void __user *argp);
static ssize_t nvmap_read(struct file * file, char __user * buf,
	size_t count, loff_t *pos);
static ssize_t nvmap_write(struct file * file, const char __user * buf,
	size_t count, loff_t *pos);
static int nvmap_open(struct inode *inode, struct file * file);
static int nvmap_close(struct inode *inode, struct file *file);
static long nvmap_ioctl(struct file *file, unsigned int cmd,
	unsigned long arg);

static struct file_operations nvmap_fops = {
	.owner		= THIS_MODULE,
	.llseek		= no_llseek,
	.read		= nvmap_read,
	.write		= nvmap_write,
	.open		= nvmap_open,
	.release	= nvmap_close,
	.unlocked_ioctl	= nvmap_ioctl,
};

int nvmap_miscdev_register(struct nvmap_info *info)
{
	char *miscdev_name;
	int err;

	miscdev_name = kmalloc(MISCDEV_NAME_SIZE, GFP_KERNEL);
	if(!miscdev_name){
		goto err_alloc_name;
	}
	snprintf(miscdev_name, MISCDEV_NAME_SIZE, "nvmap");

	info->miscdev.minor = MISC_DYNAMIC_MINOR;
	info->miscdev.name = miscdev_name;
	info->miscdev.fops = &nvmap_fops;
	err = misc_register(&info->miscdev);
	if (err) {
		pr_err("failed to register misc device\n");
		goto err_misc_register;
	}

	pr_info("misc device registered as %s\n", info->miscdev.name);
	return 0;

err_misc_register:
        kfree(info->miscdev.name);
err_alloc_name:
	return -1;
}

void nvmap_miscdev_deregister(struct nvmap_info *info)
{
	misc_deregister(&info->miscdev);

	pr_info("misc device %s unregistered\n", info->miscdev.name);
	kfree(info->miscdev.name);

	return;
}

static int nvmap_cmd_map(struct nvmap_info *info,
	void __user *argp)
{
	struct nvmap_map_req req;
	unsigned long addr_dma;

	if (copy_from_user(&req, argp, sizeof(req)))
		return -EFAULT;

	if (!req.size)
		return -EINVAL;

	addr_dma = nvmap_dma_map(info, req.addr_virtual,
		req.size);
	if(!addr_dma)
		return -EFAULT;

	req.addr_dma = addr_dma;

	if (copy_to_user(argp, &req, sizeof(req))) {
		nvmap_dma_unmap(info, req.addr_dma);
		return -EFAULT;
	}

	return 0;
}

static int nvmap_cmd_unmap(struct nvmap_info *info,
	void __user *argp)
{
	struct nvmap_unmap_req req;
	int ret;

	if (copy_from_user(&req, argp, sizeof(req)))
		return -EFAULT;

	ret = nvmap_dma_unmap(info, req.addr_dma);
	if(ret != 0)
		return ret;

	return 0;
}

static ssize_t nvmap_read(struct file * file, char __user * buf,
	size_t count, loff_t *pos)
{
	return 0;
}

static ssize_t nvmap_write(struct file * file, const char __user * buf,
	size_t count, loff_t *pos)
{
	return 0;
}

static int nvmap_open(struct inode *inode, struct file * file)
{
	struct nvmap_info *info;
	struct miscdevice *miscdev = file->private_data;
	int err;

	info = container_of(miscdev, struct nvmap_info, miscdev);
	pr_info("open req miscdev\n");

	down(&info->sem);

	// Only one process is alowed to open
	if (nvmap_info_inuse(info)) {
		err = -EBUSY;
		goto out;
	}

	nvmap_info_get(info);
	file->private_data = info;
	err = 0;

out:
	up(&info->sem);
	return err;
}

static int nvmap_close(struct inode *inode, struct file *file)
{
	struct nvmap_info *info = file->private_data;
	if (!info)
		return 0;

	pr_info("close req miscdev\n");

	down(&info->sem);

	if (nvmap_info_inuse(info)) {
		nvmap_info_put(info);
        }

	up(&info->sem);
	return 0;
}

static long nvmap_ioctl(struct file *file, unsigned int cmd,
	unsigned long arg)
{
	struct nvmap_info *info = file->private_data;
	void __user * argp = (void __user *) arg;
	int err;

	if(!info)
		return -EBADFD;

	down(&info->sem);

	switch (cmd) {
	case NVMAP_MAP:
		err = nvmap_cmd_map(info, argp);
		break;

	case NVMAP_UNMAP:
		err = nvmap_cmd_unmap(info, argp);
		break;

	default:
		err = -EINVAL;
		break;
	};

	up(&info->sem);

	return err;
}

