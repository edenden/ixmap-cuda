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

#include "ixmap.h"
#include "memory.h"

static void ixmap_irq_enable_queues(struct ixmap_handle *ih, uint64_t qmask);
static int ixmap_dma_map(struct ixmap_handle *ih, void *addr_virt,
	unsigned long *addr_dma, unsigned long size);
static int ixmap_dma_unmap(struct ixmap_handle *ih, unsigned long addr_dma);

inline uint32_t ixmap_readl(const volatile void *addr)
{
	return htole32( *(volatile uint32_t *) addr );
}

inline void ixmap_writel(uint32_t b, volatile void *addr)
{
	*(volatile uint32_t *) addr = htole32(b);
	return;
}

inline uint32_t ixmap_read_reg(struct ixmap_handle *ih, uint32_t reg)
{
	uint32_t value = ixmap_readl(ih->bar + reg);
	return value;
}

inline void ixmap_write_reg(struct ixmap_handle *ih, uint32_t reg, uint32_t value)
{
	ixmap_writel(value, ih->bar + reg);
	return;
}

inline void ixmap_write_flush(struct ixmap_handle *ih)
{
	ixmap_read_reg(ih, IXGBE_STATUS);
	return;
}

void ixmap_irq_enable(struct ixmap_handle *ih)
{
	uint32_t mask;

	mask = (IXGBE_EIMS_ENABLE_MASK & ~IXGBE_EIMS_RTX_QUEUE);

	/* XXX: Currently we don't support misc interrupts */
	mask &= ~IXGBE_EIMS_LSC;
	mask &= ~IXGBE_EIMS_TCP_TIMER;
	mask &= ~IXGBE_EIMS_OTHER;

	ixmap_write_reg(ih, IXGBE_EIMS, mask);

	ixmap_irq_enable_queues(ih, ~0);
	ixmap_write_flush(ih);

	return;
}

static void ixmap_irq_enable_queues(struct ixmap_handle *ih, uint64_t qmask)
{
	uint32_t mask;

	mask = (qmask & 0xFFFFFFFF);
	if (mask)
		ixmap_write_reg(ih, IXGBE_EIMS_EX(0), mask);
	mask = (qmask >> 32);
	if (mask)
		ixmap_write_reg(ih, IXGBE_EIMS_EX(1), mask);

	return;
}

struct ixmap_plane *ixmap_plane_alloc(struct ixmap_handle **ih_list,
	struct ixmap_buf *buf, int ih_num, int queue_index)
{
	struct ixmap_plane *plane;
	int i;

	plane = malloc(sizeof(struct ixmap_plane));
	if(!plane)
		goto err_plane_alloc;

	plane->ports = malloc(sizeof(struct ixmap_port) * ih_num);
	if(!plane->ports){
		printf("failed to allocate port for each plane\n");
		goto err_alloc_ports;
	}

	for(i = 0; i < ih_num; i++){
		plane->ports[i].interface_name = ih_list[i]->interface_name;
		plane->ports[i].irqreg[0] = ih_list[i]->bar + IXGBE_EIMS_EX(0);
		plane->ports[i].irqreg[1] = ih_list[i]->bar + IXGBE_EIMS_EX(1);
		plane->ports[i].rx_ring = &(ih_list[i]->rx_ring[queue_index]);
		plane->ports[i].tx_ring = &(ih_list[i]->tx_ring[queue_index]);
		plane->ports[i].rx_slot_next = 0;
		plane->ports[i].rx_slot_offset = i * buf->count;
		plane->ports[i].tx_suspended = 0;
		plane->ports[i].num_rx_desc = ih_list[i]->num_rx_desc;
		plane->ports[i].num_tx_desc = ih_list[i]->num_tx_desc;
		plane->ports[i].num_queues = ih_list[i]->num_queues;
		plane->ports[i].rx_budget = ih_list[i]->rx_budget;
		plane->ports[i].tx_budget = ih_list[i]->tx_budget;
		plane->ports[i].mtu_frame = ih_list[i]->mtu_frame;
		plane->ports[i].count_rx_alloc_failed = 0;
		plane->ports[i].count_rx_clean_total = 0;
		plane->ports[i].count_tx_xmit_failed = 0;
		plane->ports[i].count_tx_clean_total = 0;

		memcpy(plane->ports[i].mac_addr, ih_list[i]->mac_addr, ETH_ALEN);
	}

	return plane;

err_alloc_ports:
	free(plane);
err_plane_alloc:
	return NULL;
}

void ixmap_plane_release(struct ixmap_plane *plane)
{
	free(plane->ports);
	free(plane);

	return;
}

struct ixmap_desc *ixmap_desc_alloc(struct ixmap_handle **ih_list, int ih_num,
	int queue_index)
{
	struct ixmap_desc *desc;
	unsigned long size_tx_desc, size_rx_desc, size_mem;
	void *addr_virt, *addr_mem;
	int i, ret;
	int desc_assigned = 0;

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
	munmap(desc->addr_virt, SIZE_1GB);
err_mmap:
	free(desc);
err_alloc_desc:
	return NULL;
}

void ixmap_desc_release(struct ixmap_handle **ih_list, int ih_num,
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

	munmap(desc->addr_virt, SIZE_1GB);
	free(desc);
	return;
}

struct ixmap_buf *ixmap_buf_alloc(struct ixmap_handle **ih_list,
	int ih_num, uint32_t count, uint32_t buf_size)
{
	struct ixmap_buf *buf;
	void	*addr_virt;
	unsigned long addr_dma, size;
	int *slots;
	int ret, i, mapped_ports = 0;

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
	munmap(addr_virt, size);
err_mmap:
	free(buf->addr_dma);
err_alloc_buf_addr_dma:
	free(buf);
err_alloc_buf:
	return NULL;
}

void ixmap_buf_release(struct ixmap_buf *buf,
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

	size = buf->buf_size * buf->count;
	munmap(buf->addr_virt, size);
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

struct ixmap_handle *ixmap_open(unsigned int port_index,
	unsigned int num_queues_req, unsigned short intr_rate,
	unsigned int rx_budget, unsigned int tx_budget,
	unsigned int mtu_frame, unsigned int promisc,
	unsigned int num_rx_desc, unsigned int num_tx_desc)
{
	struct ixmap_handle *ih;
	char filename[FILENAME_SIZE];
	struct ixmap_info_req req_info;
	struct ixmap_up_req req_up;

	ih = malloc(sizeof(struct ixmap_handle));
	if (!ih)
		goto err_alloc_ih;
	memset(ih, 0, sizeof(struct ixmap_handle));

	snprintf(filename, sizeof(filename), "/dev/%s%d",
		IXMAP_IFNAME, port_index);
	ih->fd = open(filename, O_RDWR);
	if (ih->fd < 0)
		goto err_open;

	/* Get device information */
	memset(&req_info, 0, sizeof(struct ixmap_info_req));
	if(ioctl(ih->fd, IXMAP_INFO, (unsigned long)&req_info) < 0)
		goto err_ioctl_info;

	/* UP the device */
	memset(&req_up, 0, sizeof(struct ixmap_up_req));

	ih->num_interrupt_rate =
		min(intr_rate, req_info.max_interrupt_rate);
	req_up.num_interrupt_rate = ih->num_interrupt_rate;

	ih->num_queues =
		min(req_info.max_rx_queues, req_info.max_tx_queues);
	ih->num_queues = min(num_queues_req, ih->num_queues);
	req_up.num_rx_queues = ih->num_queues;
	req_up.num_tx_queues = ih->num_queues;

	if(ioctl(ih->fd, IXMAP_UP, (unsigned long)&req_up) < 0)
		goto err_ioctl_up;

	/* Map PCI config register space */
	ih->bar = mmap(NULL, req_info.mmio_size,
		PROT_READ | PROT_WRITE, MAP_SHARED, ih->fd, 0);
	if(ih->bar == MAP_FAILED)
		goto err_mmap;

	ih->rx_ring = malloc(sizeof(struct ixmap_ring) * ih->num_queues);
	if(!ih->rx_ring)
		goto err_alloc_rx_ring;

	ih->tx_ring = malloc(sizeof(struct ixmap_ring) * ih->num_queues);
	if(!ih->tx_ring)
		goto err_alloc_tx_ring;

	ih->bar_size = req_info.mmio_size;
	ih->promisc = !!promisc;
	ih->rx_budget = rx_budget;
	ih->tx_budget = tx_budget;
	ih->mtu_frame = mtu_frame;
	ih->num_rx_desc = num_rx_desc;
	ih->num_tx_desc = num_tx_desc;
	memcpy(ih->mac_addr, req_info.mac_addr, ETH_ALEN);
	snprintf(ih->interface_name, sizeof(ih->interface_name), "%s%d",
		IXMAP_IFNAME, port_index);

	return ih;

err_alloc_tx_ring:
	free(ih->rx_ring);
err_alloc_rx_ring:
	munmap(ih->bar, ih->bar_size);
err_mmap:
err_ioctl_up:
err_ioctl_info:
	close(ih->fd);
err_open:
	free(ih);
err_alloc_ih:
	return NULL;
}

void ixmap_close(struct ixmap_handle *ih)
{
	free(ih->tx_ring);
	free(ih->rx_ring);
	munmap(ih->bar, ih->bar_size);
	close(ih->fd);
	free(ih);

	return;
}

unsigned int ixmap_bufsize_get(struct ixmap_handle *ih)
{
	return ih->buf_size;
}

uint8_t *ixmap_macaddr_default(struct ixmap_handle *ih)
{
	return ih->mac_addr;
}

unsigned int ixmap_mtu_get(struct ixmap_handle *ih)
{
	return ih->mtu_frame;
}

struct ixmap_irqdev_handle *ixmap_irqdev_open(struct ixmap_plane *plane,
	unsigned int port_index, unsigned int queue_index,
	enum ixmap_irq_direction direction)
{
	struct ixmap_port *port;
	struct ixmap_irqdev_handle *irqh;
	char filename[FILENAME_SIZE];
	uint64_t qmask;

	port = &plane->ports[port_index];

	if(queue_index >= port->num_queues){
		goto err_invalid_queue_index;
	}

	switch(direction){
	case IXMAP_IRQ_RX:
		snprintf(filename, sizeof(filename), "/dev/%s-irqrx%d",
			port->interface_name, queue_index);
		qmask = 1 << queue_index;
		break;
	case IXMAP_IRQ_TX:
		snprintf(filename, sizeof(filename), "/dev/%s-irqtx%d",
			port->interface_name, queue_index);
		qmask = 1 << (queue_index + port->num_queues);
		break;
	default:
		goto err_invalid_direction;
		break;
	}

	irqh = malloc(sizeof(struct ixmap_irqdev_handle));
	if(!irqh)
		goto err_alloc_handle;

	irqh->fd = open(filename, O_RDWR);
	if(irqh->fd < 0)
		goto err_open;

	irqh->port_index = port_index;
	irqh->qmask = qmask;

	return irqh;

err_open:
	free(irqh);
err_alloc_handle:
err_invalid_direction:
err_invalid_queue_index:
	return NULL;
}

void ixmap_irqdev_close(struct ixmap_irqdev_handle *irqh)
{
	close(irqh->fd);
	free(irqh);

	return;
}

int ixmap_irqdev_setaffinity(struct ixmap_irqdev_handle *irqh,
	unsigned int core_id)
{
	struct ixmap_irqdev_info_req req_info;
	FILE *file;
	char filename[FILENAME_SIZE];
	uint32_t mask_low, mask_high;
	int ret;

	mask_low = core_id <= 31 ? 1 << core_id : 0;
	mask_high = core_id <= 31 ? 0 : 1 << (core_id - 31);

	ret = ioctl(irqh->fd, IXMAP_IRQDEV_INFO, (unsigned long)&req_info);
	if(ret < 0){
		printf("failed to UIO_IRQ_INFO\n");
		goto err_irqdev_info;
	}

	snprintf(filename, sizeof(filename),
		"/proc/irq/%d/smp_affinity", req_info.vector);
	file = fopen(filename, "w");
	if(!file){
		printf("failed to open smp_affinity\n");
		goto err_open_proc;
	}

	ret = fprintf(file, "%08x,%08x", mask_high, mask_low);
	if(ret < 0){
		printf("failed to set affinity\n");
		goto err_set_affinity;
	}

	fclose(file);
	return 0;

err_set_affinity:
	fclose(file);
err_open_proc:
err_irqdev_info:
	return -1;
}

int ixmap_irqdev_fd(struct ixmap_irqdev_handle *irqh)
{
	return irqh->fd;
}
