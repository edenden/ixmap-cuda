#ifndef _IXMAP_CUDA_H
#define _IXMAP_CUDA_H

#include <net/if.h>

#define ALIGN(x,a)		__ALIGN_MASK(x,(typeof(x))(a)-1)
#define __ALIGN_MASK(x,mask)	(((x)+(mask))&~(mask))

#define SIZE_1GB (1ul << 30)
#define SIZE_256MB (1ul << 28)

#define CONFIG_X86_L1_CACHE_SHIFT \
				(6)
#define L1_CACHE_SHIFT		(CONFIG_X86_L1_CACHE_SHIFT)
#define L1_CACHE_BYTES		(1 << L1_CACHE_SHIFT)

struct ixmap_ring {
	void		*addr_virt;
	unsigned long	addr_dma;

	uint8_t		*tail;
	uint16_t	next_to_use;
	uint16_t	next_to_clean;
	int32_t		*slot_index;
};

struct ixmap_desc {
	void			*addr_virt;
	struct ixmap_mnode	*node;
};

struct ixmap_buf {
	void			*addr_virt;
	unsigned long		*addr_dma;
	uint32_t		buf_size;
	uint32_t		count;
	int32_t			*slots;
};

struct ixmap_handle {
 	int			fd;
	void			*bar;
	unsigned long		bar_size;

	struct ixmap_ring	*tx_ring;
	struct ixmap_ring	*rx_ring;
	struct ixmap_buf	*buf;

	uint32_t		num_tx_desc;
	uint32_t		num_rx_desc;
	uint32_t		rx_budget;
	uint32_t		tx_budget;

	uint32_t		num_queues;
	uint16_t		num_interrupt_rate;
	uint32_t		promisc;
	uint32_t		mtu_frame;
	uint32_t		buf_size;
	uint8_t			mac_addr[ETH_ALEN];
	char			interface_name[IFNAMSIZ];
};

struct ixmap_port {
	void			*irqreg[2];
	struct ixmap_ring	*rx_ring;
	struct ixmap_ring	*tx_ring;
	uint32_t		rx_slot_next;
	uint32_t		rx_slot_offset;
	uint32_t		tx_suspended;
	uint32_t		mtu_frame;
	uint32_t		num_tx_desc;
	uint32_t		num_rx_desc;
	uint32_t		num_queues;
	uint32_t		rx_budget;
	uint32_t		tx_budget;
	uint8_t			mac_addr[ETH_ALEN];
	const char		*interface_name;

	unsigned long		count_rx_alloc_failed;
	unsigned long		count_rx_clean_total;
	unsigned long		count_tx_xmit_failed;
	unsigned long		count_tx_clean_total;
};

struct ixmap_plane {
	struct ixmap_port 	*ports;
};

enum {
	IXGBE_DMA_CACHE_DEFAULT = 0,
	IXGBE_DMA_CACHE_DISABLE,
	IXGBE_DMA_CACHE_WRITECOMBINE
};

#define IXMAP_MAP		_IOW('U', 210, int)
struct ixmap_map_req {
	unsigned long		addr_virt;
	unsigned long		addr_dma;
	unsigned long		size;
	uint8_t			cache;
};

#define IXMAP_UNMAP		_IOW('U', 211, int)
struct ixmap_unmap_req {
	unsigned long		addr_dma;
};

#endif /* _IXMAP_CUDA_H */
