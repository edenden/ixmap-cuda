#ifndef _IXMAP_H
#define _IXMAP_H

#include <net/if.h>

#define ALIGN(x,a)		__ALIGN_MASK(x,(typeof(x))(a)-1)
#define __ALIGN_MASK(x,mask)	(((x)+(mask))&~(mask))

#define SIZE_1GB (1ul << 30)
#define SIZE_256MB (1ul << 28)

#define CONFIG_X86_L1_CACHE_SHIFT \
				(6)
#define L1_CACHE_SHIFT		(CONFIG_X86_L1_CACHE_SHIFT)
#define L1_CACHE_BYTES		(1 << L1_CACHE_SHIFT)

/*
 * microsecond values for various ITR rates shifted by 2 to fit itr register
 * with the first 3 bits reserved 0
 */
#define IXGBE_MIN_RSC_ITR	24
#define IXGBE_100K_ITR		40
#define IXGBE_20K_ITR		200
#define IXGBE_16K_ITR		248
#define IXGBE_10K_ITR		400
#define IXGBE_8K_ITR		500

/* RX descriptor defines */
#define IXGBE_DEFAULT_RXD	512
#define IXGBE_MAX_RXD		4096
#define IXGBE_MIN_RXD		64

/* TX descriptor defines */
#define IXGBE_DEFAULT_TXD	512
#define IXGBE_MAX_TXD		4096
#define IXGBE_MIN_TXD		64

struct ixmap_irqdev_handle;

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

struct ixmap_packet {
	void			*slot_buf;
	unsigned int		slot_size;
	int			slot_index;
};

enum {
	IXGBE_DMA_CACHE_DEFAULT = 0,
	IXGBE_DMA_CACHE_DISABLE,
	IXGBE_DMA_CACHE_WRITECOMBINE
};

enum ixmap_irq_direction {
	IXMAP_IRQ_RX = 0,
	IXMAP_IRQ_TX,
};

/* Transmit Descriptor - Advanced */
union ixmap_adv_tx_desc {
	struct {
		uint64_t buffer_addr; /* Address of descriptor's data buf */
		uint32_t cmd_type_len;
		uint32_t olinfo_status;
	} read;
	struct {
		uint64_t rsvd; /* Reserved */
		uint32_t nxtseq_seed;
		uint32_t status;
	} wb;
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

void ixmap_irq_enable(struct ixmap_handle *ih);
struct ixmap_plane *ixmap_plane_alloc(struct ixmap_handle **ih_list,
	struct ixmap_buf *buf, int ih_num, int queue_index);
void ixmap_plane_release(struct ixmap_plane *plane);
struct ixmap_desc *ixmap_desc_alloc(struct ixmap_handle **ih_list, int ih_num,
	int queue_index);
void ixmap_desc_release(struct ixmap_handle **ih_list, int ih_num,
        int queue_index, struct ixmap_desc *desc);
struct ixmap_buf *ixmap_buf_alloc(struct ixmap_handle **ih_list,
	int ih_num, uint32_t count, uint32_t buf_size);
void ixmap_buf_release(struct ixmap_buf *buf,
	struct ixmap_handle **ih_list, int ih_num);
struct ixmap_handle *ixmap_open(unsigned int port_index,
	unsigned int num_queues_req, unsigned short intr_rate,
	unsigned int rx_budget, unsigned int tx_budget,
	unsigned int mtu_frame, unsigned int promisc,
	unsigned int num_rx_desc, unsigned int num_tx_desc);
void ixmap_close(struct ixmap_handle *ih);
unsigned int ixmap_bufsize_get(struct ixmap_handle *ih);
uint8_t *ixmap_macaddr_default(struct ixmap_handle *ih);
unsigned int ixmap_mtu_get(struct ixmap_handle *ih);
struct ixmap_irqdev_handle *ixmap_irqdev_open(struct ixmap_plane *plane,
	unsigned int port_index, unsigned int queue_index,
	enum ixmap_irq_direction direction);
void ixmap_irqdev_close(struct ixmap_irqdev_handle *irqh);
int ixmap_irqdev_setaffinity(struct ixmap_irqdev_handle *irqh,
	unsigned int core_id);
int ixmap_irqdev_fd(struct ixmap_irqdev_handle *irqh);

inline void ixmap_irq_unmask_queues(struct ixmap_plane *plane,
	struct ixmap_irqdev_handle *irqh);
inline unsigned int ixmap_port_index(struct ixmap_irqdev_handle *irqh);

void ixmap_rx_assign(struct ixmap_plane *plane, unsigned int port_index,
	struct ixmap_buf *buf);
void ixmap_tx_assign(struct ixmap_plane *plane, unsigned int port_index,
	struct ixmap_buf *buf, struct ixmap_packet *packet);
void ixmap_tx_xmit(struct ixmap_plane *plane, unsigned int port_index);
unsigned int ixmap_rx_clean(struct ixmap_plane *plane, unsigned int port_index,
	struct ixmap_buf *buf, struct ixmap_packet *packet);
void ixmap_tx_clean(struct ixmap_plane *plane, unsigned int port_index,
	struct ixmap_buf *buf);

uint8_t *ixmap_macaddr(struct ixmap_plane *plane,
	unsigned int port_index);

inline void *ixmap_slot_addr_virt(struct ixmap_buf *buf,
	uint16_t slot_index);
inline int ixmap_slot_assign(struct ixmap_buf *buf,
	struct ixmap_plane *plane, unsigned int port_index);
inline void ixmap_slot_release(struct ixmap_buf *buf,
	int slot_index);
inline unsigned int ixmap_slot_size(struct ixmap_buf *buf);

inline unsigned long ixmap_count_rx_alloc_failed(struct ixmap_plane *plane,
	unsigned int port_index);
inline unsigned long ixmap_count_rx_clean_total(struct ixmap_plane *plane,
	unsigned int port_index);
inline unsigned long ixmap_count_tx_xmit_failed(struct ixmap_plane *plane,
	unsigned int port_index);
inline unsigned long ixmap_count_tx_clean_total(struct ixmap_plane *plane,
	unsigned int port_index);

struct ixmap_marea *ixmap_mem_alloc(struct ixmap_desc *desc,
	unsigned int size);
void ixmap_mem_free(struct ixmap_marea *area);

void ixmap_configure_rx(struct ixmap_handle *ih);
void ixmap_configure_tx(struct ixmap_handle *ih);

inline uint32_t ixmap_read_reg(struct ixmap_handle *ih, uint32_t reg);
inline void ixmap_write_reg(struct ixmap_handle *ih, uint32_t reg, uint32_t value);

#endif /* _IXMAP_H */
