#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/epoll.h>
#include <net/ethernet.h>
#include <signal.h>
#include <sys/signalfd.h>
#include <pthread.h>

#include "ixmap.h"
#include "driver.h"

static inline uint16_t ixmap_desc_unused(struct ixmap_ring *ring,
	uint16_t num_desc);
static inline void ixmap_slot_attach(struct ixmap_ring *ring,
	uint16_t desc_index, int slot_index);
inline void ixmap_slot_release(struct ixmap_buf *buf,
	int slot_index);
static inline unsigned long ixmap_slot_addr_dma(struct ixmap_buf *buf,
	int slot_index, int port_index);

__device__ static inline uint16_t ixmap_desc_unused(struct ixmap_ring *ring,
	uint16_t num_desc)
{
        uint16_t next_to_clean = ring->next_to_clean;
        uint16_t next_to_use = ring->next_to_use;

	return next_to_clean > next_to_use
		? next_to_clean - next_to_use - 1
		: (num_desc - next_to_use) + next_to_clean - 1;
}

__device__ void ixmap_tx_assign(struct ixmap_plane *plane, unsigned int port_index,
	struct ixmap_buf *buf, struct ixmap_packet *packet)
{
	struct ixmap_port *port;
	struct ixmap_ring *tx_ring;
	union ixmap_adv_tx_desc *tx_desc;
	uint16_t unused_count;
	uint32_t tx_flags;
	uint16_t next_to_use;
	uint64_t addr_dma;
	uint32_t cmd_type;
	uint32_t olinfo_status;

	port = &plane->ports[port_index];
	tx_ring = port->tx_ring;

	unused_count = ixmap_desc_unused(tx_ring, port->num_tx_desc);
	if(!unused_count){
		port->count_tx_xmit_failed++;
		ixmap_slot_release(buf, packet->slot_index);
		return;
	}

	if(unlikely(packet->slot_size > IXGBE_MAX_DATA_PER_TXD)){
		port->count_tx_xmit_failed++;
		ixmap_slot_release(buf, packet->slot_index);
		return;
	}

	ixmap_slot_attach(tx_ring, tx_ring->next_to_use, packet->slot_index);
	addr_dma = (uint64_t)ixmap_slot_addr_dma(buf, packet->slot_index, port_index);

	/* set type for advanced descriptor with frame checksum insertion */
	tx_desc = IXGBE_TX_DESC(tx_ring, tx_ring->next_to_use);
	tx_flags = IXGBE_ADVTXD_DTYP_DATA | IXGBE_ADVTXD_DCMD_DEXT
		| IXGBE_ADVTXD_DCMD_IFCS;
	cmd_type = packet->slot_size | IXGBE_TXD_CMD_EOP | IXGBE_TXD_CMD_RS | tx_flags;
	olinfo_status = packet->slot_size << IXGBE_ADVTXD_PAYLEN_SHIFT;

	tx_desc->read.buffer_addr = htole64(addr_dma);
	tx_desc->read.cmd_type_len = htole32(cmd_type);
	tx_desc->read.olinfo_status = htole32(olinfo_status);

	next_to_use = tx_ring->next_to_use + 1;
	tx_ring->next_to_use =
		(next_to_use < port->num_tx_desc) ? next_to_use : 0;

	port->tx_suspended++;
	return;
}

__device__ uint8_t *ixmap_macaddr(struct ixmap_plane *plane,
	unsigned int port_index)
{
	return plane->ports[port_index].mac_addr;
}

__device__ static inline void ixmap_slot_attach(struct ixmap_ring *ring,
	uint16_t desc_index, int slot_index)
{
	ring->slot_index[desc_index] = slot_index;
	return;
}

__device__ inline void ixmap_slot_release(struct ixmap_buf *buf,
	int slot_index)
{
	buf->slots[slot_index] = 0;
	return;
}

__device__ static inline unsigned long ixmap_slot_addr_dma(struct ixmap_buf *buf,
	int slot_index, int port_index)
{
	return buf->addr_dma[port_index] + (buf->buf_size * slot_index);
}

