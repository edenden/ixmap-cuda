#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <net/ethernet.h>

#include "ixmap.h"

__host__ void ixmap_slot_release_cuda(struct ixmap_buf *buf,
	int slot_index)
{
	buf->slots[slot_index] = 0;
	return;
}

__device__ uint8_t *ixmap_macaddr_cuda(struct ixmap_plane *plane,
	unsigned int port_index)
{
	return plane->ports[port_index].mac_addr;
}
