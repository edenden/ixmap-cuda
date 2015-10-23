#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
#include <stddef.h>
#include <ixmap.h>

#include "main.h"
#include "forward.h"
#include "thread.h"

static int forward_arp_process(struct ixmapfwd_thread *thread,
	unsigned int port_index, struct ixmap_packet *packet);

void forward_process_tun(struct ixmapfwd_thread *thread, unsigned int port_index,
	uint8_t *read_buf, unsigned int read_size)
{
	struct ixmap_packet packet;

	if(read_size > ixmap_slot_size(thread->buf))
		goto err_slot_size;

	packet.slot_index = ixmap_slot_assign(thread->buf,
		thread->plane, port_index);
	if(packet.slot_index < 0){
		goto err_slot_assign;
	}

	packet.slot_buf = ixmap_slot_addr_virt(thread->buf, packet.slot_index);
	memcpy(packet.slot_buf, read_buf, read_size);
	packet.slot_size = read_size;

#ifdef DEBUG
	forward_dump(&packet);
#endif

	ixmap_tx_assign(thread->plane, port_index, thread->buf, &packet);
	return;

err_slot_assign:
err_slot_size:
	return;
}

static int forward_arp_process(struct ixmapfwd_thread *thread,
	unsigned int port_index, struct ixmap_packet *packet)
{
	int fd, ret;

	fd = thread->tun_plane->ports[port_index].fd;
	ret = write(fd, packet->slot_buf, packet->slot_size);
	if(ret < 0)
		goto err_write_tun;

	return -1;

err_write_tun:
	return -1;
}


