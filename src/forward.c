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

#include <cuda.h>
#include <cuda_runtime.h>

#include "linux/list.h"
#include "main.h"
#include "forward.h"
#include "thread.h"

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
	cudaMemcpy(packet.slot_buf, read_buf, read_size, cudaMemcpyHostToDevice);
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

void forward_process_synchronized(struct ixmapfwd_thread *thread,
	unsigned int port_index, struct ixmap_packet *packet,
	unsigned int num_packets, struct ixmap_packet_cuda *result, uint8_t *read_buf)
{
	int fd, i;

	cudaStreamSynchronize(thread->stream);

	for(i = 0; i < num_packets; i++){
		if(result[i].outif >= 0){
			ixmap_tx_assign(thread->plane, result[i].outif,
				thread->buf, &packet[i]);
		}else if(result[i].outif == -1){
			goto packet_drop;
		}else{
			goto packet_inject;
		}

		continue;
packet_inject:
		memcpy(read_buf, packet[i].slot_buf, packet[i].slot_size);
		fd = thread->tun_plane->ports[port_index].fd;
		write(fd, read_buf, packet[i].slot_size);
packet_drop:
		ixmap_slot_release(thread->buf, packet[i].slot_index);
	}
	return;
}

