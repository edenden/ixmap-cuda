#ifndef _IXMAPFWD_FORWARD_H
#define _IXMAPFWD_FORWARD_H

#include "thread.h"

struct ixmap_packet_cuda {
	int	outif;
};

void forward_process_offload(struct ixmapfwd_thread *thread,
	unsigned int port_index, struct ixmap_packet *packet,
	unsigned int num_packets, struct ixmapfwd_thread_cuda *thread_cuda,
	struct ixmap_packet_cuda *result, uint8_t *read_buf);
void forward_process_tun(struct ixmapfwd_thread *thread, unsigned int port_index,
	uint8_t *read_buf, unsigned int read_size);

#endif /* _IXMAPFWD_FORWARD_H */
