#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
#include <stddef.h>

extern "C" {
#include <ixmap.h>
#include <ixmap_cuda.h>

#include "linux/list_cuda.h"
#include "main.h"
#include "forward.h"
#include "thread.h"
#include "neigh.h"
#include "fib.h"
#include "misc.h"
}

extern "C"
__global__ static void forward_process(struct ixmapfwd_thread_cuda *thread_cuda,
	unsigned int num_packets, unsigned int port_index, struct ixmap_packet *packet,
	struct ixmap_packet_cuda *result);
extern "C"
__device__ static int forward_ip_process(struct ixmapfwd_thread_cuda *thread_cuda,
	unsigned int port_index, struct ixmap_packet *packet);
extern "C"
__device__ static int forward_ip6_process(struct ixmapfwd_thread_cuda *thread_cuda,
	unsigned int port_index, struct ixmap_packet *packet);

extern "C"
__host__ void forward_process_offload(struct ixmapfwd_thread *thread,
	unsigned int port_index, struct ixmap_packet *packet,
	unsigned int num_packets, struct ixmapfwd_thread_cuda *thread_cuda,
	struct ixmap_packet_cuda *result, uint8_t *read_buf)
{
	int fd, i;

	forward_process<<<CUDA_NBLOCK_MAX, CUDA_NTHREAD_MAX>>>
		(thread_cuda, num_packets, port_index, packet, result);

	cudaDeviceSynchronize();

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
		cudaMemcpy(read_buf, packet[i].slot_buf, packet[i].slot_size,
			cudaMemcpyDeviceToHost);
		fd = thread->tun_plane->ports[port_index].fd;
		write(fd, read_buf, packet[i].slot_size);
packet_drop:
		ixmap_slot_release_cuda(thread->buf, packet[i].slot_index);
	}
	return;
}

extern "C"
__global__ static void forward_process(struct ixmapfwd_thread_cuda *thread_cuda,
	unsigned int num_packets, unsigned int port_index, struct ixmap_packet *packet,
	struct ixmap_packet_cuda *result)
{
	struct ethhdr *eth;
	int index;

	index = blockDim.x * blockIdx.x + threadIdx.x;

	if(!(index < num_packets))
		goto out;

	eth = (struct ethhdr *)packet[index].slot_buf;
	switch(bswap_16(eth->h_proto)){
	case ETH_P_ARP:
		result[index].outif = -2;
		break;
	case ETH_P_IP:
		result[index].outif =
			forward_ip_process(thread_cuda, port_index, &packet[index]);
		break;
	case ETH_P_IPV6:
		result[index].outif =
			forward_ip6_process(thread_cuda, port_index, &packet[index]);
		break;
	default:
		result[index].outif = -1;
		break;
	}

out:
	return;
}

extern "C"
__device__ static int forward_ip_process(struct ixmapfwd_thread_cuda *thread_cuda,
	unsigned int port_index, struct ixmap_packet *packet)
{
	struct ethhdr		*eth;
	struct iphdr		*ip;
	struct fib_entry	*fib_entry;
	struct neigh_entry	*neigh_entry;
	uint8_t			*dst_mac, *src_mac;
	uint32_t		check;
	int			ret;

	eth = (struct ethhdr *)packet->slot_buf;
	ip = (struct iphdr *)(eth + 1);

	fib_entry = fib_lookup(thread_cuda->fib_inet, &ip->daddr);
	if(!fib_entry)
		goto packet_drop;

	if(fib_entry->port_index < 0)
		goto packet_local;

	switch(fib_entry->type){
	case FIB_TYPE_LOCAL:
		goto packet_local;
	case FIB_TYPE_LINK:
		neigh_entry = neigh_lookup_v4(
			thread_cuda->neigh_inet[fib_entry->port_index],
			&ip->daddr);
		break;
	case FIB_TYPE_FORWARD:
		neigh_entry = neigh_lookup_v4(
			thread_cuda->neigh_inet[fib_entry->port_index],
			fib_entry->nexthop);
		break;
	default:
		neigh_entry = NULL;
		break;
	}

	if(!neigh_entry)
		goto packet_local;

	if(ip->ttl == 1)
		goto packet_local;

	ip->ttl--;

	check = ip->check;
	check += bswap_16(0x0100);
	ip->check = check + ((check >= 0xFFFF) ? 1 : 0);

	dst_mac = neigh_entry->dst_mac;
	src_mac = ixmap_macaddr_cuda(thread_cuda->plane, fib_entry->port_index);
	memcpy(eth->h_dest, dst_mac, ETH_ALEN);
	memcpy(eth->h_source, src_mac, ETH_ALEN);

	ret = fib_entry->port_index;
	return ret;

packet_local:
	return -2;
packet_drop:
	return -1;
}

extern "C"
__device__ static int forward_ip6_process(struct ixmapfwd_thread_cuda *thread_cuda,
	unsigned int port_index, struct ixmap_packet *packet)
{
	struct ethhdr		*eth;
	struct ip6_hdr		*ip6;
	struct fib_entry	*fib_entry;
	struct neigh_entry	*neigh_entry;
	uint8_t			*dst_mac, *src_mac;
	int			ret;

	eth = (struct ethhdr *)packet->slot_buf;
	ip6 = (struct ip6_hdr *)(eth + 1);

	if(ip6->ip6_dst.s6_addr[0] == 0xfe
	&& (ip6->ip6_dst.s6_addr[1] & 0xc0) == 0x80)
		goto packet_local;

	fib_entry = fib_lookup(thread_cuda->fib_inet6, (uint32_t *)&ip6->ip6_dst);
	if(!fib_entry)
		goto packet_drop;

	if(fib_entry->port_index < 0)
		goto packet_local;

	switch(fib_entry->type){
	case FIB_TYPE_LOCAL:
		goto packet_local;
	case FIB_TYPE_LINK:
		neigh_entry = neigh_lookup_v6(
			thread_cuda->neigh_inet6[fib_entry->port_index],
			&ip6->ip6_dst);
		break;
	case FIB_TYPE_FORWARD:
		neigh_entry = neigh_lookup_v6(
			thread_cuda->neigh_inet6[fib_entry->port_index],
			fib_entry->nexthop);
		break;
	default:
		neigh_entry = NULL;
		break;
	}

	if(!neigh_entry)
		goto packet_local;

	if(ip6->ip6_hlim == 1)
		goto packet_local;

	ip6->ip6_hlim--;

	dst_mac = neigh_entry->dst_mac;
	src_mac = ixmap_macaddr_cuda(thread_cuda->plane, fib_entry->port_index);
	memcpy(eth->h_dest, dst_mac, ETH_ALEN);
	memcpy(eth->h_source, src_mac, ETH_ALEN);

	ret = fib_entry->port_index;
	return ret;

packet_local:
	return -2;
packet_drop:
	return -1;
}

