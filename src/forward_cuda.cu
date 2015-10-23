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

extern "C" {
#include "main.h"
#include "forward.h"
#include "thread.h"
}

__global__ void forward_process(struct ixmapfwd_thread *thread,
	unsigned int port_index, struct ixmap_packet *packet)
{
	struct ethhdr *eth;
	int ret;

	eth = (struct ethhdr *)packet->slot_buf;
	switch(ntohs(eth->h_proto)){
	case ETH_P_ARP:
		ret = forward_arp_process(thread, port_index, packet);
		break;
	case ETH_P_IP:
		ret = forward_ip_process(thread, port_index, packet);
		break;
	case ETH_P_IPV6:
		ret = forward_ip6_process(thread, port_index, packet);
		break;
	default:
		ret = -1;
		break;
	}

	if(ret < 0){
		goto packet_drop;
	}

	ixmap_tx_assign(thread->plane, ret, thread->buf, packet);
	return;

packet_drop:
	ixmap_slot_release(thread->buf, packet->slot_index);
	return;
}

__device__ static int forward_ip_process(struct ixmapfwd_thread *thread,
	unsigned int port_index, struct ixmap_packet *packet)
{
	struct ethhdr		*eth;
	struct iphdr		*ip;
	struct fib_entry	*fib_entry;
	struct neigh_entry	*neigh_entry;
	uint8_t			*dst_mac, *src_mac;
	uint32_t		check;
	int			fd, ret;

	eth = (struct ethhdr *)packet->slot_buf;
	ip = (struct iphdr *)(packet->slot_buf + sizeof(struct ethhdr));

	fib_entry = fib_lookup(thread->fib_inet, &ip->daddr);
	if(!fib_entry)
		goto packet_drop;

	if(unlikely(fib_entry->port_index < 0))
		goto packet_local;

	switch(fib_entry->type){
	case FIB_TYPE_LOCAL:
		goto packet_local;
		break;
	case FIB_TYPE_LINK:
		neigh_entry = neigh_lookup(
			thread->neigh_inet[fib_entry->port_index],
			&ip->daddr);
		break;
	case FIB_TYPE_FORWARD:
		neigh_entry = neigh_lookup(
			thread->neigh_inet[fib_entry->port_index],
			fib_entry->nexthop);
		break;
	default:
		neigh_entry = NULL;
		break;
	}

	if(!neigh_entry)
		goto packet_local;

	if(unlikely(ip->ttl == 1))
		goto packet_local;

	ip->ttl--;

	check = ip->check;
	check += htons(0x0100);
	ip->check = check + ((check >= 0xFFFF) ? 1 : 0);

	dst_mac = neigh_entry->dst_mac;
	src_mac = ixmap_macaddr(thread->plane, fib_entry->port_index);
	memcpy(eth->h_dest, dst_mac, ETH_ALEN);
	memcpy(eth->h_source, src_mac, ETH_ALEN);

	ret = fib_entry->port_index;
	return ret;

packet_local:
	fd = thread->tun_plane->ports[port_index].fd;
	write(fd, packet->slot_buf, packet->slot_size);
packet_drop:
	return -1;
}

__device__ static int forward_ip6_process(struct ixmapfwd_thread *thread,
	unsigned int port_index, struct ixmap_packet *packet)
{
	struct ethhdr		*eth;
	struct ip6_hdr		*ip6;
	struct fib_entry	*fib_entry;
	struct neigh_entry	*neigh_entry;
	uint8_t			*dst_mac, *src_mac;
	int			fd, ret;

	eth = (struct ethhdr *)packet->slot_buf;
	ip6 = (struct ip6_hdr *)(packet->slot_buf + sizeof(struct ethhdr));

	if(unlikely(ip6->ip6_dst.s6_addr[0] == 0xfe
	&& (ip6->ip6_dst.s6_addr[1] & 0xc0) == 0x80))
		goto packet_local;

	fib_entry = fib_lookup(thread->fib_inet6, (uint32_t *)&ip6->ip6_dst);
	if(!fib_entry)
		goto packet_drop;

	if(unlikely(fib_entry->port_index < 0))
		goto packet_local;

	switch(fib_entry->type){
	case FIB_TYPE_LOCAL:
		goto packet_local;
		break;
	case FIB_TYPE_LINK:
		neigh_entry = neigh_lookup(
			thread->neigh_inet6[fib_entry->port_index],
			&ip6->ip6_dst);
		break;
	case FIB_TYPE_FORWARD:
		neigh_entry = neigh_lookup(
			thread->neigh_inet6[fib_entry->port_index],
			fib_entry->nexthop);
		break;
	default:
		neigh_entry = NULL;
		break;
	}

	if(!neigh_entry)
		goto packet_local;

	if(unlikely(ip6->ip6_hlim == 1))
		goto packet_local;

	ip6->ip6_hlim--;

	dst_mac = neigh_entry->dst_mac;
	src_mac = ixmap_macaddr(thread->plane, fib_entry->port_index);
	memcpy(eth->h_dest, dst_mac, ETH_ALEN);
	memcpy(eth->h_source, src_mac, ETH_ALEN);

	ret = fib_entry->port_index;
	return ret;

packet_local:
	fd = thread->tun_plane->ports[port_index].fd;
	write(fd, packet->slot_buf, packet->slot_size);
packet_drop:
	return -1;
}

