#ifndef _IXMAPFWD_NEIGH_H
#define _IXMAPFWD_NEIGH_H

#include <linux/if_ether.h>
#include <pthread.h>
#include "hash.h"

#define GOLDEN_RATIO_PRIME_32 0x9e370001UL
#define GOLDEN_RATIO_PRIME_64 0x9e37fffffffc0001UL

struct neigh_table {
	struct hash_table	table;
	struct ixmap_marea	*area;
};

struct neigh_entry {
	struct hash_entry	hash;
	uint8_t			dst_mac[ETH_ALEN];
	uint32_t		dst_addr[4];
	struct ixmap_marea	*area;
};

#ifdef __CUDACC__
extern "C"
__device__ unsigned int neigh_key_generate_v4_cuda(void *key,
	unsigned int bit_len);
extern "C"
__device__ unsigned int neigh_key_generate_v6_cuda(void *key,
	unsigned int bit_len);
extern "C"
__device__ int neigh_key_compare_v4_cuda(void *key_tgt, void *key_ent);
extern "C"
__device__ int neigh_key_compare_v6_cuda(void *key_tgt, void *key_ent);
extern "C"
__device__ struct neigh_entry *neigh_lookup_v4(struct neigh_table *neigh,
	void *dst_addr);
__device__ struct neigh_entry *neigh_lookup_v6(struct neigh_table *neigh,
	void *dst_addr);
#endif

struct neigh_table *neigh_alloc(struct ixmap_desc *desc, int family);
void neigh_release(struct neigh_table *neigh);
int neigh_add(struct neigh_table *neigh, int family,
	void *dst_addr, void *mac_addr, struct ixmap_desc *desc);
int neigh_delete(struct neigh_table *neigh, int family,
	void *dst_addr);

#endif /* _IXMAPFWD_NEIGH_H */
