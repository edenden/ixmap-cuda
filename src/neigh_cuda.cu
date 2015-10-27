#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <netinet/ip.h>
#include <arpa/inet.h>
#include <stddef.h>

extern "C" {
#include "linux/list_cuda.h"

#include "main.h"
#include "neigh.h"
#include "hash.h"
}

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
__device__ unsigned int neigh_key_generate_v4_cuda(void *key,
	unsigned int bit_len)
{
	/* On some cpus multiply is faster, on others gcc will do shifts */
	uint32_t hash = *((uint32_t *)key) * GOLDEN_RATIO_PRIME_32;

	/* High bits are more random, so use them. */
	return hash >> (32 - bit_len);
}

extern "C"
__device__ unsigned int neigh_key_generate_v6_cuda(void *key,
	unsigned int bit_len)
{
	uint64_t hash = ((uint64_t *)key)[0] * ((uint64_t *)key)[1];
	hash *= GOLDEN_RATIO_PRIME_64;

	return hash >> (64 - bit_len);
}

extern "C"
__device__ int neigh_key_compare_v4_cuda(void *key_tgt, void *key_ent)
{
	return ((uint32_t *)key_tgt)[0] ^ ((uint32_t *)key_ent)[0] ?
		1 : 0;
}

extern "C"
__device__ int neigh_key_compare_v6_cuda(void *key_tgt, void *key_ent)
{
	return ((uint64_t *)key_tgt)[0] ^ ((uint64_t *)key_ent)[0] ?
		1 : (((uint64_t *)key_tgt)[1] ^ ((uint64_t *)key_ent)[1] ?
		1 : 0);
}

extern "C"
__device__ struct neigh_entry *neigh_lookup_v4(struct neigh_table *neigh,
	void *dst_addr)
{
	struct hash_entry *hash_entry;
	struct neigh_entry *neigh_entry;

	hash_entry = hash_lookup_v4(&neigh->table, dst_addr);
	if(!hash_entry)
		goto err_hash_lookup;

	neigh_entry = hash_entry(hash_entry, struct neigh_entry, hash);
	return neigh_entry;

err_hash_lookup:
	return NULL;
}

extern "C"
__device__ struct neigh_entry *neigh_lookup_v6(struct neigh_table *neigh,
	void *dst_addr)
{
	struct hash_entry *hash_entry;
	struct neigh_entry *neigh_entry;

	hash_entry = hash_lookup_v6(&neigh->table, dst_addr);
	if(!hash_entry)
		goto err_hash_lookup;

	neigh_entry = hash_entry(hash_entry, struct neigh_entry, hash);
	return neigh_entry;

err_hash_lookup:
	return NULL;
}

