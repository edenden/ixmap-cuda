#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <netinet/ip.h>
#include <arpa/inet.h>
#include <stddef.h>
#include <ixmap.h>

#include <driver_functions.h>
#include <driver_types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "linux/list.h"
#include "main.h"
#include "neigh.h"

static void neigh_entry_delete(struct hash_entry *entry);
static unsigned int neigh_key_generate_v4(void *key,
	unsigned int bit_len);
static unsigned int neigh_key_generate_v6(void *key,
	unsigned int bit_len);
static int neigh_key_compare_v4(void *key_tgt, void *key_ent);
static int neigh_key_compare_v6(void *key_tgt, void *key_ent);

struct neigh_table *neigh_alloc(struct ixmap_desc *desc, int family)
{
	struct neigh_table *neigh;
	cudaError_t ret_cuda;

	ret_cuda = cudaMallocManaged((void **)&neigh,
		sizeof(struct neigh_table), cudaMemAttachGlobal);
	if(ret_cuda != cudaSuccess)
		goto err_neigh_alloc;

	hash_init(&neigh->table);
	neigh->table.hash_entry_delete = neigh_entry_delete;

	switch(family){
	case AF_INET:
		neigh->table.hash_key_generate =
			neigh_key_generate_v4;
		neigh->table.hash_key_compare =
			neigh_key_compare_v4;
		break;
	case AF_INET6:
		neigh->table.hash_key_generate =
			neigh_key_generate_v6;
		neigh->table.hash_key_compare =
			neigh_key_compare_v6;
		break;
	default:
		goto err_invalid_family;
		break;
	}

	return neigh;

err_invalid_family:
	cudaFree(neigh);
err_neigh_alloc:
	return NULL;
}

void neigh_release(struct neigh_table *neigh)
{
	hash_delete_all(&neigh->table);
	cudaFree(neigh);
	return;
}

static void neigh_entry_delete(struct hash_entry *entry)
{
	struct neigh_entry *neigh_entry;

	neigh_entry = hash_entry(entry, struct neigh_entry, hash);
	cudaFree(neigh_entry);
	return;
}

static unsigned int neigh_key_generate_v4(void *key, unsigned int bit_len)
{
	/* On some cpus multiply is faster, on others gcc will do shifts */
	uint32_t hash = *((uint32_t *)key) * GOLDEN_RATIO_PRIME_32;

	/* High bits are more random, so use them. */
	return hash >> (32 - bit_len);
}

static unsigned int neigh_key_generate_v6(void *key, unsigned int bit_len)
{
	uint64_t hash = ((uint64_t *)key)[0] * ((uint64_t *)key)[1];
	hash *= GOLDEN_RATIO_PRIME_64;

	return hash >> (64 - bit_len);
}

static int neigh_key_compare_v4(void *key_tgt, void *key_ent)
{
	return ((uint32_t *)key_tgt)[0] ^ ((uint32_t *)key_ent)[0] ?
		1 : 0;
}

static int neigh_key_compare_v6(void *key_tgt, void *key_ent)
{
	return ((uint64_t *)key_tgt)[0] ^ ((uint64_t *)key_ent)[0] ?
		1 : (((uint64_t *)key_tgt)[1] ^ ((uint64_t *)key_ent)[1] ?
		1 : 0);
}

int neigh_add(struct neigh_table *neigh, int family,
	void *dst_addr, void *mac_addr, struct ixmap_desc *desc) 
{
	struct neigh_entry *neigh_entry;
	cudaError_t ret_cuda;
	int ret;

	ret_cuda = cudaMallocManaged((void **)&neigh_entry,
		sizeof(struct neigh_entry), cudaMemAttachGlobal);
	if(ret_cuda != cudaSuccess)
		goto err_alloc_entry;

	memcpy(neigh_entry->dst_mac, mac_addr, ETH_ALEN);
	switch(family){
	case AF_INET:
		memcpy(neigh_entry->dst_addr, dst_addr, 4);
		break;
	case AF_INET6:
		memcpy(neigh_entry->dst_addr, dst_addr, 16);
		break;
	default:
		goto err_invalid_family;
		break;
	}

#ifdef DEBUG
	neigh_add_print(family, dst_addr, mac_addr);
#endif

	ret = hash_add(&neigh->table, neigh_entry->dst_addr, &neigh_entry->hash);
	if(ret < 0)
		goto err_hash_add;

	return 0;

err_hash_add:
err_invalid_family:
	cudaFree(neigh_entry);
err_alloc_entry:
	return -1;
}

int neigh_delete(struct neigh_table *neigh, int family,
	void *dst_addr)
{
	int ret;

#ifdef DEBUG
	neigh_delete_print(family, dst_addr);
#endif

	ret = hash_delete(&neigh->table, dst_addr);
	if(ret < 0)
		goto err_hash_delete;

	return 0;

err_hash_delete:
	return -1;
}

