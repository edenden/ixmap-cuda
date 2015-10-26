#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>

extern "C" {
#include "linux/list_cuda.h"
#include "main.h"
#include "hash.h"
}

extern "C"
__device__ struct hash_entry *hash_lookup(struct hash_table *table,
	void *key)
{
	struct hlist_head *head;
	struct hash_entry *entry, *entry_ret;
	unsigned int hash_key;

	entry_ret = NULL;
	hash_key = table->hash_key_generate(key, HASH_BIT);
	head = &table->head[hash_key];

	hlist_for_each_entry(entry, head, list){
		if(!table->hash_key_compare(key, entry->key)){
			entry_ret = entry;
			break;
		}
	}

        return entry_ret;
}

