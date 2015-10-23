#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <netinet/ip.h>
#include <arpa/inet.h>
#include <stddef.h>
#include <ixmap.h>

#include "linux/list.h"
#include "main.h"
#include "fib.h"

static int fib_entry_identify(void *ptr, unsigned int id,
	unsigned int prefix_len);
static int fib_entry_compare(void *ptr, unsigned int prefix_len);
static void fib_entry_pull(void *ptr);
static void fib_entry_put(void *ptr);

struct fib *fib_alloc(struct ixmap_desc *desc)
{
        struct fib *fib;
	struct ixmap_marea *area;

	area = ixmap_mem_alloc(desc, sizeof(struct fib));
	if(!area)
		goto err_fib_alloc;

	fib = area->ptr;
	fib->area = area;

	lpm_init(&fib->table);

	fib->table.entry_identify	= fib_entry_identify;
	fib->table.entry_compare	= fib_entry_compare;
	fib->table.entry_pull		= fib_entry_pull;
	fib->table.entry_put		= fib_entry_put;

	return fib;

err_fib_alloc:
	return NULL;
}

void fib_release(struct fib *fib)
{
	lpm_delete_all(&fib->table);
	ixmap_mem_free(fib->area);
	return;
}

int fib_route_update(struct fib *fib, int family, enum fib_type type,
	void *prefix, unsigned int prefix_len, void *nexthop,
	int port_index, int id, struct ixmap_desc *desc)
{
	struct fib_entry *entry;
	struct ixmap_marea *area;
	int ret;

	area = ixmap_mem_alloc(desc, sizeof(struct fib_entry));
	if(!area)
		goto err_alloc_entry;

	entry = area->ptr;
	entry->area = area;

	switch(family){
	case AF_INET:
		memcpy(entry->nexthop, nexthop, 4);
		memcpy(entry->prefix, prefix, 4);
		break;
	case AF_INET6:
		memcpy(entry->nexthop, nexthop, 16);
		memcpy(entry->prefix, prefix, 16);
		break;
	default:
		goto err_invalid_family;
		break;
	}

	entry->prefix_len	= prefix_len;
	entry->port_index	= port_index;
	entry->type		= type;
	entry->id		= id;
	entry->refcount		= 0;

#ifdef DEBUG
	fib_update_print(family, type, prefix, prefix_len,
		nexthop, port_index, id);
#endif

	ret = lpm_add(&fib->table, prefix, prefix_len,
		id, entry, desc);
	if(ret < 0)
		goto err_lpm_add;

	return 0;

err_lpm_add:
err_invalid_family:
	ixmap_mem_free(entry->area);
err_alloc_entry:
	return -1;
}

int fib_route_delete(struct fib *fib, int family,
	void *prefix, unsigned int prefix_len,
	int id)
{
	int ret;

#ifdef DEBUG
	fib_delete_print(family, prefix, prefix_len, id);
#endif

	ret = lpm_delete(&fib->table, prefix, prefix_len, id);
	if(ret < 0)
		goto err_lpm_delete;

	return 0;

err_lpm_delete:
	return -1;
}

static int fib_entry_identify(void *ptr, unsigned int id,
	unsigned int prefix_len)
{
	struct fib_entry *entry;

	entry = ptr;

	if(entry->id == id
	&& entry->prefix_len == prefix_len){
		return 0;
	}else{
		return 1;
	}
}

static int fib_entry_compare(void *ptr, unsigned int prefix_len)
{
	struct fib_entry *entry;

	entry = ptr;

	return entry->prefix_len > prefix_len ?
		1 : 0;
}

static void fib_entry_pull(void *ptr)
{
	struct fib_entry *entry;

	entry = ptr;
	entry->refcount++;

	return;
}

static void fib_entry_put(void *ptr)
{
	struct fib_entry *entry;

	entry = ptr;
	entry->refcount--;

	if(!entry->refcount){
		ixmap_mem_free(entry->area);
	}
}
