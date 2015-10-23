#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <netinet/ip.h>
#include <arpa/inet.h>
#include <stddef.h>

extern "C" {
#include "linux/list_cuda.h"
#include "fib.h"
#include "lpm.h"
#include "fib_cuda.h"
#include "lpm_cuda.h"
}

__device__ struct fib_entry *fib_lookup(struct fib *fib, void *destination)
{
	struct lpm_entry *entry;

	entry = lpm_lookup(&fib->table, destination);
	if(!entry)
		goto err_lpm_lookup;

	return entry->ptr;

err_lpm_lookup:
	return NULL;
}

