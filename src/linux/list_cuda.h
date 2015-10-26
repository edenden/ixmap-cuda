#ifndef _LINUX_LIST_H
#define _LINUX_LIST_H

struct list_head {
	struct list_head *next, *prev;
};

struct hlist_head {
	struct hlist_node *first;
};

struct hlist_node {
	struct hlist_node *next, **pprev;
};

#define list_entry(ptr, type, member) \
	container_of(ptr, type, member)

#define list_first_entry(ptr, type, member) \
	list_entry((ptr)->next, type, member)

#define list_first_entry_or_null(ptr, type, member) \
	(!list_empty_cuda(ptr) ? list_first_entry(ptr, type, member) : NULL)

#define hlist_entry(ptr, type, member) container_of(ptr,type,member)

#define hlist_entry_safe(ptr, type, member)				\
	({ typeof(ptr) ____ptr = (ptr);					\
		____ptr ? hlist_entry(____ptr, type, member) : NULL;	\
	})

#define hlist_for_each_entry(pos, head, member) \
	for (pos = hlist_entry_safe((head)->first, typeof(*(pos)), member);	\
	pos;									\
	pos = hlist_entry_safe((pos)->member.next, typeof(*(pos)), member))

#endif
