#include "arena.h"
#include <stdio.h>

Arena Arena_init(ptrdiff_t cap) {
    Arena a = {0};
    a.beg = (char*)malloc(cap);
    a.ptr = a.beg;
    a.end = a.beg ? a.beg+cap : 0;
    return a;
}

void *alloc(Arena *a, ptrdiff_t size, ptrdiff_t align, ptrdiff_t count, int flags) {
    ptrdiff_t padding = -(uintptr_t)a->beg & (align - 1);
    ptrdiff_t available = a->end - a->beg - padding;
    if (available < 0 || count > available/size) {
        fprintf(stderr, "You ran out of memory!!!\n"
                        "Tried to alloc %td more bytes\n", count*size);
        abort();  // one possible out-of-memory policy
    }
    void *p = a->beg + padding;
    a->beg += padding + count*size;
    return memset(p, 0, count*size);
}
