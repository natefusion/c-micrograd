#ifndef ARENA_H
#define ARENA_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *ptr; // holds ptr from malloc for freeing
    char *beg;
    char *end;
} Arena;

Arena Arena_init(ptrdiff_t cap);
void *alloc(Arena *a, ptrdiff_t size, ptrdiff_t align, ptrdiff_t count,
            int flags);

#define new(...)            newx(__VA_ARGS__,new4,new3,new2)(__VA_ARGS__)
#define newx(a,b,c,d,e,...) e
#define new2(a, t)          (t *)alloc(a, sizeof(t), _Alignof(t), 1, 0)
#define new3(a, t, n)       (t *)alloc(a, sizeof(t), _Alignof(t), n, 0)
#define new4(a, t, n, f) (t *)alloc(a, sizeof(t), _Alignof(t), n, f)

#endif // ARENA_H
