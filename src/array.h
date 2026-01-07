#ifndef ARRAY_H
#define ARRAY_H

#include <assert.h>

#include "arena.h"

#define Array_define(T, name) typedef struct { T *data; int len; } name
#define Array_make(arena, T, initializer, size) (((T)) { .data = ((T))new((arena), (T), (size)), len = (size) })

/* #define Owned_Array_define(T, name, size) typedef struct { (T)[(size)] data; int len; } name */
/* #define Owned_Array_make(name, initializer) ((name)) { .data = (initializer), .len = sizeof((initializer)) } */

#define len(x) (maybe_ref((x))->len)
#define data(x) (maybe_ref((x))->data)

#define at(v, index) data((v))[(void)assert((index) < len((v))), (index)]


#define is_pointer(p)  (__builtin_classify_type(p) == 5)
#define maybe_ref(x) __builtin_choose_expr(is_pointer((x)), (x), &(x))

#endif // ARRAY_H
