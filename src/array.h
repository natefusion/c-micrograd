#ifndef ARRAY_H
#define ARRAY_H

#include <assert.h>

#define typeof_element(_struct,el) typeof(((_struct *)(0))->el)
#define Array_define(T, name) typedef struct { T *data; int len; } name
#define Array_literal(name, ...) (name) { .data = (typeof_element((name)))__VA_ARGS__, .len = sizeof((typeof_element((name)))__VA_ARGS__)/sizeof((type)__VA_ARGS__[0]) }

#define len(x) (maybe_ref((x))->len)
#define data(x) (maybe_ref((x))->data)

#define at(v, index) data((v))[(void)assert((index) < len((v))), (index)]

#define is_pointer(p)  (__builtin_classify_type(p) == 5)
#define maybe_ref(x) __builtin_choose_expr(is_pointer((x)), (x), &(x))

#define TYPE_TO_FORMAT_SPECIFIER(type)          \
    _Generic((type),                            \
             int: "%d",                         \
             float: "%f",                       \
             double: "%f",                      \
             default: "%d"                      \
        )


#define Array_print(array)                                              \
    do {                                                                \
        printf("{");                                                    \
        int _len = len((array));                                        \
        for (int i = 0; i < _len; ++i) {                                \
            printf(TYPE_TO_FORMAT_SPECIFIER(data(array)[0]), at((array), i)); \
            if (i < _len - 1) {                                         \
                printf(", ");                                           \
            }                                                           \
        }                                                               \
        printf("}\n");                                                  \
    } while (0)


#endif // ARRAY_H
