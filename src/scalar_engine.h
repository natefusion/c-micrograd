#ifdef TENSOR_ENGINE_H
#error "You can't include both the scalar engine and the tensor engine"
#endif

#ifndef SCALAR_ENGINE_H
#define SCALAR_ENGINE_H

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "arena.h"

typedef struct Value Value;

typedef struct {
    double lhs;
    double rhs;
} Partial_Derivative;

typedef struct {
    Value *lhs;
    Value *rhs;
} Children;

enum Flags : uint8_t {
    Flags_visited = 0b00000001,
    Flags_arity_1 = 0b00000010,
    Flags_arity_2 = 0b00000100,
};

struct Value {
    double data;
    Partial_Derivative (*pd_fn)(Value*);
    double grad;
    Children children;
    enum Flags flags;
    // debugging
    char const *op;
    char const *name;
};

#define VALUE_(namae, value) Value* namae = value; namae->name = #namae

void   value_reset(void);
Value* value_make(Arena* arena, double number);
Value* value_add(Arena* arena, Value *a, Value *b);
Value* value_add(Arena *arena, Value *a, Value *b);
Value* value_mul(Arena *arena, Value *a, Value *b);
Value* value_sub(Arena *arena, Value *a, Value *b);
Value* value_div(Arena *arena, Value *a, Value *b);
Value* value_tanh(Arena *arena, Value *a);
Value* value_relu(Arena *arena, Value *a);
Value* value_expt(Arena *arena, Value *base, Value *power);

void value_backward(Arena *arena, Value *value);
void value_draw_computational_graph(Value *v);

#endif // SCALAR_ENGINE_H
