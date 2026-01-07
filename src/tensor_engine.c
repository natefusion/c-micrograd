#include <stdarg.h>

#include "tensor_engine.h"

static Arena arena = {0};
static ptrdiff_t Mem = 0;
static char *arena_ptr = NULL;


// this should act like a circular buffer.
// maybe fuse this with the original
static Arena arena_temp = {0};
static ptrdiff_t Mem_temp = 1000;
static char *arena_temp_ptr = NULL;

static int Tensors = 0;
static Tensor** Topological_sorted_values = NULL;

void tensor_set_mem(ptrdiff_t bytes) {
    Mem = bytes;
}

void tensor_reset(void) {
    if (arena_ptr) {
        free(arena_ptr);
        arena = (Arena) {0};
    }
    
    arena = Arena_init(Mem);
    arena_ptr = arena.beg;

    if (arena_temp_ptr) {
        free(arena_temp_ptr);
        arena_temp = (Arena) {0};
    }

    arena_temp = Arena_init(Mem_temp);
    arena_temp_ptr = arena_temp.beg;


    Tensors = 0;
    Topological_sorted_values = NULL;
}

int tensor_numel(Array_int shape) {
    int elements = 0;
    for (int i = 0; i < len(shape); ++i) elements *= at(shape, i);
    return elements;
}

bool tensor_is_scalar(Tensor *v) {
    return tensor_numel(v->shape) == 1;
}

bool tensor_is_shape_eq(Array_int lhs, Array_int rhs) {
    if (len(lhs) != len(rhs)) return false;
    for (int i = 0; i < len(lhs); ++i) if (at(lhs, i) != at(rhs, i)) return false;
    return true;
}

// analagous to the CL function `row-major-aref`
int tensor_subscripts_to_row_major_index(Tensor *v, int subscript, ...) {
    int index = subscript;
    va_list args;
    va_start(args, subscript);
    for (int i = 0; i < len(v->shape); ++i) {
        int ith_subscript = va_arg(args, int);
        index += ith_subscript * at(v->shape, i);
    }
    va_end(args);

    return index;
}

Tensor* tensor_make_arena(Arena *arena, double number, Array_int shape) {
    Tensor* z = new(arena, Tensor);

    int elements = tensor_numel(shape);
    
    *z = (Tensor) {
        .data = new(arena, double, elements),
        .grad = new(arena, double, elements),
        .shape = shape,
    };

    // already zeroed
    if (number != 0.0) for (int i = 0; i < elements; ++i) z->data[i] = number;

    Tensors++;
    
    return z;
}

Tensor *tensor_make(double number, Array_int shape) {
    return tensor_make_arena(&arena, number, shape);
}

Tensor* tensor_make_temp(double number, Array_int shape) {
    return tensor_make_arena(&arena_temp, number, shape);
}

Tensor *tensor_make_with_data(Tensor *copy_from) {
    Tensor *z = tensor_make(0.0, copy_from->shape);
    memcpy(z->data, copy_from->data, tensor_numel(z->shape)*sizeof(z->data));
    return z;
}


double add_fn(double a, double b) { return a + b; }
double sub_fn(double a, double b) { return a - b; }
double mul_fn(double a, double b) { return a * b; }
double div_fn(double a, double b) { return a / b; }

double inv_fn(double a) { return 1.0 / a; }
double neg_fn(double a) { return -a; }

double relu(double a) { return a > 0.0 ? a : 0.0; }
double diff_relu(double a) { return a > 0.0 ? 1.0 : 0.0; }

void tensor_apply_scalar_right(Tensor *dest, Dyadic_Fn op, Tensor *a, double scalar) {
    for (int i = 0; i < tensor_numel(a->shape); ++i) tensor_at(dest, i) = op(tensor_at(a, i), scalar);
}

void tensor_apply_scalar_left(Tensor *dest, Dyadic_Fn op, double scalar, Tensor *a) {
    for (int i = 0; i < tensor_numel(a->shape); ++i) tensor_at(dest, i) = op(scalar, tensor_at(a, i));
}

void tensor_dyadic_apply(Tensor *dest, Dyadic_Fn op, Tensor *a, Tensor *b) {
    for (int i = 0; i < tensor_numel(a->shape); ++i) tensor_at(dest, i) = op(tensor_at(a, i), tensor_at(b, i));
}

void tensor_monadic_apply(Tensor *dest, Monadic_Fn op, Tensor* a) {
    for (int i = 0; i < tensor_numel(a->shape); ++i) tensor_at(dest, i) = op(tensor_at(a, i));
}

Tensor *tensor_element_wise_dyadic_operation(Dyadic_Fn op, Tensor *a, Tensor *b) {
    Tensor *z;
    
    if (0) {
    } else if (tensor_is_scalar(a)) {
        z = tensor_make(0.0, b->shape);
        double scalar = tensor_at(a, 0);

        tensor_apply_scalar_right(z, op, b, scalar);
    } else if (tensor_is_scalar(b)) {
        z = tensor_make(0.0, a->shape);
        double scalar = tensor_at(b, 0);
        tensor_apply_scalar_right(z, op, a, scalar);
    } else if (tensor_is_shape_eq(a->shape, b->shape)){
        z = tensor_make(0.0, a->shape);
        tensor_dyadic_apply(z, op, a, b);
    } else {
        abort();
    }

    z->children = (Children) {a, b};
    z->flags = Flags_arity_2;
    
    return z;
}

Tensor *tensor_element_wise_monadic_operation(Monadic_Fn op, Tensor *a) {
    Tensor *z = tensor_make(0.0, a->shape);
    tensor_monadic_apply(z, op, a);
    z->children = (Children) {a};
    z->flags = Flags_arity_1;
    
    return z;
}

Partial_Derivative tensor_add_pd(Tensor* v) {
    Tensor *scalar = tensor_make(1.0, v->shape);
    return (Partial_Derivative) {scalar, scalar};
}

Tensor* tensor_add(Tensor *a, Tensor *b) {
    Tensor *z = tensor_element_wise_dyadic_operation(add_fn, a, b);
    z->pd_fn = tensor_add_pd;
    z->op = "+";
    
    return z;
}

Partial_Derivative tensor_mul_pd(Tensor* v) {
    return (Partial_Derivative) {v->children.rhs, v->children.lhs};
}

Tensor* tensor_mul(Tensor *a, Tensor *b) {
    Tensor *z = tensor_element_wise_dyadic_operation(mul_fn, a, b);
    z->pd_fn = tensor_add_pd;
    z->op = "*";
    
    return z;
}

Partial_Derivative tensor_sub_pd(Tensor* v) {
    Tensor *scalar1 = tensor_make(1.0, v->shape);
    Tensor *scalar2 = tensor_make(-1.0, v->shape);
    return (Partial_Derivative) {scalar1, scalar2};
}

Tensor* tensor_sub(Tensor *a, Tensor *b) {
    Tensor *z = tensor_element_wise_dyadic_operation(sub_fn, a, b);
    z->pd_fn = tensor_sub_pd;
    z->op = "-";
    
    return z;
}

Partial_Derivative tensor_div_pd(Tensor* v) {
    Tensor *a = tensor_make(0.0, v->shape);
    tensor_monadic_apply(a, inv_fn, v->children.rhs);
    
    Tensor *b = tensor_make(0.0, v->shape);
    tensor_apply_scalar_right(b, pow, v->children.rhs, 2.0);
    tensor_dyadic_apply(b, div_fn, v->children.lhs, b);
    tensor_monadic_apply(b, neg_fn, b);

    return (Partial_Derivative) {a, b};
}

Tensor* tensor_div(Tensor *a, Tensor *b) {
    Tensor *z = tensor_element_wise_dyadic_operation(div_fn, a, b);
    z->pd_fn = tensor_div_pd;
    z->op = "/";
    
    return z;
}

Partial_Derivative tensor_tanh_pd(Tensor* v) {
    Tensor *a = tensor_make(0.0, v->shape);
    tensor_monadic_apply(a, tanh, v->children.lhs);
    tensor_apply_scalar_right(a, pow, a, 2.0);
    tensor_apply_scalar_left(a, sub_fn, 1.0, a);
    
    return (Partial_Derivative) {a};
}

Tensor* tensor_tanh(Tensor *a) {
    Tensor *z = tensor_element_wise_monadic_operation(tanh, a);
    z->pd_fn = tensor_tanh_pd;
    z->op = "tanh";
    
    return z;
}

Partial_Derivative tensor_relu_pd(Tensor* v) {
    Tensor *a = tensor_make(0.0, v->shape);
    tensor_monadic_apply(a, diff_relu, v->children.rhs);
    
    return (Partial_Derivative) {a};
}

Tensor* tensor_relu(Tensor *a) {
    Tensor *z = tensor_element_wise_monadic_operation(relu, a);
    z->pd_fn = tensor_relu_pd;
    z->op = "relu";
    
    return z;
}

Partial_Derivative tensor_expt_pd(Tensor* v) {
    Tensor *base = v->children.lhs;
    Tensor *power = v->children.rhs;

    Tensor *a = tensor_make(0.0, v->shape);
    Tensor *b = tensor_make(0.0, v->shape);
    Tensor *temp = tensor_make_temp(0.0, v->shape);

    tensor_dyadic_apply(b, pow, base, power);
    tensor_monadic_apply(temp, log, base);
    tensor_dyadic_apply(b, mul_fn, b, temp);

    tensor_apply_scalar_right(temp, sub_fn, power, 1.0);
    tensor_dyadic_apply(a, pow, base, temp);
    tensor_dyadic_apply(a, mul_fn, power, a);

    return (Partial_Derivative) {a, b};
}

Tensor* tensor_expt(Tensor *a, Tensor *power) {
    Tensor *z = tensor_element_wise_monadic_operation(relu, a);
    z->pd_fn = tensor_expt_pd;
    z->op = "expt";
    
    return z;
}

void tensor_topological_sort(Tensor **topo_array, int *offset_from_end, Tensor *value) {
    if ((value->flags & Flags_visited) != 0) {
        return;
    }

    value->flags |= Flags_visited;
    
    if ((value->flags & Flags_arity_1) != 0) {
        tensor_topological_sort(topo_array, offset_from_end, value->children.lhs);
    } else if ((value->flags & Flags_arity_2) != 0) {
        tensor_topological_sort(topo_array, offset_from_end, value->children.lhs);
        tensor_topological_sort(topo_array, offset_from_end, value->children.rhs);
    }

    topo_array[Tensors - 1 - *offset_from_end] = value;
    *offset_from_end += 1;
}

void tensor_backward(Tensor *value) {
    if (Topological_sorted_values == NULL) {
        Topological_sorted_values = new(&arena, Tensor*, Tensors);
        int offset_from_end = 0;
        tensor_topological_sort(Topological_sorted_values, &offset_from_end, value);
    }

    for (int i = 0; i < len(value->shape); ++i) value->grad[i] = 1.0;

    for (int i = 0; i < Tensors; ++i) {
        Tensor *v = Topological_sorted_values[i];
        if ((v->flags & Flags_arity_1) != 0) {
            Partial_Derivative pd = v->pd_fn(v);
            for (int i = 0; i < len(pd.lhs->shape); ++i) {
                v->children.lhs->grad[i] += tensor_at(pd.lhs, i) * v->grad[i];
            }
        } else if ((v->flags & Flags_arity_2) != 0) {
            Partial_Derivative pd = v->pd_fn(v);
            for (int i = 0; i < len(pd.lhs->shape); ++i) {
                v->children.lhs->grad[i] += tensor_at(pd.lhs, i) * v->grad[i];
            }

            for (int i = 0; i < len(pd.rhs->shape); ++i) {
                v->children.rhs->grad[i] += tensor_at(pd.rhs, i) * v->grad[i];
            }
        }
    }
}
