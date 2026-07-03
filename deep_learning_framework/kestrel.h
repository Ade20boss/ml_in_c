#ifndef KESTREL_H_
#define KESTREL_H_


#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#ifndef KESTREL_ASSERT
#include <assert.h>
#define KESTREL_ASSERT assert
#endif // KESTREL_ASSERT

#define MATRIX_PRINT(m) print_matrix(m, #m)
#define MAT_POS(m, r, c) ((m).es[(r) * (m).stride + (c)])

typedef struct
{
    size_t offset;
    size_t capacity;
    uint8_t *data;

} Chunk_memory;

uint8_t *arena_init(Chunk_memory *arena);
void *custom_alloc(size_t size, Chunk_memory *arena);
int arena_reset(Chunk_memory *arena);
int custom_free(Chunk_memory *arena);

typedef struct
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;

} matrix;


matrix matrix_allocate(Chunk_memory *arena, size_t rows, size_t cols);
void matrix_randomize(matrix matrix, float low, float high);
void matrix_fill(matrix matrix, float value);
void matrix_dotproduct(matrix destination, matrix source_a, matrix source_b);
void matrix_activate(matrix matrix);
matrix row_matricize(matrix matrix, size_t row_num);
matrix column_matricize(matrix matrix, size_t col_num);
void matrix_copy(matrix destination, matrix source);
void matrix_sum(matrix destination, matrix source);
void print_matrix(matrix matrix, const char *name);
float rand_float(void);
float sigmoid(float value);


#endif //KESTREL_H_


#ifdef KESTREL_CODE

uint8_t *arena_init(Chunk_memory *arena)
{
    if (arena->data == NULL)
    {
        arena->data = (uint8_t *)malloc(sizeof(uint8_t) * arena->capacity);
        arena->offset = 0;

        if (arena->data == NULL)
        {
            return NULL;
        }
    }
    return arena->data;
}


void *custom_alloc(size_t size, Chunk_memory *arena)
{
    if (arena->offset + size > arena->capacity)
    {
        return NULL;
    }

    uint8_t *ptr = arena->data + arena->offset;
    arena->offset += size;
    return (void *)ptr;
}

int arena_reset(Chunk_memory *arena)
{
    if (arena->data == NULL)
    {
        return 1;
    }

    else
    {
        arena->offset = 0;
        return 0;
    }
}

int custom_free(Chunk_memory *arena)
{
    if (arena->data == NULL)
    {
        return 1;
    }
    else
    {
        free(arena->data);
        arena->data = NULL;
        arena->offset = 0;
        arena->capacity = 0;
    }
    return 0;
}


float rand_float()
{
    return (float)rand() / (float)RAND_MAX;
}


matrix matrix_allocate(Chunk_memory *arena, size_t rows, size_t cols)
{
    matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = custom_alloc(rows * cols * sizeof(*m.es), arena);
    KESTREL_ASSERT(m.es != NULL);
    return m;
}


void matrix_fill(matrix matrix, float value)
{
    for (size_t i = 0; i < matrix.rows; i++)
    {
        for (size_t j = 0; j < matrix.cols; j++)
        {
            MAT_POS(matrix, i, j) = value;
        }
    }
}


void matrix_dotproduct(matrix destination, matrix source_a, matrix source_b)
{
    // for matrix multiplication to work, columns of a must equal rows of b
    // otherwise the elements don't line up and the whole thing falls apart
    KESTREL_ASSERT(source_a.cols == source_b.rows);

    // the resulting matrix always has the same rows as a and same cols as b
    // if destination isn't that shape, we're writing into the wrong place
    KESTREL_ASSERT(destination.rows == source_a.rows && destination.cols == source_b.cols);

    // zero out destination first, because we're using +=
    // if we don't, we're just accumulating on top of garbage
    matrix_fill(destination, 0.0f);

    // locks us into each row of the result, one at a time
    for (size_t i = 0; i < destination.rows; i++)
    {
        // locks us into each column of the result
        // at this point (i, j) is the exact cell we're about to compute
        for (size_t j = 0; j < destination.cols; j++)
        {
            // this is the actual accumulation engine
            // once we're locked into the right row and the right element,
            // k does two things at once — it's the column index for source_a (source_a[i][k])
            // and the row index for source_b (source_b[k][j]) at the same time
            // so as k increases, it walks across the columns of source_a
            // while simultaneously walking down the rows of source_b
            // multiplying the matching pair at each step and accumulating into destination[i][j]
            for (size_t k = 0; k < source_a.cols; k++)
            {
                // destination[i][j] += source_a[i][k] * source_b[k][j]
                // each matrix uses its own stride because they can have different layouts in memory
                MAT_POS(destination, i, j) += MAT_POS(source_a, i, k) * MAT_POS(source_b, k, j);
            }
        }
    }
}


matrix row_matricize(matrix m, size_t row_num)
{
    KESTREL_ASSERT(row_num < m.rows);
    return (matrix){.rows = 1, .cols = m.cols, .stride = m.stride, .es = &MAT_POS(m, row_num, 0)};
}


void matrix_copy(matrix destination, matrix source)
{
    KESTREL_ASSERT(destination.rows == source.rows && destination.cols == source.cols);
    for (size_t i = 0; i < destination.rows; i++)
    {
        for (size_t j = 0; j < destination.cols; j++)
        {
            MAT_POS(destination, i, j) = MAT_POS(source, i, j);
        }
    }
}

void matrix_sum_bias(matrix destination, matrix source)
{
    /*
    So what we're doing here essentially is adding the bias to the weighted sum
    When the layer computes the result, we just add the bias to every computed value
    */
    KESTREL_ASSERT(destination.cols == source.cols && source.rows == 1);
    for (size_t i = 0; i < destination.rows; i++)
    {
        for (size_t j = 0; j < destination.cols; j++)
        {
            MAT_POS(destination, i, j) += MAT_POS(source, 0, j);
        }
    }
}

void print_matrix(matrix m, const char *name)
{
    printf("%s = {", name);
    printf("\n");
    for (size_t i = 0; i < m.rows; i++)
    {
        printf("    [");
        for (size_t j = 0; j < m.cols; j++)
        {
            printf("%f", MAT_POS(m, i, j));

            if (j < m.cols - 1)
            {
                printf("    ");
            }
        }
        printf("]\n");
    }

    printf("}\n\n");
}


void matrix_randomize(matrix matrix, float low, float high)
{

    for (size_t i = 0; i < matrix.rows; i++)
    {
        for (size_t j = 0; j < matrix.cols; j++)
        {
            MAT_POS(matrix, i, j) = rand_float() * (high - low) + low;
        }
    }
}

float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}


void matrix_activate(matrix matrix)
{
    for (size_t i = 0; i < matrix.rows; i++)
    {
        for (size_t j = 0; j < matrix.cols; j++)
        {
            MAT_POS(matrix, i, j) = sigmoid(MAT_POS(matrix, i, j));
        }
    }
}


#endif //KESTREL_CODE
