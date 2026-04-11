#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//OR GATE
float or_training_set[][3] = { //Dataset(Description of how the model is supposed to behave)
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

#define set_count (sizeof(or_training_set)/sizeof(or_training_set[0]))

float cost(float w1, float w2)
{
    float error_result = 0.0f;
    for (size_t i = 0; i < set_count; i++)
    {
        float x1 = or_training_set[i][0];
        float x2 = or_training_set[i][1];
        float y = (x1*w1) + (x2*w2);

        float d = y - or_training_set[i][2];

        error_result += d*d;

    }
    error_result /= set_count;
    return error_result;
}

float random_float()
{

    return (float) rand() / (float)RAND_MAX;
}


int main()
{

    srand(time(0));

    float w1 = random_float();
    float w2 = random_float();
    float epsilon = 1e-3;
    float rate = 1e-3;

    for (size_t i = 0; i < 300; i++)
    {
        float c = cost(w1, w2);
        printf("Untrained Weight 1: %f, Untrained Weight 2: %f, Error_cost: %f\n", w1, w2, c);
        float dw1 = (cost(w1 + epsilon, w2) - c)/epsilon;
        float dw2 = (cost(w1, w2 + epsilon) - c)/epsilon;
        w1 -= rate*dw1;
        w2 -= rate*dw2;
    }

}



