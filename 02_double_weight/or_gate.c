#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//OR GATE
float or_training_set[][3] = { //Dataset(Description of how the model is supposed to behave)
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1},
};

#define set_count (sizeof(or_training_set)/sizeof(or_training_set[0]))


float sigmoid(float x)
{
    return 1.f/(1.f + expf(-x));
}

float cost(float w1, float w2, float b)
{
    float error_result = 0.0f;
    for (size_t i = 0; i < set_count; i++)
    {
        float x1 = or_training_set[i][0];
        float x2 = or_training_set[i][1];
        float y = sigmoid((x1*w1) + (x2*w2) + b);
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
    float b = random_float();
    float epsilon = 1e-1;
    float rate = 1e-1;
    float c;

    for (size_t i = 0; i < 320000; i++)
    {
        c = cost(w1, w2, b);
        float dw1 = (cost(w1 + epsilon, w2, b) - c)/epsilon;
        float dw2 = (cost(w1, w2 + epsilon, b) - c)/epsilon;
        float db = (cost(w1, w2, b+epsilon) - c)/epsilon;
        w1 -= rate*dw1;
        w2 -= rate*dw2;
        b -= rate*db;
        printf("Training Weight 1: %f, Training Weight 2: %f, Training Bias: %f, Error_cost: %f\n", w1, w2, b, c);
    }

    printf("Trained Weight 1: %f, Trained Weight 2: %f, Trained Bias: %f,  Error_cost: %f\n", w1, w2, b, c);

    printf("\n");
    for (size_t i = 0; i < set_count; i++)
    {
        float x1 = or_training_set[i][0];
        float x2 = or_training_set[i][1];
        printf("%f | %f = %f\n", x1, x2, sigmoid((x1*w1) + (x2*w2) + b));
    }

}



