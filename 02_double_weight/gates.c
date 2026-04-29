#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>




//Function to classify output of the linear equation. Forces
//the output of the linear equation between 0.0 and 1.0
float sigmoid(float x)

{
    return 1.f/(1.f + expf(-x));
}


//Function to generate a random float for the weights and the bias
float random_float()
{
    //Dividing the generated number by RAND_MAX forces the production of a
    //floating point number between 0 and 1.
    return (float) rand() / (float)RAND_MAX;
    //RAND_MAX - Essentially the largest number that rand can produce though
}



//COST FUNCTION - calculates how wrong the model is
float cost(float w1, float w2, float b, float training_set[][3], size_t set_count)
{
    float error_result = 0.0f;

    //Calculate how wrong the model is for each input, adds the errors together
    for (size_t i = 0; i < set_count; i++)
    {
        float x1 = training_set[i][0];
        float x2 = training_set[i][1];
        float y = sigmoid((x1*w1) + (x2*w2) + b);
        float d = y - training_set[i][2];

        error_result += d*d;

    }

    //This then averages it by dividing the accumulated error by the number of inputs
    error_result /= set_count;
    return error_result;
}


int classifier(float x1, float x2, float w1, float w2, float b)
{
    float classer = sigmoid((x1*w1) + (x2*w2) + b);

    if (classer >= 0.5)
    {
        return 1;
    }

    else 
    {
        return 0;
    }
}




//Main function
int main()
{

    float training_set[][3] = { //Dataset(Description of how the model is supposed to behave)
        {0, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 1},
    };

    size_t length = sizeof(training_set)/sizeof(training_set[0]);

    

    //seed rand to produce new numbers everytime the program runs
    srand(time(0));


    //Initiate weights and bias and nudger and learning rate
    float w1 = random_float();
    float w2 = random_float();
    float b = random_float();
    float epsilon = 1e-1;
    float rate = 1e-1;
    float c;


    //Gradient descent: Continuously nudge the weights and bias to get derivatives and then continuously
    //toggle the weights and bias until model conforms with data
    for (size_t i = 0; i < 320000; i++)
    {
        c = cost(w1, w2, b, training_set, length);
        //calculate numerical derivative for weights and bias and nudges them based on those derivatives
        float dw1 = (cost(w1 + epsilon, w2, b, training_set, length) - c)/epsilon;
        float dw2 = (cost(w1, w2 + epsilon, b, training_set, length) - c)/epsilon;
        float db = (cost(w1, w2, b+epsilon, training_set, length) - c)/epsilon;
        w1 -= rate*dw1;
        w2 -= rate*dw2;
        b -= rate*db;
        if (i % 1000 == 0)
        {
            printf("Training Weight 1: %f, Training Weight 2: %f, Training Bias: %f, Error_cost: %f\n", w1, w2, b, c);
        }
        
    }

    printf("Trained Weight 1: %f, Trained Weight 2: %f, Trained Bias: %f,  Error_cost: %f\n", w1, w2, b, c);

    printf("\n");
    for (size_t i = 0; i < length; i++)
    {
        float x1 = training_set[i][0];
        float x2 = training_set[i][1];
        printf("%f | %f = %d\n", x1, x2, classifier(x1, x2, w1, w2, b));
    }


    return 0;

}



