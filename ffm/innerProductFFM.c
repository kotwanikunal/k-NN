#include "innerProductFFM.h"

float innerprod_native(float* queryArr, float* inputArr, int length) {
    float sum = 0.0f;


    for (int i = 0; i < length; i++) {


        float acc = queryArr[i] * inputArr[i];


        sum += acc;


    }





    // scale due to lucene restrictions.


    if (


        sum < 0.0f


    ) {


        sum = 1 / (1 + -1 * sum);


    } else {


        sum += 1;


    }
    return sum;
}