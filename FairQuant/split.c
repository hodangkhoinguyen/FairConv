/*
 -----------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang and Suman Jana
 ** This file is part of the ReluVal project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 */

#include "split.h"

#define AVG_WINDOW 5
#define MAX_THREAD 56
#define MIN_DEPTH_PER_THREAD 5 

float rateCert = 0, rateFals = 0, rateAdv = 0, rateUncer = 1;

struct timeval start, curr, finish, last_finish;

/*
 * Check the existance of concrete adversarial examples
 * It takes in the network and input ranges.
 * If a concrete adversarial example is found,
 * return adv_found 1
 */
int check_adv(struct NNet* nnet, struct Subproblem *subp)
{

    float a0[nnet->inputSize];
    float a1[nnet->inputSize];
    struct Matrix adv0 = {a0, 1, nnet->inputSize};
    struct Matrix adv1 = {a1, 1, nnet->inputSize};

    // trying 10 different samples
    for (int n=0; n<10; n++){

        // concrete data point is gender + some point for all other features
        for (int i=0; i<nnet->inputSize; i++) {
            if (i == nnet->sens_feature_idx){
                a0[i] = nnet->mins[i];  //for PA=0
                a1[i] = nnet->maxes[i]; //for PA=1
            }
            else {
                int upper = (int) subp->input.upper_matrix.data[i];
                int lower = (int) subp->input.lower_matrix.data[i];
                int middle = n*(lower+upper)/10; // floor

                a0[i] = (float) middle;
                a1[i] = (float) middle;
            }
        }

        float out0[nnet->outputSize];
        float out1[nnet->outputSize];
        struct Matrix output0 = {out0, nnet->outputSize, 1};
        struct Matrix output1 = {out1, nnet->outputSize, 1};

        forward_prop(nnet, &adv0, &output0);
        forward_prop(nnet, &adv1, &output1);

        // for sigmoid, one output node
        int out0Pos = (output0.data[0] > 0);
        int out1Pos = (output1.data[0] > 0);

        // at any point, this sample is adv, then we can return here
        // and no need to check further
        if (out0Pos != out1Pos) {
            return 1;   //is_adv!
        } 
    }

    // if we made it here, that means no adv sample was found
    return 0;
}