#ifndef PROP_H
#define PROP_H

#include "matrix.h"
#include "interval.h"
#include "nnet.h"


/* read the weights and store in the equation intervals */
void affineTransform(struct Interval *interval, struct Matrix *posMatrix, struct Matrix *negMatrix, struct Interval *outInterval, int overWrite);

/* computes the concrete bounds based on the symbolic linear equation */
void computeAllBounds(float *eqLow, float *eqUp, struct Interval *input, int inputSize, float *low, float *lowsUp, float *upsLow, float *up);

/* zeroes out the inactive intervals */
void zero_interval(struct Interval *interval, int eqSize, int neuron);

/* Uses sgemm to calculate the output */
int forward_prop(struct NNet *network, struct Matrix *input, struct Matrix *output);

/* The back prop to calculate the gradient */
void backward_prop(struct NNet *nnet, struct Interval *grad, int R[][nnet->maxLayerSize], int y);

/* ReluVal / Neurify forward propagation method adapted for FairVal */
void forward_prop_fair(struct NNet *nnet, struct Interval *input, struct Interval *output, int (*R)[nnet->maxLayerSize]);

#endif