/*
 ------------------------------------------------------------------
 ** Top contributors:
 **   Shiqi Wang and Suman Jana
 ** This file is part of the ReluVal project.
 ** Copyright (c) 2018-2019 by the authors listed in the file LICENSE
 ** and their institutional affiliations.
 ** All rights reserved.
 -----------------------------------------------------------------
 */

#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <signal.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include "nnet.h"
#include "prop.h"

#ifndef SPLIT_H
#define SPLIT_H

/* Rates of certified, falsified, and uncertain regions */
extern float rateCert, rateFals, rateUncer, rateAdv;

/* Time record */
extern struct timeval start, curr, finish, last_finish;

/*
 * Check the concrete adversarial examples of 
 * the middle point of given input ranges.
 */
int check_adv(struct NNet *nnet, struct Subproblem *subp);

#endif