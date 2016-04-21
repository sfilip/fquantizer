/**
 * @file fpminimax.h
 * @author Silviu Filip
 * @date 2 April 2015
 * @brief the core LLL-based quantization routines
 *
 */
#ifndef FPMINIMAX_H
#define FPMINIMAX_H

#include "util.h"
#include "roots.h"

void fpminimaxWithNeighborhoodSearch(
        mpfr::mpreal& minError,
        std::vector<mpfr::mpreal>& lllFreeA,
        std::vector<mpfr::mpreal>& freeA,
        std::vector<mpfr::mpreal>& fixedA,
        std::vector<mpfr::mpreal>& interpolationPoints,
        std::vector<Band>& freqBands,
        std::vector<mpfr::mpreal>& weights,
        mpfr::mpreal& scalingFactor,
        mp_prec_t prec = 165u);

void fpminimaxWithNeighborhoodSearchDiscrete(
        mpfr::mpreal& minError,
        std::vector<mpfr::mpreal>& lllFreeA,
        std::vector<mpfr::mpreal>& freeA,
        std::vector<mpfr::mpreal>& fixedA,
        std::vector<mpfr::mpreal>& interpolationPoints,
        std::vector<Band>& freqBands,
        std::vector<mpfr::mpreal>& weights,
        mpfr::mpreal& scalingFactor,
        mp_prec_t prec = 165ul);

void fpminimaxWithExactCVP(
        mpfr::mpreal& minError,
        std::vector<mpfr::mpreal>& lllFreeA,
        std::vector<mpfr::mpreal>& freeA,
        std::vector<mpfr::mpreal>& fixedA,
        std::vector<mpfr::mpreal>& interpolationPoints,
        std::vector<Band>& freqBands,
        std::vector<mpfr::mpreal>& weights,
        mpfr::mpreal& scalingFactor,
        mp_prec_t prec = 165ul);



#endif
