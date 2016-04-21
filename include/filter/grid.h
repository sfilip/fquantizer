/**
 * @file grid.h
 * @author Silviu Filip
 * @date 12 October 2015
 * @brief Functions for working on an arbitrary discretization of the
 * approximation domain
 *
 */

#ifndef GRID_H
#define GRID_H

#include "util.h"
#include "cheby.h"
#include "barycentric.h"

struct GridPoint
{
    mpfr::mpreal omega;
    mpfr::mpreal x;
    mpfr::mpreal D;
    mpfr::mpreal W;
};

void generateGrid(std::vector<GridPoint>& grid, std::size_t degree,
    std::vector<Band>& freqBands, std::size_t density = 16u,
    mp_prec_t prec = 165ul);

void getError(mpfr::mpreal& error, std::vector<Band>& chebyBands,
    GridPoint& p, std::vector<mpfr::mpreal>& a, mp_prec_t prec = 165ul);

void computeDenseNorm(mpfr::mpreal& normValue,
    std::vector<mpfr::mpreal>& bandNorms,
    std::vector<Band>& chebyBands, std::vector<GridPoint>& grid,
    std::vector<mpfr::mpreal>& a, mp_prec_t prec = 165ul);

#endif  // GRID_H
