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
    double omega;
    double x;
    double D;
    double W;
};

void generateGrid(std::vector<GridPoint>& grid, std::size_t degree,
    std::vector<Band>& freqBands, std::size_t density = 16u,
    mp_prec_t prec = 165ul);


void computeDenseNorm(double& normValue,
    std::vector<double>& bandNorms,
    std::vector<Band>& chebyBands, std::vector<GridPoint>& grid,
    std::vector<double>& a);

#endif  // GRID_H
