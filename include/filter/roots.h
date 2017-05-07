/**
 * @file roots.h
 * @author Silviu Filip
 * @date 15 April 2015
 * @brief The routines the used for finding local extrema and roots of
 * error functions
 *
 */

#ifndef ROOTS_H
#define ROOTS_H

#include "util.h"
#include "cheby.h"
#include "barycentric.h"
#include "eigenvalue.h"

void findEigenZeros(std::vector<mpfr::mpreal> &a,
        std::vector<mpfr::mpreal> &zeros,
        std::vector<Band> &freqBands,
        std::vector<Band> &chebyBands,
        mpfr_prec_t prec = 165u);

void findEigenZeros(std::vector<mpfr::mpreal> &a,
                    std::vector<mpfr::mpreal> &zeros,
                    std::vector<mpfr::mpreal> &refs,
                    std::vector<Band> &freqBands,
                    std::vector<Band> &chebyBands,
                    mpfr_prec_t prec = 165u);

void findEigenExtremas(std::vector<mpfr::mpreal> &a,
        std::vector<mpfr::mpreal> &extremas,
        std::vector<Band> &freqBands,
        std::vector<Band> &chebyBands,
        mpfr_prec_t prec = 165u);

void getError(mpfr::mpreal &error, std::vector<Band> &chebyBands,
              mpfr::mpreal &x, std::vector<mpfr::mpreal> &a, mpfr_prec_t prec = 165u);

void computeNorm(std::pair<mpfr::mpreal, mpfr::mpreal>& norm,
        std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>>& bandNorms,
        std::vector<mpfr::mpreal>& a,
        std::vector<Band>& freqBands,
        std::vector<Band>& chebyBands,
        mpfr_prec_t prec = 165u);

void generateGridPoints(std::vector<mpfr::mpreal>& grid, std::size_t degree,
        std::vector<Band>& freqBands, mpfr_prec_t prec = 165u);


void computeDenseNorm(mpfr::mpreal& normValue, std::vector<Band>& chebyBands,
        std::vector<mpfr::mpreal>& a, std::vector<mpfr::mpreal>& grid,
        mpfr_prec_t prec = 165u);


void findEigenZeros(std::vector<double> &a,
        std::vector<double> &zeros,
        std::vector<BandD> &freqBands,
        std::vector<BandD> &chebyBands);

void findEigenZeros(std::vector<double> &a,
                    std::vector<double> &zeros,
                    std::vector<double> &refs,
                    std::vector<BandD> &freqBands,
                    std::vector<BandD> &chebyBands);

void findEigenExtremas(std::vector<double> &a,
        std::vector<double> &extremas,
        std::vector<BandD> &freqBands,
        std::vector<BandD> &chebyBands);

void computeNorm(std::pair<double, double>& norm,
        std::vector<std::pair<double, double>>& bandNorms,
        std::vector<double>& a,
        std::vector<BandD>& freqBands,
        std::vector<BandD>& chebyBands);

void generateGridPoints(std::vector<double>& grid, std::size_t degree,
        std::vector<BandD>& freqBands);


void computeDenseNorm(double& normValue, std::vector<BandD>& chebyBands,
        std::vector<double>& a, std::vector<double>& grid);

#endif // ROOTS_H
