/**
 * @file afp.h
 * @author Silviu Filip
 * @date 21 April 2016
 * @brief Utilities for performing the search for approximate Fekete points
 * */

#ifndef AFP_H_
#define AFP_H_

#include "band.h"
#include "cheby.h"
#include "util.h"

typedef Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, Eigen::Dynamic> MatrixXq;
typedef Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> VectorXq;

void generateAFPMatrix(
    MatrixXq &A, std::size_t degree, std::vector<mpfr::mpreal> &meshPoints,
    std::function<mpfr::mpreal(mpfr::mpreal)> &weightFunction);

// approximate Fekete points
void AFP(std::vector<mpfr::mpreal> &points, MatrixXq &A,
         std::vector<mpfr::mpreal> &meshPoints);

void generateVandermondeMatrix(
    MatrixXq &Vm, std::vector<mpfr::mpreal> &grid,
    std::function<mpfr::mpreal(mpfr::mpreal)> &weightFunction,
    std::size_t degree, mp_prec_t prec = 165u);

void linspace(std::vector<mpfr::mpreal> &points, mpfr::mpreal &a,
              mpfr::mpreal &b, std::size_t N, mp_prec_t prec = 165u);

void bandCount(std::vector<Band> &chebyBands, std::vector<mpfr::mpreal> &x);

void chebyMeshGeneration(std::vector<mpfr::mpreal> &chebyMesh,
                         std::vector<Band> &chebyBands, std::size_t degree,
                         mp_prec_t prec = 165u);

#endif /* AFP_H_ */
