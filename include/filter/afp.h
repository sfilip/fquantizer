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

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;

void generateAFPMatrix(
    MatrixXd &A, std::size_t degree, std::vector<double> &meshPoints,
    std::function<double(double)> &weightFunction);

// approximate Fekete points
void AFP(std::vector<double> &points, MatrixXd &A,
         std::vector<double> &meshPoints);

void generateVandermondeMatrix(
    MatrixXd &Vm, std::vector<double> &grid,
    std::function<double(double)> &weightFunction,
    std::size_t degree, mp_prec_t prec = 165u);

void linspace(std::vector<double> &points, double &a,
              double &b, std::size_t N);

void bandCount(std::vector<BandD> &chebyBands, std::vector<double> &x);

void chebyMeshGeneration(std::vector<double> &chebyMesh,
                         std::vector<BandD> &chebyBands, std::size_t degree);

#endif /* AFP_H_ */
