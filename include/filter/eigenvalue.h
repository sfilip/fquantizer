/**
 * @file eigenvalue.h
 * @author Silviu Filip
 * @date 12 March 2015
 * @brief Functions used to generate eigenvalues of colleague matrices
 *
 */

#ifndef EIGENVALUE_H_
#define EIGENVALUE_H_

#include "util.h"
#include "cheby.h"
#include "barycentric.h"

/** Eigen matrix container for mpfr::mpreal values */
typedef Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, Eigen::Dynamic> MatrixXq;
/** Eigen vector container for mpfr::mpreal values */
typedef Eigen::Matrix<std::complex<mpfr::mpreal>, Eigen::Dynamic, 1> VectorXcq;

/*! Funtion that generates the colleague matrix for a Chebyshev interpolant
 * expressed using the basis of Chebyshev polynomials of the first kind
 * (WITH or WITHOUT balancing in the vein of [Parlett&Reinsch1969] "Balancing a Matrix for
 * Calculation of Eigenvalues and Eigenvectors")
 * @param[out] C the corresponding colleague matrix
 * @param[in] a the coefficients of the Chebyshev interpolant
 * @param[in] withBalancing perform a balancing operation on the colleague matrix C
 * @param[in] prec the working precision used for the computations
 */
void generateColleagueMatrix1stKind(MatrixXq& C,
        std::vector<mpfr::mpreal>& a,
        bool withBalancing,
        mp_prec_t prec = 165ul);

void generateColleagueMatrix2ndKind(MatrixXq& C,
        std::vector<mpfr::mpreal>& a,
        bool withBalancing,
        mp_prec_t prec = 165ul);

/*! Funtion that generates the colleague matrix for a Chebyshev interpolant
 * expressed using the basis of Chebyshev polynomials of the second kind
 * (WITH or WITHOUT balancing in the vein of [Parlett&Reinsch1969] "Balancing a Matrix for
 * Calculation of Eigenvalues and Eigenvectors")
 * @param[out] C the corresponding colleague matrix
 * @param[in] a the coefficients of the Chebyshev interpolant
 * @param[in] withBalancing perform a balancing operation on the colleague matrix C
 * @param[in] prec the working precision used for the computations
 */
void generateColleagueMatrix2ndKindWithBalancing(MatrixXq& C,
        std::vector<mpfr::mpreal>& a,
        mp_prec_t prec = 165ul);

/*! Function that computes the eigenvalues of a given matrix
 * @param[out] eigenvalues the computed eigenvalues
 * @param[in] C the corresponding matrix
 */
void determineEigenvalues(VectorXcq &eigenvalues,
        MatrixXq &C);

/*! Function that computes the real values located inside
 * an interval \f$[a,b]\f$ from a vector of complex values
 * @param[out] realValues the set of real values inside \f$[a,b]\f$
 * @param[in] complexValues the set of complex values to search from
 * @param[in] a left side of the closed interval
 * @param[in] b right side of the closed interval
 */
void getRealValues(std::vector<mpfr::mpreal> &realValues,
        VectorXcq &complexValues,
        mpfr::mpreal &a, mpfr::mpreal &b);


/** Eigen matrix container for double values */
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
/** Eigen vector container for double values */
typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> VectorXcd;


/*! Function that generates the colleague matrix for a Chebyshev interpolant
 * expressed using the basis of Chebyshev polynomials of the first kind
 * (WITH or WITHOUT balancing in the vein of [Parlett&Reinsch1969] "Balancing a Matrix for
 * Calculation of Eigenvalues and Eigenvectors")
 * @param[out] C the corresponding colleague matrix
 * @param[in] a the coefficients of the Chebyshev interpolant
 * @param[in] withBalancing perform a balancing operation on the colleague matrix C
 */
void generateColleagueMatrix1stKind(MatrixXd& C,
        std::vector<double>& a, bool withBalancing = true);

/*! Function that generates the colleague matrix for a Chebyshev interpolant
 * expressed using the basis of Chebyshev polynomials of the second kind
 * (WITH or WITHOUT balancing)
 * @param[out] C the corresponding colleague matrix
 * @param[in] a the coefficients of the Chebyshev interpolant
 * @param[in] withBalancing perform a balancing operation on the colleague matrix C
 */
void generateColleagueMatrix2ndKind(MatrixXd& C,
        std::vector<double>& a, bool withBalancing = true);


/*! Function that computes the eigenvalues of a given matrix
 * @param[out] eigenvalues the computed eigenvalues
 * @param[in] C the corresponding matrix
 */
void determineEigenvalues(VectorXcd &eigenvalues,
        MatrixXd &C);

/*! Function that computes the real values located inside
 * an interval \f$[a,b]\f$ from a vector of complex values
 * @param[out] realValues the set of real values inside \f$[a,b]\f$
 * @param[in] complexValues the set of complex values to search from
 * @param[in] a left side of the closed interval
 * @param[in] b right side of the closed interval
 */
void getRealValues(std::vector<double> &realValues,
        VectorXcd &complexValues,
        double &a, double &b);


#endif
