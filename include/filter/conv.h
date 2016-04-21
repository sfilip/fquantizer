/*
 * conv.h
 *
 *  Created on: Sep 26, 2013
 *      Author: silviu
 */

#ifndef CONV_H_
#define CONV_H_

#include "util.h"

/*! Performs a rounding operation from a double precision floating point
 * number to a fixed point value
 * @param doubleValue the value to round
 * @param intBits number of bits used to store the integer part of the
 * fixed point number
 * @param fracBits number of bits used to store the fractional part of
 * the fixed point number
 * @return the rounded value in fixed point
 */
double doubleToFixedPoint(double doubleValue, int intBits, int fracBits);

/*! Performs a rounding operation from a multiple precision floating point
 * number to a fixed point value
 * @param value the value to round
 * @param intBits number of bits used to store the integer part of the
 * fixed point number
 * @param fracBits number of bits used to store the fractional part of
 * the fixed point number
 * @return the rounded value in fixed point as a fractional number
 */
std::pair<long, long> mprealToFixedPoint(mpfr::mpreal& value, int intBits,
        int fracBits);

std::pair<std::pair<int, int>, std::pair<long, long>>
                        findBestFormatMP(mpfr::mpreal &value, int size);
std::pair<std::pair<int, int>, double>
                        findBestFormatD(double &value, int size);
std::pair<std::pair<int, int>, double>
                        defaultFormatD(double &value, int size);
std::pair<std::pair<int, int>, double>
                        customFormatD(double &value, int intBits, int fracBits);

/*! Rounds and formats a mpreal variable so that it can be printed in a
 * human readable form (offers a finer control than the printing functions
 * supplied by the mpreal interface)
 * @param data the variable to be converted
 * @param base the base in which the output will be represented (default is
 * decimal)
 * @param precision number of digits used to represent the output (default is
 * 30 digits)
 * @return the string represented the formatted mpfr::mpreal variable
 */
std::string mprealToString(mpfr::mpreal &data, int base = 10,
        std::size_t precision = 30);

/*! Transforms a given mpreal variable in its corresponding rational number
 * equivalent, which will be stored in a mpq_t variable
 * @param[out] ratVal the rational value of the given number
 * @param[in]  mprealVal the mpreal value to be converted
 */
void mprealToMpq(mpq_t& ratVal, mpfr::mpreal& mprealVal);

/*! Transforms a given mpreal variable in its corresponding mantissa-exponent
 * pair, meaning x = M * 2^e
 * @param[out] M the mantissa value
 * @param[out] e the exponent value
 * @param[in]  x the mpreal value we wish to decompose
 */
void mantissaExponentDecomposition(mpz_t& M, mp_exp_t& e,
        mpfr::mpreal &x);

void mprealToFixedPoint(std::vector<mpfr::mpreal>& naiveA,
        std::vector<mpfr::mpreal>& a, int intBits, int fracBits);

#endif /* CONV_H_ */
