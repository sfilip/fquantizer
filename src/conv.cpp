/*
 * conv.cpp
 *
 *  Created on: Sep 26, 2013
 *      Author: silviu
 */

#include "filter/conv.h"

double doubleToFixedPoint(double doubleValue, int intBits, int fracBits) {
  long scaleFactor = 1 << fracBits;
  long maxValue = (1 << (intBits + fracBits)) - 1;
  long minValue = -maxValue - 1;

  double scaledDValue = doubleValue * scaleFactor;
  long scaledLValue = lround(scaledDValue);
  if (scaledLValue > maxValue)
    scaledLValue = maxValue;
  else if (scaledLValue < minValue)
    scaledLValue = minValue;

  double roundedValue = (double)scaledLValue / scaleFactor;
  return roundedValue;
}

std::pair<long, long> mprealToFixedPoint(mpfr::mpreal &value, int intBits,
                                         int fracBits) {
  long scaleFactor = 1l << fracBits;
  long maxValue = (1l << (intBits + fracBits)) - 1l;
  long minValue = -maxValue - 1l;

  mpfr::mpreal scaledMPValue = value * scaleFactor;
  long scaledLValue = scaledMPValue.toLong(GMP_RNDN);
  if (scaledLValue > maxValue)
    scaledLValue = maxValue;
  else if (scaledLValue < minValue)
    scaledLValue = minValue;

  return std::make_pair(scaledLValue, scaleFactor);
}

std::pair<std::pair<int, int>, std::pair<long, long>>
findBestFormatMP(mpfr::mpreal &value, int size) {

  std::pair<int, int> format{-size, size};
  std::pair<long, long> formatDecomp =
      mprealToFixedPoint(value, format.first, format.second);
  mpfr::mpreal bestFit = formatDecomp.first;
  bestFit /= formatDecomp.second;
  for (int i{-size + 1}, j{2 * size - 1}; i <= size; ++i, --j) {
    std::pair<long, long> fixDecomp = mprealToFixedPoint(value, i, j);
    mpfr::mpreal fixVal = fixDecomp.first;
    fixVal /= fixDecomp.second;
    if (mpfr::abs(value - bestFit) > mpfr::abs(value - fixVal)) {
      format = std::make_pair(i, j);
      formatDecomp = fixDecomp;
      bestFit = fixVal;
    }
  }

  return std::make_pair(format, formatDecomp);
}

std::pair<std::pair<int, int>, double> findBestFormatD(double &value,
                                                       int size) {
  std::pair<int, int> format{-size, size};
  double bestRounding = doubleToFixedPoint(value, format.first, format.second);
  for (int i{-size + 1}, j{2 * size - 1}; i <= size; ++i, --j) {
    double currentRounding = doubleToFixedPoint(value, i, j);
    if (fabs(value - bestRounding) > fabs(value - currentRounding)) {
      bestRounding = currentRounding;
      format = std::make_pair(i, j);
    }
  }

  return std::make_pair(format, bestRounding);
}

std::pair<std::pair<int, int>, double> defaultFormatD(double &value, int size) {
  return std::make_pair(std::make_pair(0, size),
                        doubleToFixedPoint(value, 0, size));
}

std::pair<std::pair<int, int>, double> customFormatD(double &value, int intBits,
                                                     int fracBits) {
  return std::make_pair(std::make_pair(intBits, fracBits),
                        doubleToFixedPoint(value, intBits, fracBits));
}

std::string mprealToString(mpfr::mpreal &data, int base,
                           std::size_t precision) {
  char *buffer;
  mp_exp_t exp;
  buffer =
      mpfr_get_str(nullptr, &exp, base, precision, data.mpfr_ptr(), GMP_RNDN);
  std::ostringstream sbuf;
  if (mpfr::sgn(data) == -1)
    sbuf << "-0." << (buffer + 1) << "e" << exp;
  else
    sbuf << "0." << buffer << "e" << exp;
  mpfr_free_str(buffer);
  return sbuf.str();
}

void mprealToMpq(mpq_t &ratVal, mpfr::mpreal &mprealVal) {

  mpfr_exp_t expVal = mpfr_get_z_2exp(mpq_numref(ratVal), mprealVal.mpfr_ptr());
  mpz_set_ui(mpq_denref(ratVal), 1u);
  if (expVal < 0)
    mpz_ui_pow_ui(mpq_denref(ratVal), 2u, (unsigned)(-expVal));
  else
    mpz_mul_2exp(mpq_numref(ratVal), mpq_numref(ratVal), (mp_bitcnt_t)expVal);

  mpq_canonicalize(ratVal);
}

void mantissaExponentDecomposition(mpz_t &M, mp_exp_t &e, mpfr::mpreal &x) {
  if (mpfr_zero_p(x.mpfr_ptr())) {
    mpz_set_ui(M, 0u);
    e = 0;
    return;
  }
  // compute the normalized decomposition of the data
  // (with respect to the precision used to represent
  // the mpreal variable x)
  e = mpfr_get_z_exp(M, x.mpfr_ptr());
  while (mpz_divisible_ui_p(M, 2u)) {
    mpz_divexact_ui(M, M, 2u);
    ++e;
  }
}

void mprealToFixedPoint(std::vector<mpfr::mpreal> &naiveA,
                        std::vector<mpfr::mpreal> &a, int intBits,
                        int fracBits) {
  naiveA.resize(a.size());
  for (std::size_t i = 0u; i < a.size(); ++i)
    naiveA[i] =
        mpfr::mpreal(doubleToFixedPoint(a[i].toDouble(), intBits, fracBits));
}
