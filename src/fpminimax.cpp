#include "filter/fpminimax.h"
#include "filter/conv.h"
#include "filter/grid.h"
#include "filter/plotting.h"
#include "filter/roots.h"
#include <chrono>
#include <cstdlib>
#include <fplll.h>
#include <fstream>
#include <gmpxx.h>
#include <sstream>

typedef Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, Eigen::Dynamic> MatrixXq;
typedef Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> VectorXq;

std::pair<mpz_class, mp_exp_t> mpfrDecomp(mpfr::mpreal &val) {
  if (val == 0)
    return std::make_pair(mpz_class(0l), 0);
  mpz_t buffer;
  mpz_init(buffer);
  mp_exp_t exp;
  // compute the normalized decomposition of the data
  // (in regard to the precision of the internal data representation)
  exp = mpfr_get_z_2exp(buffer, val.mpfr_ptr());
  while (mpz_divisible_ui_p(buffer, 2ul)) {
    mpz_divexact_ui(buffer, buffer, 2ul);
    ++exp;
  }
  mpz_class signif(buffer);
  mpz_clear(buffer);
  return std::make_pair(signif, exp);
}

std::pair<mpz_class, mp_exp_t> mpfrDecomp(double val) {
  if (val == 0.0)
    return std::make_pair(mpz_class(0l), 0);
  mpz_t buffer;
  mpfr_t mpfr_value;
  mpfr_init2(mpfr_value, 54);
  mpfr_set_d(mpfr_value, val, GMP_RNDN);
  mpz_init(buffer);
  mp_exp_t exp;
  // compute the normalized decomposition of the data
  // (in regard to the precision of the internal data representation)
  exp = mpfr_get_z_exp(buffer, mpfr_value);
  while (mpz_divisible_ui_p(buffer, 2ul)) {
    mpz_divexact_ui(buffer, buffer, 2ul);
    ++exp;
  };
  mpz_class signif(buffer);
  mpz_clear(buffer);
  mpfr_clear(mpfr_value);
  return std::make_pair(signif, exp);
}

void gsOrthogonalization(std::vector<std::vector<mpfr::mpreal>> &v) {
  std::size_t n{v.size()};
  std::vector<std::vector<mpfr::mpreal>> r(n);
  std::vector<std::vector<mpfr::mpreal>> q(n);
  for (std::size_t i{0u}; i < n; ++i) {
    r[i].resize(n);
    r[i][i] = 0;
    q[i] = v[i];
    for (std::size_t j{0u}; j < v[i].size(); ++j)
      r[i][i] += v[i][j] * v[i][j];
    r[i][i] = mpfr::sqrt(r[i][i]);
    for (std::size_t j{0u}; j < q[i].size(); ++j)
      q[i][j] /= r[i][i];
    for (std::size_t j{i + 1u}; j < n; ++j) {
      r[i][j] = 0;
      for (std::size_t k{0u}; k < v[j].size(); ++k)
        r[i][j] += q[i][k] * v[j][k];
      for (std::size_t k{0u}; k < v[j].size(); ++k)
        v[j][k] -= r[i][j] * q[i][k];
    }
  }
}

void createFIRBasisType1(fplll::ZZ_mat<mpz_t> &basis,
                         std::vector<mpfr::mpreal> &nodes,
                         std::vector<mpfr::mpreal> &iT,
                         std::vector<mpz_class> &nT,
                         std::vector<mpfr::mpreal> &weights,
                         mpfr::mpreal &scalingFactor, std::size_t n,
                         mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  // store the min value for an exponent used to represent
  // in mantissa-exponent form the lattice basis and the
  // vector T to be approximated using the LLL approach
  mp_exp_t minExp = 0;
  // determine the scaling factor for the basis vectors and
  // the unknown vector to be approximated by T
  mpreal powBuffer;

  for (std::size_t i = 0u; i < nodes.size(); ++i) {
    iT[i] *= weights[i];
    for (std::size_t j = 0u; j < n; ++j) {
      powBuffer = weights[i] * mpfr::cos(nodes[i] * j) / scalingFactor;
      std::pair<mpz_class, mp_exp_t> decomp = mpfrDecomp(powBuffer);
      if (j > 0u)
        decomp.second += 1u;
      else
        decomp.second += 0u;
      if (decomp.second < minExp)
        minExp = decomp.second;
    }
  }

  mpfr::mpreal resultBuffer;
  for (auto &i : iT) {
    std::pair<mpz_class, mp_exp_t> decomp = mpfrDecomp(i);
    if (decomp.second < minExp)
      minExp = decomp.second;
  }

  // scale the basis and vector T
  basis.resize(n, nodes.size());
  mpz_t intBuffer;
  mpz_init(intBuffer);
  for (std::size_t i = 0u; i < n; ++i)
    for (std::size_t j = 0u; j < nodes.size(); ++j) {
      powBuffer = weights[j] * mpfr::cos(nodes[j] * i) / scalingFactor;
      std::pair<mpz_class, mp_exp_t> decomp = mpfrDecomp(powBuffer);
      mp_exp_t nexp = decomp.second - minExp;
      if (i > 0u)
        nexp += 1u;
      else
        nexp += 0u;
      mpz_ui_pow_ui(intBuffer, 2u, (unsigned int)nexp);
      mpz_mul(intBuffer, intBuffer, decomp.first.get_mpz_t());
      mpz_set(basis(i, j).getData(), intBuffer);
    }

  nT.clear();
  for (std::size_t i = 0u; i < iT.size(); ++i) {
    std::pair<mpz_class, mp_exp_t> decomp = mpfrDecomp(iT[i]);
    mp_exp_t nexp = decomp.second - minExp;
    mpz_ui_pow_ui(intBuffer, 2u, (unsigned int)nexp);
    mpz_mul(intBuffer, intBuffer, decomp.first.get_mpz_t());
    nT.push_back(mpz_class(intBuffer));
  }

  // clean-up
  mpz_clear(intBuffer);

  mpreal::set_default_prec(prevPrec);
}

void maxVectorNorm(mpz_t *maxNorm, fplll::ZZ_mat<mpz_t> &basis) {

  mpz_t maxValue;
  mpz_t iter;
  mpz_init(maxValue);
  mpz_init(iter);
  mpz_set_ui(maxValue, 0);

  for (int i = 0; i < basis.GetNumRows(); ++i) {
    mpz_set_ui(iter, 0);
    for (int j = 0; j < basis.GetNumCols(); ++j) {
      mpz_addmul(iter, basis(i, j).getData(), basis(i, j).getData());
    }
    if (mpz_cmp(iter, maxValue) > 0)
      mpz_set(maxValue, iter);
  }
  mpfr_t fMaxValue;
  mpfr_init_set_z(fMaxValue, maxValue, GMP_RNDN);
  mpfr_sqrt(fMaxValue, fMaxValue, GMP_RNDN);
  mpfr_get_z(*maxNorm, fMaxValue, GMP_RNDD);

  mpz_clear(iter);
  mpz_clear(maxValue);
  mpfr_clear(fMaxValue);
}

void extractBasisVectors(std::vector<std::vector<mpfr::mpreal>> &vBasis,
                         fplll::ZZ_mat<mpz_t> &basis, std::size_t dimRows,
                         std::size_t dimCols) {
  vBasis.resize(dimRows);
  for (std::size_t i = 0u; i < dimRows; ++i) {
    vBasis[i].resize(dimCols);
    for (std::size_t j = 0u; j < dimCols; ++j) {
      vBasis[i][j] = basis(i, j).GetData();
    }
  }
}

void applyKannanEmbedding(fplll::ZZ_mat<mpz_t> &basis,
                          std::vector<mpz_class> &t) {

  int outCols = basis.GetNumCols() + 1;
  int outRows = basis.GetNumRows() + 1;

  mpz_t w; // the CVP t vector weight
  mpz_init(w);
  maxVectorNorm(&w, basis);
  basis.resize(outRows, outCols);

  for (int i = 0; i < outRows - 1; ++i)
    mpz_set_ui(basis(i, outCols - 1).getData(), 0);
  for (int j = 0; j < outCols - 1; ++j)
    mpz_set(basis(outRows - 1, j).getData(), t[j].get_mpz_t());
  mpz_set(basis(outRows - 1, outCols - 1).getData(), w);
  mpz_clear(w);
}

// TODO
void fpminimaxKernel(std::vector<double> &lllCoeffs,
                     std::vector<double> &roundedCoeffs,
                     std::vector<mpfr::mpreal> &nodes,
                     std::vector<std::vector<double>> &svpCoeffs,
                     std::vector<mpfr::mpreal> &iT, mpfr::mpreal &scalingFactor,
                     std::vector<mpfr::mpreal> &weights, std::size_t n,
                     mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  std::vector<mpz_class> nT(iT.size());
  fplll::ZZ_mat<mpz_t> basis;
  createFIRBasisType1(basis, nodes, iT, nT, weights, scalingFactor, n, prec);
  applyKannanEmbedding(basis, nT);
  fplll::ZZ_mat<mpz_t> u(basis.GetNumRows(), basis.GetNumCols());

  fplll::lllReduction(basis, u, 0.99, 0.51);
  // Uncomment the next line and comment the next one for BKZ or HKZ reduction
  // fplll::bkzReduction(basis, u, 8);

  std::vector<mpz_class> intLLLCoeffs;
  std::vector<std::vector<mpz_class>> intSVPCoeffs(n);
  int xdp1 =
      (int)mpz_get_si(u.Get(u.GetNumRows() - 1, u.GetNumCols() - 1).GetData());

  mpz_t coeffAux;
  mpz_init(coeffAux);
  switch (xdp1) {
  case 1:
    for (int i{0}; i < u.GetNumCols() - 1; ++i) {
      mpz_neg(coeffAux, u.Get(u.GetNumRows() - 1, i).GetData());
      intLLLCoeffs.push_back(mpz_class(coeffAux));
      for (int j{0}; j < (int)n; ++j) {
        mpz_set(coeffAux, u.Get(j, i).GetData());
        intSVPCoeffs[j].push_back(mpz_class(coeffAux));
      }
    }
    break;
  case -1:
    for (int i = 0; i < u.GetNumCols() - 1; ++i) {
      intLLLCoeffs.push_back(mpz_class(u.Get(u.GetNumRows() - 1, i).GetData()));
      for (int j = 0; j < (int)n; ++j) {
        mpz_set(coeffAux, u.Get(j, i).GetData());
        intSVPCoeffs[j].push_back(mpz_class(coeffAux));
      }
    }
    break;
  default:
    std::cout << "Failed to generate the polynomial approximation\n";
    break;
  }
  mpz_clear(coeffAux);

  svpCoeffs.resize(intSVPCoeffs.size());
  for (std::size_t i = 0u; i < intLLLCoeffs.size(); ++i) {
    mpreal newCoeff;
    lllCoeffs[i] = intLLLCoeffs[i].get_d() / scalingFactor.toDouble();
    if (i > 0u)
      lllCoeffs[i] *= 2;
    else
      lllCoeffs[i] *= 1;
    for (std::size_t j = 0u; j < svpCoeffs.size(); ++j) {
      double buffer = intSVPCoeffs[j][i].get_d() / scalingFactor.toDouble();
      if (i > 0u)
        svpCoeffs[j].push_back(buffer * 2);
      else
        svpCoeffs[j].push_back(buffer * 1);
    }
  }

  mpreal::set_default_prec(prevPrec);
}

void cvpKernel(std::vector<double> &lllCoeffs, std::vector<mpfr::mpreal> &nodes,
               std::vector<mpfr::mpreal> &iT, mpfr::mpreal &scalingFactor,
               std::vector<mpfr::mpreal> &weights, std::size_t n,
               mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  std::vector<mpz_class> nT(iT.size());
  fplll::ZZ_mat<mpz_t> basis;
  std::vector<fplll::Integer> intTarget(nT.size());
  std::vector<fplll::Integer> solCoord;
  createFIRBasisType1(basis, nodes, iT, nT, weights, scalingFactor, n, prec);
  for (std::size_t i = 0u; i < nT.size(); ++i)
    mpz_set(intTarget[i].getData(), nT[i].get_mpz_t());
  fplll::closestVector(basis, intTarget, solCoord);

  for (std::size_t i = 0u; i < solCoord.size(); ++i) {
    lllCoeffs[i] = solCoord[i].get_d() / scalingFactor.toDouble();
    if (i > 0u)
      lllCoeffs[i] *= 2;
    else
      lllCoeffs[i] *= 1;
  }

  mpreal::set_default_prec(prevPrec);
}

void fpminimaxWithExactCVP(mpfr::mpreal &minError,
                           std::vector<mpfr::mpreal> &lllFreeA,
                           std::vector<mpfr::mpreal> &freeA,
                           std::vector<mpfr::mpreal> &fixedA,
                           std::vector<mpfr::mpreal> &interpolationPoints,
                           std::vector<Band> &freqBands,
                           std::vector<mpfr::mpreal> &weights,
                           mpfr::mpreal &scalingFactor, mp_prec_t prec) {
  using namespace mpfr;
  mpreal::set_default_prec(prec);

  std::vector<Band> chebyBands;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ, prec);

  std::vector<mpfr::mpreal> v(interpolationPoints.size());
  std::vector<mpfr::mpreal> fv(interpolationPoints.size());
  for (std::size_t i = 0u; i < interpolationPoints.size(); ++i) {
    v[i] = mpfr::acos(interpolationPoints[i]);
    evaluateClenshaw(fv[i], freeA, interpolationPoints[i], prec);
  }

  std::vector<double> freeADouble(freeA.size());
  std::vector<mpfr::mpreal> roundedA = freeA;
  for (std::size_t i = 0u; i < freeA.size(); ++i) {
    freeADouble[i] = freeA[i].toDouble();
    if (i > 0u) {
      roundedA[i] *= scalingFactor;
      roundedA[i] >>= 1;
    } else {
      roundedA[i] *= scalingFactor;
      roundedA[i] >>= 0;
    }
    roundedA[i] = roundedA[i].toLong(GMP_RNDN);
    if (i > 0u) {
      roundedA[i] /= scalingFactor;
      roundedA[i] <<= 1;
    } else {
      roundedA[i] /= scalingFactor;
      roundedA[i] <<= 0;
    }
  }

  for (std::size_t i = 0u; i < fixedA.size(); ++i)
    roundedA.push_back(fixedA[i]);

  std::pair<mpfr::mpreal, mpfr::mpreal> naiveNorm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(naiveNorm, bandNorms, roundedA, freqBands, chebyBands, prec);
  std::cout << "Naive rounding error\t= " << mprealToString(naiveNorm.second)
            << std::endl;

  std::vector<double> lllA(freeA.size());
  cvpKernel(lllA, v, fv, scalingFactor, weights, freeA.size(), prec);

  std::vector<mpfr::mpreal> mpLLLA(freeA.size());

  for (std::size_t i = 0u; i < freeA.size(); ++i)
    mpLLLA[i] = lllA[i];

  for (std::size_t i = 0u; i < fixedA.size(); ++i)
    mpLLLA.push_back(fixedA[i]);

  std::pair<mpfr::mpreal, mpfr::mpreal> lllNorm;
  computeNorm(lllNorm, bandNorms, mpLLLA, freqBands, chebyBands, prec);
  std::cout << "CVP-based error\t= " << mprealToString(lllNorm.second)
            << std::endl;
  minError = lllNorm.second;
  mpfr::mpreal zero = 0;
  for (std::size_t k{0u}; k < bandNorms.size(); ++k)
    std::cout << "Band " << k
              << " error = " << mprealToString(bandNorms[k].second)
              << std::endl;
}

void fpminimaxWithNeighborhoodSearch(
    mpfr::mpreal &minError, std::vector<mpfr::mpreal> &lllFreeA,
    std::vector<mpfr::mpreal> &freeA, std::vector<mpfr::mpreal> &fixedA,
    std::vector<mpfr::mpreal> &interpolationPoints,
    std::vector<Band> &freqBands, std::vector<mpfr::mpreal> &weights,
    mpfr::mpreal &scalingFactor, mp_prec_t prec) {
  using namespace mpfr;
  mpreal::set_default_prec(prec);

  std::vector<mpfr::mpreal> grid;
  generateGridPoints(grid, freeA.size(), freqBands, prec);

  std::vector<Band> chebyBands;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ, prec);

  std::vector<mpfr::mpreal> v(interpolationPoints.size());
  std::vector<mpfr::mpreal> fv(interpolationPoints.size());
  for (std::size_t i = 0u; i < interpolationPoints.size(); ++i) {
    v[i] = mpfr::acos(interpolationPoints[i]);
    evaluateClenshaw(fv[i], freeA, interpolationPoints[i], prec);
  }

  std::vector<double> freeADouble(freeA.size());
  std::vector<mpfr::mpreal> roundedA = freeA;
  for (std::size_t i = 0u; i < freeA.size(); ++i) {
    freeADouble[i] = freeA[i].toDouble();
    if (i > 0u) {
      roundedA[i] *= scalingFactor;
      roundedA[i] >>= 1;
    } else {
      roundedA[i] *= scalingFactor;
      roundedA[i] >>= 0;
    }
    roundedA[i] = roundedA[i].toLong(GMP_RNDN);
    if (i > 0u) {
      roundedA[i] /= scalingFactor;
      roundedA[i] <<= 1;
    } else {
      roundedA[i] /= scalingFactor;
      roundedA[i] <<= 0;
    }
  }

  for (std::size_t i = 0u; i < fixedA.size(); ++i)
    roundedA.push_back(fixedA[i]);

  std::pair<mpfr::mpreal, mpfr::mpreal> naiveNorm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(naiveNorm, bandNorms, roundedA, freqBands, chebyBands, prec);
  std::cout << "Naive rounding error\t= " << mprealToString(naiveNorm.second)
            << std::endl;

  std::vector<std::vector<double>> svpVectors(freeA.size());
  std::vector<std::vector<mpfr::mpreal>> mpSVPVectors(freeA.size());
  std::vector<double> lllA(freeA.size());
  fpminimaxKernel(lllA, freeADouble, v, svpVectors, fv, scalingFactor, weights,
                  freeA.size(), prec);

  std::vector<mpfr::mpreal> mpLLLA(freeA.size());

  for (std::size_t i = 0u; i < freeA.size(); ++i)
    mpLLLA[i] = lllA[i];

  for (std::size_t i = 0u; i < svpVectors.size(); ++i) {
    mpSVPVectors[i].resize(svpVectors[i].size());
    for (std::size_t j = 0u; j < svpVectors[i].size(); ++j) {
      mpSVPVectors[i][j] = svpVectors[i][j];
    }
  }

  for (std::size_t i = 0u; i < fixedA.size(); ++i)
    mpLLLA.push_back(fixedA[i]);

  std::pair<mpfr::mpreal, mpfr::mpreal> lllNorm;
  computeNorm(lllNorm, bandNorms, mpLLLA, freqBands, chebyBands, prec);
  std::cout << "LLL initial error\t= " << mprealToString(lllNorm.second)
            << std::endl;
  minError = lllNorm.second;
  mpfr::mpreal zero = 0;
  for (std::size_t k{0u}; k < bandNorms.size(); ++k)
    std::cout << "Band " << k
              << " error = " << mprealToString(bandNorms[k].second)
              << std::endl;

  std::pair<mpfr::mpreal, mpfr::mpreal> lllBestNorm;
  std::vector<mpfr::mpreal> mpBufferA(lllA.size() + fixedA.size());
  std::vector<mpfr::mpreal> mpFinalA;
  for (std::size_t i = lllA.size(); i < lllA.size() + fixedA.size(); ++i)
    mpBufferA[i] = fixedA[i - lllA.size()];
  mpFinalA = mpBufferA;
  for (std::size_t i = 0u; i < lllA.size(); ++i)
    mpFinalA[i] = lllA[i];
  std::vector<mpfr::mpreal> initialLLL = mpLLLA;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> searchBandNorms(
      chebyBands.size());

  for (std::size_t index1 = 0u; index1 < 8; ++index1)
    for (std::size_t index2 = index1 + 1u; index2 < 14; ++index2)

      for (int direction1 = -1; direction1 < 2; ++direction1)
        for (int direction2 = -1; direction2 < 2; ++direction2) {
          for (std::size_t i{0u}; i < freeA.size(); ++i) {
            double vA = lllA[i] + svpVectors[index1][i] * direction1 +
                        svpVectors[index2][i] * direction2;
            mpBufferA[i] = vA;
          }
          std::pair<mpfr::mpreal, mpfr::mpreal> bufferNorm;
          mpfr::mpreal denseNorm;
          computeNorm(bufferNorm, searchBandNorms, mpBufferA, freqBands,
                      chebyBands, prec);
          // computeDenseNorm(denseNorm, chebyBands, mpBufferA, grid, prec);

          if (bufferNorm.second < lllNorm.second) {
            lllNorm = bufferNorm;
            minError = lllNorm.second;
            bandNorms = searchBandNorms;
            std::cout << "Index1 = " << index1 << " Direction1 = " << direction1
                      << std::endl;
            std::cout << "Index2 = " << index2 << " Direction2 = " << direction2
                      << std::endl;

            std::cout << "LLL search norm\t= "
                      << mprealToString(bufferNorm.second) << std::endl;
            for (std::size_t k{0u}; k < bandNorms.size(); ++k)
              std::cout << "Band " << k
                        << " error = " << mprealToString(bandNorms[k].second)
                        << std::endl;
            // std::cout << "LLL grid norm\t= " << mprealToString(denseNorm) <<
            // std::endl;

            mpFinalA = mpBufferA;
          }
        }

  lllFreeA.resize(freeA.size());
  for (std::size_t i = 0u; i < freeA.size(); ++i)
    lllFreeA[i] = mpFinalA[i];

  std::cout << "Final quantization parameters:\n";
  for (std::size_t i = 0u; i < lllFreeA.size(); ++i) {
    mpfr::mpreal roundedBuffer = roundedA[i];
    mpfr::mpreal finalBuffer = mpFinalA[i];
    mpfr::mpreal initBuffer = initialLLL[i];
    if (i > 0u) {
      roundedBuffer *= scalingFactor;
      finalBuffer *= scalingFactor;
      initBuffer *= scalingFactor;
      roundedBuffer >>= 1;
      finalBuffer >>= 1;
      initBuffer >>= 1;

      mpLLLA[i] = finalBuffer / scalingFactor;
      mpLLLA[i] *= 2;
    } else {
      roundedBuffer *= scalingFactor;
      finalBuffer *= scalingFactor;
      initBuffer *= scalingFactor;
      roundedBuffer >>= 0;
      finalBuffer >>= 0;
      initBuffer >>= 0;

      mpLLLA[i] = finalBuffer / (scalingFactor);
      mpLLLA[i] *= 1;
    }
    finalBuffer = finalBuffer.toLong(GMP_RNDN);

    std::cout << "Index " << i << " value = " << roundedBuffer.toLong() << "\t"
              << initBuffer.toLong() << "\t" << finalBuffer.toLong()
              << std::endl;
  }
  computeNorm(lllNorm, bandNorms, mpLLLA, freqBands, chebyBands, prec);
  std::cout << "LLL final error\t= " << mprealToString(lllNorm.second)
            << std::endl;
}

void fpminimaxWithNeighborhoodSearchDiscrete(
    mpfr::mpreal &minError, std::vector<mpfr::mpreal> &lllFreeA,
    std::vector<mpfr::mpreal> &freeA, std::vector<mpfr::mpreal> &fixedA,
    std::vector<mpfr::mpreal> &interpolationPoints,
    std::vector<Band> &freqBands, std::vector<mpfr::mpreal> &weights,
    mpfr::mpreal &scalingFactor, mp_prec_t prec) {
  using namespace mpfr;
  mpreal::set_default_prec(prec);

  std::vector<GridPoint> grid;
  generateGrid(grid, freeA.size(), freqBands, 16u, prec);

  std::vector<Band> chebyBands;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ, prec);

  std::vector<mpfr::mpreal> v(interpolationPoints.size());
  std::vector<mpfr::mpreal> fv(interpolationPoints.size());
  for (std::size_t i = 0u; i < interpolationPoints.size(); ++i) {
    v[i] = mpfr::acos(interpolationPoints[i]);
    evaluateClenshaw(fv[i], freeA, interpolationPoints[i], prec);
  }

  std::vector<double> freeADouble(freeA.size());
  std::vector<mpfr::mpreal> roundedA = freeA;
  for (std::size_t i = 0u; i < freeA.size(); ++i) {
    freeADouble[i] = freeA[i].toDouble();
    if (i > 0u) {
      roundedA[i] *= scalingFactor;
      roundedA[i] >>= 1;
    } else {
      roundedA[i] *= scalingFactor;
      roundedA[i] >>= 0;
    }
    roundedA[i] = roundedA[i].toLong(GMP_RNDN);
    if (i > 0u) {
      roundedA[i] /= scalingFactor;
      roundedA[i] <<= 1;
    } else {
      roundedA[i] /= scalingFactor;
      roundedA[i] <<= 0;
    }
  }

  for (std::size_t i = 0u; i < fixedA.size(); ++i)
    roundedA.push_back(fixedA[i]);

  mpfr::mpreal naiveNorm;
  std::vector<mpfr::mpreal> bandNorms(chebyBands.size());
  computeDenseNorm(naiveNorm, bandNorms, chebyBands, grid, roundedA, prec);
  std::cout << "Naive rounding error\t= " << mprealToString(naiveNorm)
            << std::endl;

  std::vector<std::vector<double>> svpVectors(freeA.size());
  std::vector<std::vector<mpfr::mpreal>> mpSVPVectors(freeA.size());
  std::vector<double> lllA(freeA.size());
  fpminimaxKernel(lllA, freeADouble, v, svpVectors, fv, scalingFactor, weights,
                  freeA.size(), prec);

  std::vector<mpfr::mpreal> mpLLLA(freeA.size());

  for (std::size_t i = 0u; i < freeA.size(); ++i)
    mpLLLA[i] = lllA[i];

  for (std::size_t i = 0u; i < svpVectors.size(); ++i) {
    mpSVPVectors[i].resize(svpVectors[i].size());
    for (std::size_t j = 0u; j < svpVectors[i].size(); ++j) {
      mpSVPVectors[i][j] = svpVectors[i][j];
    }
  }

  for (std::size_t i = 0u; i < fixedA.size(); ++i)
    mpLLLA.push_back(fixedA[i]);

  mpfr::mpreal lllNorm;
  computeDenseNorm(lllNorm, bandNorms, chebyBands, grid, mpLLLA, prec);
  std::cout << "LLL initial error\t= " << mprealToString(lllNorm) << std::endl;
  for (std::size_t k{0u}; k < bandNorms.size(); ++k)
    std::cout << "Band " << k << " error = " << mprealToString(bandNorms[k])
              << std::endl;

  mpfr::mpreal lllBestNorm;
  std::vector<mpfr::mpreal> mpBufferA(lllA.size() + fixedA.size());
  std::vector<mpfr::mpreal> mpFinalA;
  for (std::size_t i = lllA.size(); i < lllA.size() + fixedA.size(); ++i)
    mpBufferA[i] = fixedA[i - lllA.size()];
  mpFinalA = mpBufferA;
  for (std::size_t i = 0u; i < lllA.size(); ++i)
    mpFinalA[i] = lllA[i];
  std::vector<mpfr::mpreal> initialLLL = mpLLLA;
  std::vector<mpfr::mpreal> searchBandNorms(chebyBands.size());

  for (std::size_t index1 = 0u; index1 < 3; ++index1)
    for (std::size_t index2 = index1 + 1u; index2 < mpLLLA.size() - 1; ++index2)

      for (int direction1 = -1; direction1 < 2; ++direction1)
        for (int direction2 = -1; direction2 < 2; ++direction2)

        {
          for (std::size_t i{0u}; i < freeA.size(); ++i) {
            double vA = lllA[i] + svpVectors[index1][i] * direction1 +
                        svpVectors[index2][i] *
                            direction2; // + svpVectors[index3][i] * direction3;
            mpBufferA[i] = vA;
          }
          mpfr::mpreal bufferNorm;
          computeDenseNorm(bufferNorm, searchBandNorms, chebyBands, grid,
                           mpBufferA, prec);

          if (bufferNorm < lllNorm) {
            lllNorm = bufferNorm;
            bandNorms = searchBandNorms;
            std::cout << "Index1 = " << index1 << " Direction1 = " << direction1
                      << std::endl;
            std::cout << "Index2 = " << index2 << " Direction2 = " << direction2
                      << std::endl;

            std::cout << "LLL search norm\t= " << mprealToString(bufferNorm)
                      << std::endl;
            for (std::size_t k{0u}; k < bandNorms.size(); ++k)
              std::cout << "Band " << k
                        << " error = " << mprealToString(bandNorms[k])
                        << std::endl;

            mpFinalA = mpBufferA;
          }
        }

  lllFreeA.resize(freeA.size());
  for (std::size_t i = 0u; i < freeA.size(); ++i)
    lllFreeA[i] = mpFinalA[i];

  for (std::size_t i = 0u; i < lllFreeA.size(); ++i) {
    mpfr::mpreal roundedBuffer = roundedA[i];
    mpfr::mpreal finalBuffer = mpFinalA[i];
    mpfr::mpreal initBuffer = initialLLL[i];
    if (i > 0u) {
      roundedBuffer *= scalingFactor;
      finalBuffer *= scalingFactor;
      initBuffer *= scalingFactor;
      roundedBuffer >>= 1;
      finalBuffer >>= 1;
      initBuffer >>= 1;

      mpLLLA[i] = finalBuffer / scalingFactor;
      mpLLLA[i] *= 2;
    } else {
      roundedBuffer *= scalingFactor;
      finalBuffer *= scalingFactor;
      initBuffer *= scalingFactor;
      roundedBuffer >>= 0;
      finalBuffer >>= 0;
      initBuffer >>= 0;

      mpLLLA[i] = finalBuffer / (scalingFactor);
      mpLLLA[i] *= 1;
    }
    finalBuffer = finalBuffer.toLong(GMP_RNDN);
  }
  std::cout << std::endl;
  computeDenseNorm(lllNorm, bandNorms, chebyBands, grid, mpLLLA, prec);
  std::cout << "LLL final error\t= " << mprealToString(lllNorm) << std::endl;
}
