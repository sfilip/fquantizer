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
#include <iomanip>
#include <random>

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
      mpz_set(basis(i, j).get_data(), intBuffer);
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


void createFIRBasisType1V2(fplll::ZZ_mat<mpz_t> &basis,
                         std::vector<mpfr::mpreal> &nodes,
                         std::vector<mpfr::mpreal> &iT1,
                         std::vector<mpz_class> &nT1,
                         std::vector<mpfr::mpreal> &iT2,
                         std::vector<mpz_class> &nT2,
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
    iT1[i] *= weights[i];
    iT2[i] *= weights[i];
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
  for (auto &i : iT1) {
    std::pair<mpz_class, mp_exp_t> decomp = mpfrDecomp(i);
    if (decomp.second < minExp)
      minExp = decomp.second;
  }

  for (auto &i : iT2) {
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
      mpz_set(basis(i, j).get_data(), intBuffer);
    }

  nT1.clear();
  nT2.clear();
  for (std::size_t i = 0u; i < iT1.size(); ++i) {
    std::pair<mpz_class, mp_exp_t> decomp = mpfrDecomp(iT1[i]);
    mp_exp_t nexp = decomp.second - minExp;
    mpz_ui_pow_ui(intBuffer, 2u, (unsigned int)nexp);
    mpz_mul(intBuffer, intBuffer, decomp.first.get_mpz_t());
    nT1.push_back(mpz_class(intBuffer));


    decomp = mpfrDecomp(iT2[i]);
    nexp = decomp.second - minExp;
    mpz_ui_pow_ui(intBuffer, 2u, (unsigned int)nexp);
    mpz_mul(intBuffer, intBuffer, decomp.first.get_mpz_t());
    nT2.push_back(mpz_class(intBuffer));

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

  for (int i = 0; i < basis.get_rows(); ++i) {
    mpz_set_ui(iter, 0);
    for (int j = 0; j < basis.get_cols(); ++j) {
      mpz_addmul(iter, basis(i, j).get_data(), basis(i, j).get_data());
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
      vBasis[i][j] = basis(i, j).get_data();
    }
  }
}

void applyKannanEmbedding(fplll::ZZ_mat<mpz_t> &basis,
                          std::vector<mpz_class> &t) {

  int outCols = basis.get_cols() + 1;
  int outRows = basis.get_rows() + 1;

  mpz_t w; // the CVP t vector weight
  mpz_init(w);
  maxVectorNorm(&w, basis);
  basis.resize(outRows, outCols);

  for (int i = 0; i < outRows - 1; ++i)
    mpz_set_ui(basis(i, outCols - 1).get_data(), 0);
  for (int j = 0; j < outCols - 1; ++j)
    mpz_set(basis(outRows - 1, j).get_data(), t[j].get_mpz_t());
  mpz_set(basis(outRows - 1, outCols - 1).get_data(), w);
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
  fplll::ZZ_mat<mpz_t> u(basis.get_rows(), basis.get_cols());

  fplll::lll_reduction(basis, u, 0.99, 0.51);
  // Uncomment the next line and comment the next one for BKZ or HKZ reduction
  //fplll::bkzReduction(basis, u, 8u);

  std::vector<mpz_class> intLLLCoeffs;
  std::vector<std::vector<mpz_class>> intSVPCoeffs(n);
  int xdp1 =
      (int)mpz_get_si(u(u.get_rows() - 1, u.get_cols() - 1).get_data());

  mpz_t coeffAux;
  mpz_init(coeffAux);
  switch (xdp1) {
  case 1:
    for (int i{0}; i < u.get_cols() - 1; ++i) {
      mpz_neg(coeffAux, u(u.get_rows() - 1, i).get_data());
      intLLLCoeffs.push_back(mpz_class(coeffAux));
      for (int j{0}; j < (int)n; ++j) {
        mpz_set(coeffAux, u(j, i).get_data());
        intSVPCoeffs[j].push_back(mpz_class(coeffAux));
      }
    }
    break;
  case -1:
    for (int i = 0; i < u.get_cols() - 1; ++i) {
      intLLLCoeffs.push_back(mpz_class(u(u.get_rows() - 1, i).get_data()));
      for (int j = 0; j < (int)n; ++j) {
        mpz_set(coeffAux, u(j, i).get_data());
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



void fpminimaxKernelV2(std::vector<double> &lllCoeffs1,
                     std::vector<double> &lllCoeffs2,
                     std::vector<double> &roundedCoeffs,
                     std::vector<mpfr::mpreal> &nodes,
                     std::vector<std::vector<double>> &svpCoeffs,
                     std::vector<mpfr::mpreal> &iT1,
                     std::vector<mpfr::mpreal> &iT2,
                     mpfr::mpreal &scalingFactor,
                     std::vector<mpfr::mpreal> &weights, std::size_t n,
                     mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  std::vector<mpz_class> nT1(iT1.size());
  std::vector<mpz_class> nT2(iT2.size());

  fplll::ZZ_mat<mpz_t> basis;
  fplll::ZZ_mat<mpz_t> origBasis;
  createFIRBasisType1V2(basis, nodes, iT1, nT1, iT2, nT2, weights, scalingFactor, n, prec);
  applyKannanEmbedding(basis, nT1);
  fplll::ZZ_mat<mpz_t> u(basis.get_rows(), basis.get_cols());


  fplll::lll_reduction(basis, u, 0.99, 0.51);
  //fplll::bkzReduction(basis, u, 8u);

  mpz_t maxValue;
  mpz_t iter;
  mpz_init(maxValue);
  mpz_init(iter);

  for (int i = 0; i < basis.get_rows() - 1; ++i) {
    mpz_set_ui(maxValue, 0);
    for (int j = 0; j < basis.get_cols(); ++j) {
        mpz_abs(iter, basis(i, j).get_data());
        if (mpz_cmp(iter, maxValue) > 0)
        mpz_set(maxValue, iter);
    }
  }

  mpz_clear(iter);
  mpz_clear(maxValue);



  std::vector<mpz_class> intLLLCoeffs1;
  std::vector<std::vector<mpz_class>> intSVPCoeffs(n);
  int xdp1 =
      (int)mpz_get_si(u(u.get_rows() - 1, u.get_cols() - 1).get_data());

  mpz_t coeffAux;
  mpz_init(coeffAux);
  switch (xdp1) {
  case 1:
    for (int i{0}; i < u.get_cols() - 1; ++i) {
      mpz_neg(coeffAux, u(u.get_rows() - 1, i).get_data());
      intLLLCoeffs1.push_back(mpz_class(coeffAux));
      //std::cout << mpz_class(coeffAux) << std::endl;
      for (int j{0}; j < (int)n; ++j) {
        mpz_set(coeffAux, u(j, i).get_data());
        intSVPCoeffs[j].push_back(mpz_class(coeffAux));
      }
    }
    break;
  case -1:
    for (int i = 0; i < u.get_cols() - 1; ++i) {
      intLLLCoeffs1.push_back(mpz_class(u(u.get_rows() - 1, i).get_data()));

      //std::cout << mpz_class(coeffAux) << std::endl;
      for (int j = 0; j < (int)n; ++j) {
        mpz_set(coeffAux, u(j, i).get_data());
        intSVPCoeffs[j].push_back(mpz_class(coeffAux));

      }
    }
    break;
  default:
    std::cout << "Failed to generate the polynomial approximation\n";
    break;
  }

  svpCoeffs.resize(intSVPCoeffs.size());
  for (std::size_t i = 0u; i < intLLLCoeffs1.size(); ++i) {
    mpreal newCoeff;
    lllCoeffs1[i] = intLLLCoeffs1[i].get_d() / scalingFactor.toDouble();
    if (i > 0u)
      lllCoeffs1[i] *= 2;
    else
      lllCoeffs1[i] *= 1;
    for (std::size_t j = 0u; j < svpCoeffs.size(); ++j) {
      double buffer = intSVPCoeffs[j][i].get_d() / scalingFactor.toDouble();
      if (i > 0u)
        svpCoeffs[j].push_back(buffer * 2);
      else
        svpCoeffs[j].push_back(buffer * 1);
    }
  }

  // Get the representation for the second
  fplll::ZZ_mat<mpz_t> basis2;
  basis2.resize(basis.get_rows() - 1u, basis.get_cols() - 1u);
  for(std::size_t i{0u}; i < basis2.get_rows(); ++i)
      for(std::size_t j{0u}; j < basis2.get_cols(); ++j)
        mpz_set(basis2(i, j).get_data(), basis(i, j).get_data());

  std::vector<mpz_class> intLLLCoeffs2;

  applyKannanEmbedding(basis2, nT2);
  fplll::ZZ_mat<mpz_t> u2(basis2.get_rows(), basis2.get_cols());

  fplll::lll_reduction(basis2, u2, 0.99, 0.51);
  //fplll::bkzReduction(basis2, u2, 8u);


  int xdp2 =
      (int)mpz_get_si(u2(u2.get_rows() - 1, u2.get_cols() - 1).get_data());


  mpz_t coeffBuffer;
  mpz_init(coeffBuffer);
  for(std::size_t i{0u}; i < intLLLCoeffs1.size(); ++i)
  {
    mpz_set_ui(coeffBuffer, 0u);
    for(std::size_t j{0u}; j < intLLLCoeffs1.size(); ++j) {
        mpz_addmul(coeffBuffer, u2(u2.get_rows() - 1u, j).get_data(), u(j, i).get_data());
    }
    if(xdp2 == 1)
        mpz_neg(coeffBuffer, coeffBuffer);
        intLLLCoeffs2.push_back(mpz_class(coeffBuffer));
    //std::cout << mpz_class(coeffBuffer) << std::endl;
  }
  for (std::size_t i = 0u; i < intLLLCoeffs2.size(); ++i) {
    mpreal newCoeff;
    lllCoeffs2[i] = intLLLCoeffs2[i].get_d() / scalingFactor.toDouble();
    if (i > 0u)
      lllCoeffs2[i] *= 2;
    else
      lllCoeffs2[i] *= 1;
  }

  mpz_clear(coeffAux);
  mpz_clear(coeffBuffer);
  mpreal::set_default_prec(prevPrec);
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
  std::vector<mpfr::mpreal> idealV(interpolationPoints.size());
  for (std::size_t i = 0u; i < interpolationPoints.size(); ++i) {
    v[i] = mpfr::acos(interpolationPoints[i]);
    evaluateClenshaw(fv[i], freeA, interpolationPoints[i], prec);
    mpfr::mpreal buffer;
    for(std::size_t j{0u}; j < freqBands.size(); ++j) {
        if (mpfr::abs(v[i] - freqBands[j].stop) < 1e-14)
            v[i] = freqBands[j].stop;
        if (mpfr::abs(v[i] - freqBands[j].start) < 1e-14)
            v[i] = freqBands[j].start;
    }

    computeIdealResponseAndWeight(idealV[i], buffer, v[i],
            freqBands);
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

  std::vector<double> doubleA(roundedA.size());
  for (std::size_t i = 0u; i < roundedA.size(); ++i)
    doubleA[i] = roundedA[i].toDouble();


  double naiveNorm;
  std::vector<double> bandNorms(chebyBands.size());
  computeDenseNorm(naiveNorm, bandNorms, chebyBands, grid, doubleA);
  std::cout << "Naive rounding error\t= " << naiveNorm << std::endl;

  std::vector<std::vector<double>> svpVectors(freeA.size());
  std::vector<std::vector<mpfr::mpreal>> mpSVPVectors(freeA.size());
  std::vector<double> lllA1(freeA.size());
  std::vector<double> lllA2(freeA.size());
  auto start = std::chrono::steady_clock::now();
  fpminimaxKernelV2(lllA1, lllA2, freeADouble, v, svpVectors, fv, idealV, scalingFactor, weights,
                  freeA.size(), prec);
  auto stop = std::chrono::steady_clock::now();
  auto diff = stop - start;
  std::cout << "\nReduction = " << std::chrono::duration<double,std::milli>(diff).count() << " ms\n";

  for (std::size_t i = 0u; i < svpVectors.size(); ++i) {
    mpSVPVectors[i].resize(svpVectors[i].size());
    for (std::size_t j = 0u; j < svpVectors[i].size(); ++j) {
      mpSVPVectors[i][j] = svpVectors[i][j];
    }
  }


  // using minimax
  std::vector<mpfr::mpreal> mpLLLA1(freeA.size());

  for (std::size_t i = 0u; i < freeA.size(); ++i)
    mpLLLA1[i] = lllA1[i];



  for (std::size_t i = 0u; i < fixedA.size(); ++i)
    mpLLLA1.push_back(fixedA[i]);

  start = std::chrono::steady_clock::now();

  double lllNorm;
  computeDenseNorm(lllNorm, bandNorms, chebyBands, grid, lllA1);
  std::cout << "LLL initial error\t= " << lllNorm << std::endl;
  for (std::size_t k{0u}; k < bandNorms.size(); ++k)
    std::cout << "Band " << k << " error = " << bandNorms[k] << std::endl;


  double lllBestNorm;
  std::vector<mpfr::mpreal> mpBufferA(lllA1.size() + fixedA.size());
  std::vector<mpfr::mpreal> mpFinalA1;
  for (std::size_t i = lllA1.size(); i < lllA1.size() + fixedA.size(); ++i)
    mpBufferA[i] = fixedA[i - lllA1.size()];
  mpFinalA1 = mpBufferA;
  for (std::size_t i = 0u; i < lllA1.size(); ++i)
    mpFinalA1[i] = lllA1[i];
  std::vector<mpfr::mpreal> initialLLL1 = mpLLLA1;
  std::vector<double> searchBandNorms(chebyBands.size());


  for (std::size_t index1 = 0u; index1 < 9; ++index1)
    for (std::size_t index2 = index1 + 1u; index2 < mpLLLA1.size(); ++index2)
      for (int direction1 = -1; direction1 < 2; ++direction1)
        for (int direction2 = -1; direction2 < 2; ++direction2)
        {
          for (std::size_t i{0u}; i < freeA.size(); ++i) {
            double vA = lllA1[i] + svpVectors[index1][i] * direction1 +
                        svpVectors[index2][i] *
                            direction2;
            mpBufferA[i] = vA;
            doubleA[i] = vA;
          }
          double bufferNorm;
          computeDenseNorm(bufferNorm, searchBandNorms, chebyBands, grid, doubleA);

          if (bufferNorm < lllNorm) {
            lllNorm = bufferNorm;
            bandNorms = searchBandNorms;
            mpFinalA1 = mpBufferA;
          }
        }



  lllFreeA.resize(freeA.size());
  for (std::size_t i = 0u; i < freeA.size(); ++i)
    lllFreeA[i] = mpFinalA1[i];

  for (std::size_t i = 0u; i < lllFreeA.size(); ++i) {
    mpfr::mpreal roundedBuffer = roundedA[i];
    mpfr::mpreal finalBuffer = mpFinalA1[i];
    mpfr::mpreal initBuffer = initialLLL1[i];
    if (i > 0u) {
      roundedBuffer *= scalingFactor;
      finalBuffer *= scalingFactor;
      initBuffer *= scalingFactor;
      roundedBuffer >>= 1;
      finalBuffer >>= 1;
      initBuffer >>= 1;

      mpLLLA1[i] = finalBuffer / scalingFactor;
      mpLLLA1[i] *= 2;
    } else {
      roundedBuffer *= scalingFactor;
      finalBuffer *= scalingFactor;
      initBuffer *= scalingFactor;
      roundedBuffer >>= 0;
      finalBuffer >>= 0;
      initBuffer >>= 0;

      mpLLLA1[i] = finalBuffer / (scalingFactor);
      mpLLLA1[i] *= 1;
    }
    finalBuffer = finalBuffer.toLong(GMP_RNDN);
  }


  for(std::size_t i{0u}; i < mpLLLA1.size(); ++i)
    doubleA[i] = mpLLLA1[i].toDouble();

  double lllNorm1;
  computeDenseNorm(lllNorm1, bandNorms, chebyBands, grid, doubleA);
  std::cout << "LLL final error\t= " << lllNorm << std::endl;
  stop = std::chrono::steady_clock::now();
  diff = stop - start;
  std::cout << "\nVicinity search = " << std::chrono::duration<double,std::milli>(diff).count() << " ms\n";


  // using ideal
  std::vector<mpfr::mpreal> mpLLLA2(freeA.size());

  for (std::size_t i = 0u; i < freeA.size(); ++i)
    mpLLLA2[i] = lllA2[i];



  for (std::size_t i = 0u; i < fixedA.size(); ++i)
    mpLLLA2.push_back(fixedA[i]);

  start = std::chrono::steady_clock::now();

  computeDenseNorm(lllNorm, bandNorms, chebyBands, grid, lllA2);
  std::cout << "LLL initial error\t= " << lllNorm << std::endl;
  for (std::size_t k{0u}; k < bandNorms.size(); ++k)
    std::cout << "Band " << k << " error = " << bandNorms[k] << std::endl;


  std::vector<mpfr::mpreal> mpFinalA2;
  for (std::size_t i = lllA2.size(); i < lllA2.size() + fixedA.size(); ++i)
    mpBufferA[i] = fixedA[i - lllA2.size()];
  mpFinalA2 = mpBufferA;
  for (std::size_t i = 0u; i < lllA2.size(); ++i)
    mpFinalA2[i] = lllA2[i];
  std::vector<mpfr::mpreal> initialLLL2 = mpLLLA2;



  for (std::size_t index1 = 0u; index1 < 9; ++index1)
    for (std::size_t index2 = index1 + 1u; index2 < mpLLLA1.size(); ++index2)

      for (int direction1 = -1; direction1 < 2; ++direction1)
        for (int direction2 = -1; direction2 < 2; ++direction2)

        {
          for (std::size_t i{0u}; i < freeA.size(); ++i) {
            double vA = lllA2[i] + svpVectors[index1][i] * direction1 +
                        svpVectors[index2][i] *
                            direction2;
            mpBufferA[i] = vA;
            doubleA[i] = vA;
          }
          double bufferNorm;
          computeDenseNorm(bufferNorm, searchBandNorms, chebyBands, grid, doubleA);

          if (bufferNorm < lllNorm) {
            lllNorm = bufferNorm;
            bandNorms = searchBandNorms;
            mpFinalA2 = mpBufferA;
          }
        }




  for (std::size_t i = 0u; i < lllFreeA.size(); ++i) {
    mpfr::mpreal roundedBuffer = roundedA[i];
    mpfr::mpreal finalBuffer = mpFinalA2[i];
    mpfr::mpreal initBuffer = initialLLL2[i];
    if (i > 0u) {
      roundedBuffer *= scalingFactor;
      finalBuffer *= scalingFactor;
      initBuffer *= scalingFactor;
      roundedBuffer >>= 1;
      finalBuffer >>= 1;
      initBuffer >>= 1;

      mpLLLA2[i] = finalBuffer / scalingFactor;
      mpLLLA2[i] *= 2;
    } else {
      roundedBuffer *= scalingFactor;
      finalBuffer *= scalingFactor;
      initBuffer *= scalingFactor;
      roundedBuffer >>= 0;
      finalBuffer >>= 0;
      initBuffer >>= 0;

      mpLLLA2[i] = finalBuffer / (scalingFactor);
      mpLLLA2[i] *= 1;
    }
    finalBuffer = finalBuffer.toLong(GMP_RNDN);
  }
  stop = std::chrono::steady_clock::now();
  diff = stop - start;
  std::cout << "\nVicinity search = " << std::chrono::duration<double,std::milli>(diff).count() << " ms\n";


  double bestNorm = lllNorm1;
  std::vector<double> bestA = doubleA;
  std::vector<mpfr::mpreal> mpLLLA = mpLLLA1;
  for(std::size_t i{0u}; i < mpLLLA2.size(); ++i)
    doubleA[i] = mpLLLA2[i].toDouble();

  double lllNorm2;
  computeDenseNorm(lllNorm2, bandNorms, chebyBands, grid, doubleA);
  std::cout << "LLL final error\t= " << lllNorm << std::endl;

  if(lllNorm2 < lllNorm1)
  {
      bestA = doubleA;
      mpLLLA = mpLLLA2;
      bestNorm = lllNorm2;
      lllFreeA.resize(freeA.size());
      for (std::size_t i = 0u; i < freeA.size(); ++i)
        lllFreeA[i] = mpFinalA2[i];
  }

}


void fpminimaxWithNeighborhoodSearchDiscreteRand(
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
  std::vector<mpfr::mpreal> idealV(interpolationPoints.size());
  for (std::size_t i = 0u; i < interpolationPoints.size(); ++i) {
    v[i] = mpfr::acos(interpolationPoints[i]);
    evaluateClenshaw(fv[i], freeA, interpolationPoints[i], prec);
    mpfr::mpreal buffer;
    for(std::size_t j{0u}; j < freqBands.size(); ++j) {
        if (mpfr::abs(v[i] - freqBands[j].stop) < 1e-14)
            v[i] = freqBands[j].stop;
        if (mpfr::abs(v[i] - freqBands[j].start) < 1e-14)
            v[i] = freqBands[j].start;
    }

    computeIdealResponseAndWeight(idealV[i], buffer, v[i],
            freqBands);
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

  std::vector<double> doubleA(roundedA.size());
  for (std::size_t i = 0u; i < roundedA.size(); ++i)
    doubleA[i] = roundedA[i].toDouble();


  double naiveNorm;
  std::vector<double> bandNorms(chebyBands.size());
  computeDenseNorm(naiveNorm, bandNorms, chebyBands, grid, doubleA);
  std::cout << "Naive rounding error\t= " << naiveNorm << std::endl;

  std::vector<std::vector<double>> svpVectors(freeA.size());
  std::vector<std::vector<mpfr::mpreal>> mpSVPVectors(freeA.size());
  std::vector<double> lllA1(freeA.size());
  std::vector<double> lllA2(freeA.size());
  auto start = std::chrono::steady_clock::now();
  fpminimaxKernelV2(lllA1, lllA2, freeADouble, v, svpVectors, fv, idealV, scalingFactor, weights,
                  freeA.size(), prec);
  auto stop = std::chrono::steady_clock::now();
  auto diff = stop - start;
  std::cout << "\nReduction = " << std::chrono::duration<double,std::milli>(diff).count() << " ms\n";

  for (std::size_t i = 0u; i < svpVectors.size(); ++i) {
    mpSVPVectors[i].resize(svpVectors[i].size());
    for (std::size_t j = 0u; j < svpVectors[i].size(); ++j) {
      mpSVPVectors[i][j] = svpVectors[i][j];
    }
  }

  // using minimax
  std::vector<mpfr::mpreal> mpLLLA1(freeA.size());

  for (std::size_t i = 0u; i < freeA.size(); ++i)
    mpLLLA1[i] = lllA1[i];



  for (std::size_t i = 0u; i < fixedA.size(); ++i)
    mpLLLA1.push_back(fixedA[i]);
  start = std::chrono::steady_clock::now();

  double lllNorm;
  computeDenseNorm(lllNorm, bandNorms, chebyBands, grid, lllA1);
  std::cout << "LLL initial error\t= " << lllNorm << std::endl;
  for (std::size_t k{0u}; k < bandNorms.size(); ++k)
    std::cout << "Band " << k << " error = " << bandNorms[k] << std::endl;


  double lllBestNorm;
  std::vector<mpfr::mpreal> mpBufferA(lllA1.size() + fixedA.size());
  std::vector<mpfr::mpreal> mpFinalA1;
  for (std::size_t i = lllA1.size(); i < lllA1.size() + fixedA.size(); ++i)
    mpBufferA[i] = fixedA[i - lllA1.size()];
  mpFinalA1 = mpBufferA;
  for (std::size_t i = 0u; i < lllA1.size(); ++i)
    mpFinalA1[i] = lllA1[i];
  std::vector<mpfr::mpreal> initialLLL1 = mpLLLA1;
  std::vector<double> searchBandNorms(chebyBands.size());

  start = std::chrono::steady_clock::now();
  stop = std::chrono::steady_clock::now();
  diff = stop - start;

  std::random_device r;

  std::default_random_engine e(r());
  std::uniform_int_distribution<int> ud1(0, 3);
  std::uniform_int_distribution<int> ud2(0, mpLLLA1.size()-1);
  std::uniform_int_distribution<int> ud3(-1,1);


  while(std::chrono::duration<double,std::milli>(diff).count() < 60000.0)
  {
    std::size_t index1 = ud1(e);
    std::size_t index2 = ud2(e);

    int direction1 = ud3(e);
    int direction2 = ud3(e);


          for (std::size_t i{0u}; i < freeA.size(); ++i) {
            double vA = lllA1[i] + svpVectors[index1][i] * direction1 +
                        svpVectors[index2][i] *
                            direction2;
            mpBufferA[i] = vA;
            doubleA[i] = vA;
          }
          double bufferNorm;
          computeDenseNorm(bufferNorm, searchBandNorms, chebyBands, grid, doubleA);

          if (bufferNorm < lllNorm) {
            lllNorm = bufferNorm;
            bandNorms = searchBandNorms;
            std::cout << "Index1 = " << index1 << " Direction1 = " << direction1
                      << std::endl;
            std::cout << "Index2 = " << index2 << " Direction2 = " << direction2
                      << std::endl;

            std::cout << "LLL search norm\t= " << bufferNorm << std::endl;
            for (std::size_t k{0u}; k < bandNorms.size(); ++k)
              std::cout << "Band " << k << " error = " << bandNorms[k] << std::endl;

            mpFinalA1 = mpBufferA;
          }
        stop = std::chrono::steady_clock::now();
        diff = stop - start;


  }


  lllFreeA.resize(freeA.size());
  for (std::size_t i = 0u; i < freeA.size(); ++i)
    lllFreeA[i] = mpFinalA1[i];

  for (std::size_t i = 0u; i < lllFreeA.size(); ++i) {
    mpfr::mpreal roundedBuffer = roundedA[i];
    mpfr::mpreal finalBuffer = mpFinalA1[i];
    mpfr::mpreal initBuffer = initialLLL1[i];
    if (i > 0u) {
      roundedBuffer *= scalingFactor;
      finalBuffer *= scalingFactor;
      initBuffer *= scalingFactor;
      roundedBuffer >>= 1;
      finalBuffer >>= 1;
      initBuffer >>= 1;

      mpLLLA1[i] = finalBuffer / scalingFactor;
      mpLLLA1[i] *= 2;
    } else {
      roundedBuffer *= scalingFactor;
      finalBuffer *= scalingFactor;
      initBuffer *= scalingFactor;
      roundedBuffer >>= 0;
      finalBuffer >>= 0;
      initBuffer >>= 0;

      mpLLLA1[i] = finalBuffer / (scalingFactor);
      mpLLLA1[i] *= 1;
    }
    finalBuffer = finalBuffer.toLong(GMP_RNDN);
  }



  for(std::size_t i{0u}; i < mpLLLA1.size(); ++i)
    doubleA[i] = mpLLLA1[i].toDouble();

  double lllNorm1;
  computeDenseNorm(lllNorm1, bandNorms, chebyBands, grid, doubleA);
  std::cout << "LLL final error\t= " << lllNorm << std::endl;
  stop = std::chrono::steady_clock::now();
  diff = stop - start;
  std::cout << "\nVicinity search = " << std::chrono::duration<double,std::milli>(diff).count() << " ms\n";


  // using ideal
  std::vector<mpfr::mpreal> mpLLLA2(freeA.size());

  for (std::size_t i = 0u; i < freeA.size(); ++i)
    mpLLLA2[i] = lllA2[i];



  for (std::size_t i = 0u; i < fixedA.size(); ++i)
    mpLLLA2.push_back(fixedA[i]);
  start = std::chrono::steady_clock::now();

  computeDenseNorm(lllNorm, bandNorms, chebyBands, grid, lllA2);
  std::cout << "LLL initial error\t= " << lllNorm << std::endl;
  for (std::size_t k{0u}; k < bandNorms.size(); ++k)
    std::cout << "Band " << k << " error = " << bandNorms[k] << std::endl;


  std::vector<mpfr::mpreal> mpFinalA2;
  for (std::size_t i = lllA2.size(); i < lllA2.size() + fixedA.size(); ++i)
    mpBufferA[i] = fixedA[i - lllA2.size()];
  mpFinalA2 = mpBufferA;
  for (std::size_t i = 0u; i < lllA2.size(); ++i)
    mpFinalA2[i] = lllA2[i];
  std::vector<mpfr::mpreal> initialLLL2 = mpLLLA2;


  start = std::chrono::steady_clock::now();
  stop = std::chrono::steady_clock::now();
  diff = stop - start;

  while(std::chrono::duration<double,std::milli>(diff).count() < 60000)
  {
    std::size_t index1 = ud1(e);
    std::size_t index2 = ud2(e);

    int direction1 = ud3(e);
    int direction2 = ud3(e);

          for (std::size_t i{0u}; i < freeA.size(); ++i) {
            double vA = lllA2[i] + svpVectors[index1][i] * direction1 +
                        svpVectors[index2][i] *
                            direction2;
            mpBufferA[i] = vA;
            doubleA[i] = vA;
          }
          double bufferNorm;
          computeDenseNorm(bufferNorm, searchBandNorms, chebyBands, grid, doubleA);

          if (bufferNorm < lllNorm) {
            lllNorm = bufferNorm;
            bandNorms = searchBandNorms;
            std::cout << "Index1 = " << index1 << " Direction1 = " << direction1
                      << std::endl;
            std::cout << "Index2 = " << index2 << " Direction2 = " << direction2
                      << std::endl;

            std::cout << "LLL search norm\t= " << bufferNorm << std::endl;
            for (std::size_t k{0u}; k < bandNorms.size(); ++k)
              std::cout << "Band " << k << " error = " << bandNorms[k] << std::endl;

            mpFinalA2 = mpBufferA;
          }
        stop = std::chrono::steady_clock::now();
        diff = stop - start;


  }



  for (std::size_t i = 0u; i < lllFreeA.size(); ++i) {
    mpfr::mpreal roundedBuffer = roundedA[i];
    mpfr::mpreal finalBuffer = mpFinalA2[i];
    mpfr::mpreal initBuffer = initialLLL2[i];
    if (i > 0u) {
      roundedBuffer *= scalingFactor;
      finalBuffer *= scalingFactor;
      initBuffer *= scalingFactor;
      roundedBuffer >>= 1;
      finalBuffer >>= 1;
      initBuffer >>= 1;

      mpLLLA2[i] = finalBuffer / scalingFactor;
      mpLLLA2[i] *= 2;
    } else {
      roundedBuffer *= scalingFactor;
      finalBuffer *= scalingFactor;
      initBuffer *= scalingFactor;
      roundedBuffer >>= 0;
      finalBuffer >>= 0;
      initBuffer >>= 0;

      mpLLLA2[i] = finalBuffer / (scalingFactor);
      mpLLLA2[i] *= 1;
    }
    finalBuffer = finalBuffer.toLong(GMP_RNDN);
  }
  stop = std::chrono::steady_clock::now();
  diff = stop - start;
  std::cout << "\nVicinity search = " << std::chrono::duration<double,std::milli>(diff).count() << " ms\n";



  double bestNorm = lllNorm1;
  std::vector<double> bestA = doubleA;
  std::vector<mpfr::mpreal> mpLLLA = mpLLLA1;
  for(std::size_t i{0u}; i < mpLLLA2.size(); ++i)
    doubleA[i] = mpLLLA2[i].toDouble();

  double lllNorm2;
  computeDenseNorm(lllNorm2, bandNorms, chebyBands, grid, doubleA);
  std::cout << "LLL final error\t= " << lllNorm << std::endl;

  if(lllNorm2 < lllNorm1)
  {
      bestA = doubleA;
      mpLLLA = mpLLLA2;
      bestNorm = lllNorm2;
      lllFreeA.resize(freeA.size());
      for (std::size_t i = 0u; i < freeA.size(); ++i)
        lllFreeA[i] = mpFinalA2[i];
  }
}







void fpminimaxWithNeighborhoodSearchDiscreteFull(
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
    mpfr::mpreal buffer;
    computeIdealResponseAndWeight(fv[i], buffer, v[i],
            freqBands);
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

  std::vector<double> doubleA(roundedA.size());
  for (std::size_t i = 0u; i < roundedA.size(); ++i)
    doubleA[i] = roundedA[i].toDouble();


  double naiveNorm;
  std::vector<double> bandNorms(chebyBands.size());
  computeDenseNorm(naiveNorm, bandNorms, chebyBands, grid, doubleA);
  std::cout << "Naive rounding error\t= " << naiveNorm << std::endl;

  std::vector<std::vector<double>> svpVectors(freeA.size());
  std::vector<std::vector<mpfr::mpreal>> mpSVPVectors(freeA.size());
  std::vector<double> lllA(freeA.size());

  auto start = std::chrono::steady_clock::now();
  fpminimaxKernel(lllA, freeADouble, v, svpVectors, fv, scalingFactor, weights,
                  freeA.size(), prec);
  auto stop = std::chrono::steady_clock::now();
  auto diff = stop - start;
  std::cout << "\nReduction = " << std::chrono::duration<double,std::milli>(diff).count() << " ms\n";


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


  double lllNorm;
  computeDenseNorm(lllNorm, bandNorms, chebyBands, grid, lllA);
  std::cout << "LLL initial error\t= " << lllNorm << std::endl;
  for (std::size_t k{0u}; k < bandNorms.size(); ++k)
    std::cout << "Band " << k << " error = " << bandNorms[k] << std::endl;


  double lllBestNorm;
  std::vector<mpfr::mpreal> mpBufferA(lllA.size() + fixedA.size());
  std::vector<mpfr::mpreal> mpFinalA;
  for (std::size_t i = lllA.size(); i < lllA.size() + fixedA.size(); ++i)
    mpBufferA[i] = fixedA[i - lllA.size()];
  mpFinalA = mpBufferA;
  for (std::size_t i = 0u; i < lllA.size(); ++i)
    mpFinalA[i] = lllA[i];
  std::vector<mpfr::mpreal> initialLLL = mpLLLA;
  std::vector<double> searchBandNorms(chebyBands.size());




  start = std::chrono::steady_clock::now();


  for (std::size_t index1 = 0u; index1 < 9; ++index1)
    for (std::size_t index2 = index1 + 1u; index2 < mpLLLA.size(); ++index2)

      for (int direction1 = -1; direction1 < 2; ++direction1)
        for (int direction2 = -1; direction2 < 2; ++direction2)

        {
          for (std::size_t i{0u}; i < freeA.size(); ++i) {
            double vA = lllA[i] + svpVectors[index1][i] * direction1 +
                        svpVectors[index2][i] *
                            direction2;
            mpBufferA[i] = vA;
            doubleA[i] = vA;
          }
          double bufferNorm;
          computeDenseNorm(bufferNorm, searchBandNorms, chebyBands, grid, doubleA);

          if (bufferNorm < lllNorm) {
            lllNorm = bufferNorm;
            bandNorms = searchBandNorms;

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
  stop = std::chrono::steady_clock::now();
  diff = stop - start;
  std::cout << "\nVicinity search = " << std::chrono::duration<double,std::milli>(diff).count() << " ms\n";



  for(std::size_t i{0u}; i < mpLLLA.size(); ++i)
  	doubleA[i] = mpLLLA[i].toDouble();


  computeDenseNorm(lllNorm, bandNorms, chebyBands, grid, doubleA);
  std::cout << "LLL final error\t= " << lllNorm << std::endl;

  mpfr::mpreal buffInit = 1u;
  buffInit /= scalingFactor;
  mpfr::mpreal buffRest = 2u;
  buffRest /= scalingFactor;
  std::vector<double> buffA;
  std::vector<double> bestA = doubleA;
  double bestNorm = lllNorm;

  for(std::size_t i{0u}; i < mpLLLA.size(); ++i)
      {
        buffA = doubleA;
        if(i == 0u)
            buffA[i] -= buffInit.toDouble();
        else
            buffA[i] -= buffRest.toDouble();
          double bufferNorm;
          computeDenseNorm(bufferNorm, searchBandNorms, chebyBands, grid, buffA);
          if(bufferNorm < bestNorm)
          {
              bestA = buffA;
              bestNorm = bufferNorm;
              std::cout << "New error = " << bufferNorm << std::endl;
          }


        buffA = doubleA;
        if(i == 0u)
            buffA[i] += buffInit.toDouble();
        else
            buffA[i] += buffRest.toDouble();


          computeDenseNorm(bufferNorm, searchBandNorms, chebyBands, grid, buffA);
          if(bufferNorm < bestNorm)
          {
              bestA = buffA;
              bestNorm = bufferNorm;
              std::cout << "New error = " << bufferNorm << std::endl;
          }

       }



  for(std::size_t i{0u}; i < mpLLLA.size() - 1; ++i)
      for(std::size_t j{i + 1u}; j < mpLLLA.size(); ++j)
      {
        buffA = doubleA;
        if(i == 0u)
            buffA[i] -= buffInit.toDouble();
        else
            buffA[i] -= buffRest.toDouble();
        buffA[j] -= buffRest.toDouble();
          double bufferNorm;
          computeDenseNorm(bufferNorm, searchBandNorms, chebyBands, grid, buffA);
          if(bufferNorm < bestNorm)
          {
              bestA = buffA;
              bestNorm = bufferNorm;
              std::cout << "New error = " << bufferNorm << std::endl;
          }


        buffA = doubleA;
        if(i == 0u)
            buffA[i] -= buffInit.toDouble();
        else
            buffA[i] -= buffRest.toDouble();
        buffA[j] += buffRest.toDouble();

          computeDenseNorm(bufferNorm, searchBandNorms, chebyBands, grid, buffA);
          if(bufferNorm < bestNorm)
          {
              bestA = buffA;
              bestNorm = bufferNorm;
              std::cout << "New error = " << bufferNorm << std::endl;
          }


        buffA = doubleA;
        if(i == 0u)
            buffA[i] += buffInit.toDouble();
        else
            buffA[i] += buffRest.toDouble();
        buffA[j] -= buffRest.toDouble();

          computeDenseNorm(bufferNorm, searchBandNorms, chebyBands, grid, buffA);
          if(bufferNorm < bestNorm)
          {
              bestA = buffA;
              bestNorm = bufferNorm;
              std::cout << "New error = " << bufferNorm << std::endl;
          }


        buffA = doubleA;
        if(i == 0u)
            buffA[i] += buffInit.toDouble();
        else
            buffA[i] += buffRest.toDouble();
        buffA[j] += buffRest.toDouble();


          computeDenseNorm(bufferNorm, searchBandNorms, chebyBands, grid, buffA);
          if(bufferNorm < bestNorm)
          {
              bestA = buffA;
              bestNorm = bufferNorm;
              std::cout << "New error = " << bufferNorm << std::endl;
          }

       }


}


void fpminimaxWithNeighborhoodSearchDiscreteMinimax(
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
  std::vector<mpfr::mpreal> idealV(interpolationPoints.size());
  for (std::size_t i = 0u; i < interpolationPoints.size(); ++i) {
    v[i] = mpfr::acos(interpolationPoints[i]);
    for(std::size_t j{0u}; j < freqBands.size(); ++j) {
        if (mpfr::abs(v[i] - freqBands[j].stop) < 1e-14)
            v[i] = freqBands[j].stop;
        if (mpfr::abs(v[i] - freqBands[j].start) < 1e-14)
            v[i] = freqBands[j].start;
    }

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

  std::vector<double> doubleA(roundedA.size());
  for (std::size_t i = 0u; i < roundedA.size(); ++i)
    doubleA[i] = roundedA[i].toDouble();

  double naiveNorm;
  std::vector<double> bandNorms(chebyBands.size());
  computeDenseNorm(naiveNorm, bandNorms, chebyBands, grid, doubleA);
  std::cout << "Naive rounding error\t= " << naiveNorm << std::endl;

  std::vector<std::vector<double>> svpVectors(freeA.size());
  std::vector<std::vector<mpfr::mpreal>> mpSVPVectors(freeA.size());
  std::vector<double> lllA(freeA.size());

  auto start = std::chrono::steady_clock::now();
  fpminimaxKernel(lllA, freeADouble, v, svpVectors, fv, scalingFactor, weights,
                  freeA.size(), prec);
  auto stop = std::chrono::steady_clock::now();
  auto diff = stop - start;
  std::cout << "\nReduction = " << std::chrono::duration<double,std::milli>(diff).count() << " ms\n";


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


  double lllNorm;
  computeDenseNorm(lllNorm, bandNorms, chebyBands, grid, lllA);
  std::cout << "LLL initial error\t= " << lllNorm << std::endl;
  for (std::size_t k{0u}; k < bandNorms.size(); ++k)
    std::cout << "Band " << k << " error = " << bandNorms[k] << std::endl;

  double lllBestNorm;
  std::vector<mpfr::mpreal> mpBufferA(lllA.size() + fixedA.size());
  std::vector<mpfr::mpreal> mpFinalA;
  for (std::size_t i = lllA.size(); i < lllA.size() + fixedA.size(); ++i)
    mpBufferA[i] = fixedA[i - lllA.size()];
  mpFinalA = mpBufferA;
  for (std::size_t i = 0u; i < lllA.size(); ++i)
    mpFinalA[i] = lllA[i];
  std::vector<mpfr::mpreal> initialLLL = mpLLLA;
  std::vector<double> searchBandNorms(chebyBands.size());




  start = std::chrono::steady_clock::now();


  for (std::size_t index1 = 0u; index1 < 9; ++index1)
    for (std::size_t index2 = index1 + 1u; index2 < mpLLLA.size(); ++index2)

      for (int direction1 = -1; direction1 < 2; ++direction1)
        for (int direction2 = -1; direction2 < 2; ++direction2)
        {
          for (std::size_t i{0u}; i < freeA.size(); ++i) {
            double vA = lllA[i] + svpVectors[index1][i] * direction1 +
                        svpVectors[index2][i] * direction2;
            mpBufferA[i] = vA;
            doubleA[i] = vA;
          }
          double bufferNorm;
          computeDenseNorm(bufferNorm, searchBandNorms, chebyBands, grid, doubleA);

          if (bufferNorm < lllNorm) {
            lllNorm = bufferNorm;
            bandNorms = searchBandNorms;

            mpFinalA = mpBufferA;
          }
        }



  lllFreeA.resize(freeA.size());
  for (std::size_t i = 0u; i < freeA.size(); ++i)
    lllFreeA[i] = mpFinalA[i];

  for (std::size_t i = 0u; i < lllFreeA.size(); ++i) {
    mpfr::mpreal finalBuffer = mpFinalA[i];
    if (i > 0u) {
      finalBuffer *= scalingFactor;
      finalBuffer >>= 1;

      mpLLLA[i] = finalBuffer / scalingFactor;
      mpLLLA[i] *= 2;
    } else {
      finalBuffer *= scalingFactor;
      finalBuffer >>= 0;

      mpLLLA[i] = finalBuffer / (scalingFactor);
      mpLLLA[i] *= 1;
    }
    finalBuffer = finalBuffer.toLong(GMP_RNDN);
  }
  stop = std::chrono::steady_clock::now();
  diff = stop - start;
  std::cout << "\nVicinity search = " << std::chrono::duration<double,std::milli>(diff).count() << " ms\n";


  for(std::size_t i{0u}; i < mpLLLA.size(); ++i)
    doubleA[i] = mpLLLA[i].toDouble();

  computeDenseNorm(lllNorm, bandNorms, chebyBands, grid, doubleA);
  std::cout << "LLL final error\t= " << lllNorm << std::endl;
}
