#include "filter/roots.h"
#include "filter/pm.h"
#include <algorithm>

int maxDegree = 8;
int intervalDensity = 16;

int gridDensity = 16;

void generateGridPoints(std::vector<mpfr::mpreal> &grid, std::size_t degree,
                        std::vector<Band> &freqBands, mpfr_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  mpfr::mpreal increment = mpfr::const_pi();
  increment /= (degree * gridDensity);
  int bandIndex = 0;
  while (bandIndex < freqBands.size()) {
    grid.push_back(freqBands[bandIndex].start);
    while (grid.back() <= freqBands[bandIndex].stop) {
      grid.push_back(grid.back() + increment);
    }
    grid[grid.size() - 1] = freqBands[bandIndex].stop;
    ++bandIndex;
  }

  mpreal::set_default_prec(prevPrec);
}

void uniformSplit(std::vector<Interval> &subIntervals, const std::size_t N,
                  std::vector<Band> &chebyBands, mpfr_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  mpreal totalSize = 0;
  for (std::size_t i = 0u; i < chebyBands.size(); ++i)
    totalSize += (chebyBands[i].stop - chebyBands[i].start);

  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    mpreal bandSI = (chebyBands[i].stop - chebyBands[i].start) / totalSize;
    bandSI *= (intervalDensity * N);
    bandSI = ceil(bandSI);
    std::size_t siCount = bandSI.toULong();
    mpreal leftEdge = chebyBands[i].start;

    mpreal siWidth = (chebyBands[i].stop - chebyBands[i].start) / siCount;
    for (int j = 0; j < (int)siCount - 1; ++j) {
      subIntervals.push_back(std::make_pair(leftEdge, leftEdge + siWidth));
      leftEdge += siWidth;
    }
    subIntervals.push_back(std::make_pair(leftEdge, chebyBands[i].stop));
  }

  mpreal::set_default_prec(prevPrec);
}

void getError(mpfr::mpreal &error, std::vector<Band> &chebyBands,
              mpfr::mpreal &x, std::vector<mpfr::mpreal> &a, mpfr_prec_t prec) {
  evaluateClenshaw(error, a, x, prec);
  mpfr::mpreal D, W;
  computeIdealResponseAndWeight(D, W, x, chebyBands);
  error = W * (D - error);
}

void computeDenseNorm(mpfr::mpreal &normValue, std::vector<Band> &chebyBands,
                      std::vector<mpfr::mpreal> &a,
                      std::vector<mpfr::mpreal> &grid, mpfr_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  mpfr::mpreal currentMax, currentValue;
  normValue = 0;
  for (auto &it : grid) {
    currentValue = mpfr::cos(it);
    getError(currentMax, chebyBands, currentValue, a, prec);
    currentMax = mpfr::abs(currentMax);
    if (currentMax > normValue)
      normValue = currentMax;
  }

  mpreal::set_default_prec(prevPrec);
}

void findEigenZeros(std::vector<mpfr::mpreal> &a,
                    std::vector<mpfr::mpreal> &zeros,
                    std::vector<Band> &freqBands, std::vector<Band> &chebyBands,
                    mpfr_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpfr::mpreal ia = -1;
  mpfr::mpreal ib = 1;

  std::vector<Interval> subIntervals;
  uniformSplit(subIntervals, a.size(), chebyBands, prec);

  std::vector<mpfr::mpreal> chebyNodes(maxDegree + 1);
  generateEquidistantNodes(chebyNodes, maxDegree, prec);
  applyCos(chebyNodes, chebyNodes);

  std::vector<std::vector<mpfr::mpreal>> intervalZeros(subIntervals.size());
#pragma omp parallel for
  for (std::size_t i = 0u; i < subIntervals.size(); ++i) {
    std::vector<mpfr::mpreal> siCN(maxDegree + 1);
    changeOfVariable(siCN, chebyNodes, subIntervals[i].first,
                     subIntervals[i].second);

    std::vector<mpfr::mpreal> fx(maxDegree + 1);
    for (std::size_t j = 0u; j < fx.size(); ++j)
      getError(fx[j], chebyBands, siCN[j], a, prec);

    std::vector<mpfr::mpreal> chebyCoeffs(maxDegree + 1);
    generateChebyshevCoefficients(chebyCoeffs, fx, maxDegree, prec);

    // zero-free subinterval testing
    mpfr::mpreal B0 = 0;
    for (std::size_t j = 1u; j < chebyCoeffs.size(); ++j)
      B0 += mpfr::abs(chebyCoeffs[j]);
    if (B0 >= mpfr::abs(chebyCoeffs[0])) {
      MatrixXq Cm(maxDegree, maxDegree);
      generateColleagueMatrix1stKind(Cm, chebyCoeffs, true, prec);
      std::vector<mpfr::mpreal> eigenRoots;
      VectorXcq roots;
      determineEigenvalues(roots, Cm);
      getRealValues(eigenRoots, roots, ia, ib);
      changeOfVariable(eigenRoots, eigenRoots, subIntervals[i].first,
                       subIntervals[i].second);
      for (std::size_t j = 0u; j < eigenRoots.size(); ++j)
        intervalZeros[i].push_back(eigenRoots[j]);
    }
  }
  for (std::size_t i = 0u; i < intervalZeros.size(); ++i)
    for (std::size_t j = 0u; j < intervalZeros[i].size(); ++j)
      zeros.push_back(intervalZeros[i][j]);

  std::sort(zeros.begin(), zeros.end(),
            [](const mpfr::mpreal &lhs, const mpfr::mpreal &rhs) {
              return lhs < rhs;
            });

  mpreal::set_default_prec(prevPrec);
}

void findEigenExtremas(std::vector<mpfr::mpreal> &a,
                       std::vector<mpfr::mpreal> &extremas,
                       std::vector<Band> &freqBands,
                       std::vector<Band> &chebyBands, mpfr_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpfr::mpreal ia = -1;
  mpfr::mpreal ib = 1;

  std::vector<Interval> subIntervals;
  uniformSplit(subIntervals, a.size(), chebyBands, prec);

  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> pExtremas;
  mpfr::mpreal edgeValues;
  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    getError(edgeValues, chebyBands, chebyBands[i].start, a, prec);
    pExtremas.push_back(std::make_pair(chebyBands[i].start, edgeValues));
    getError(edgeValues, chebyBands, chebyBands[i].stop, a, prec);
    pExtremas.push_back(std::make_pair(chebyBands[i].stop, edgeValues));
  }

  std::vector<mpfr::mpreal> chebyNodes(maxDegree + 1);
  generateEquidistantNodes(chebyNodes, maxDegree, prec);
  applyCos(chebyNodes, chebyNodes);

  std::vector<std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>>>
      intervalExtremas(subIntervals.size());
#pragma omp parallel for
  for (std::size_t i = 0u; i < subIntervals.size(); ++i) {
    std::vector<mpfr::mpreal> siCN(maxDegree + 1);
    changeOfVariable(siCN, chebyNodes, subIntervals[i].first,
                     subIntervals[i].second);

    std::vector<mpfr::mpreal> fx(maxDegree + 1);
    for (std::size_t j = 0u; j < fx.size(); ++j)
      getError(fx[j], chebyBands, siCN[j], a, prec);

    std::vector<mpfr::mpreal> chebyCoeffs(maxDegree + 1);
    generateChebyshevCoefficients(chebyCoeffs, fx, maxDegree, prec);
    std::vector<mpfr::mpreal> derivCoeffs(maxDegree);
    derivativeCoefficients1stKind(derivCoeffs, chebyCoeffs);

    // zero-free subinterval testing
    mpfr::mpreal B0 = 0;
    for (std::size_t j = 1u; j < derivCoeffs.size(); ++j)
      B0 += mpfr::abs(derivCoeffs[j]);
    if (B0 >= mpfr::abs(derivCoeffs[0])) {
      MatrixXq Cm(maxDegree - 1, maxDegree - 1);
      generateColleagueMatrix1stKind(Cm, derivCoeffs, true, prec);
      std::vector<mpfr::mpreal> eigenRoots;
      VectorXcq roots;
      determineEigenvalues(roots, Cm);
      getRealValues(eigenRoots, roots, ia, ib);
      changeOfVariable(eigenRoots, eigenRoots, subIntervals[i].first,
                       subIntervals[i].second);
      for (std::size_t j = 0u; j < eigenRoots.size(); ++j) {
        getError(B0, chebyBands, eigenRoots[j], a, prec);
        intervalExtremas[i].push_back(std::make_pair(eigenRoots[j], B0));
      }
    }
  }
  for (std::size_t i = 0u; i < intervalExtremas.size(); ++i)
    for (std::size_t j = 0u; j < intervalExtremas[i].size(); ++j)
      pExtremas.push_back(intervalExtremas[i][j]);

  std::sort(pExtremas.begin(), pExtremas.end(),
            [](const std::pair<mpfr::mpreal, mpfr::mpreal> &lhs,
               const std::pair<mpfr::mpreal, mpfr::mpreal> &rhs) {
              return lhs.first < rhs.first;
            });

  // remove duplicates
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> pExtremas2;
  pExtremas2.push_back(pExtremas[0]);
  for (std::size_t i = 1u; i < pExtremas.size(); ++i) {
    while (pExtremas2.back() == pExtremas[i] && i < pExtremas.size())
      ++i;
    pExtremas2.push_back(pExtremas[i]);
  }

  // remove any spurious potential extrema
  std::vector<int> monotonicity;
  for (std::size_t i = 0u; i < pExtremas.size() - 1; ++i)
    if (pExtremas[i].second < pExtremas[i + 1].second)
      monotonicity.push_back(1);
    else
      monotonicity.push_back(-1);

  std::size_t extremaIt = 0u;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> alternatingExtrema;
  mpfr::mpreal minError = INT_MAX;
  mpfr::mpreal maxError = INT_MIN;
  mpfr::mpreal absError;

  while (extremaIt < pExtremas2.size()) {
    std::pair<mpfr::mpreal, mpfr::mpreal> maxErrorPoint;
    maxErrorPoint = pExtremas2[extremaIt];
    while (extremaIt < pExtremas2.size() - 1 &&
           mpfr::sgn(maxErrorPoint.second) *
                   mpfr::sgn(pExtremas2[extremaIt + 1].second) >
               0 &&
           monotonicity[extremaIt] == monotonicity[extremaIt + 1]) {
      ++extremaIt;
      if (mpfr::abs(maxErrorPoint.second) <
          mpfr::abs(pExtremas2[extremaIt].second))
        maxErrorPoint = pExtremas2[extremaIt];
    }
    extremas.push_back(maxErrorPoint.first);
    ++extremaIt;
  }

  mpreal::set_default_prec(prevPrec);
}

void computeNorm(std::pair<mpfr::mpreal, mpfr::mpreal> &norm,
                 std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> &bandNorms,
                 std::vector<mpfr::mpreal> &a, std::vector<Band> &freqBands,
                 std::vector<Band> &chebyBands, mpfr_prec_t prec) {

  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpfr::mpreal ia = -1;
  mpfr::mpreal ib = 1;

  norm.second = 0;
  for (auto &bn : bandNorms)
    bn.second = 0;

  std::vector<Interval> subIntervals;
  uniformSplit(subIntervals, a.size(), chebyBands, prec);

  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> pExtremas;
  mpfr::mpreal edgeValues;
  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    getError(edgeValues, chebyBands, chebyBands[i].start, a, prec);
    pExtremas.push_back(std::make_pair(chebyBands[i].start, edgeValues));
    getError(edgeValues, chebyBands, chebyBands[i].stop, a, prec);
    pExtremas.push_back(std::make_pair(chebyBands[i].stop, edgeValues));
  }

  std::vector<mpfr::mpreal> chebyNodes(maxDegree + 1);
  generateEquidistantNodes(chebyNodes, maxDegree, prec);
  applyCos(chebyNodes, chebyNodes);

  std::vector<std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>>>
      intervalExtremas(subIntervals.size());
#pragma omp parallel for
  for (std::size_t i = 0u; i < subIntervals.size(); ++i) {
    std::vector<mpfr::mpreal> siCN(maxDegree + 1);
    changeOfVariable(siCN, chebyNodes, subIntervals[i].first,
                     subIntervals[i].second);

    std::vector<mpfr::mpreal> fx(maxDegree + 1);
    for (std::size_t j = 0u; j < fx.size(); ++j)
      getError(fx[j], chebyBands, siCN[j], a, prec);

    std::vector<mpfr::mpreal> chebyCoeffs(maxDegree + 1);
    generateChebyshevCoefficients(chebyCoeffs, fx, maxDegree, prec);
    std::vector<mpfr::mpreal> derivCoeffs(maxDegree);
    derivativeCoefficients1stKind(derivCoeffs, chebyCoeffs);

    // zero-free subinterval testing
    mpfr::mpreal B0 = 0;
    for (std::size_t j = 1u; j < derivCoeffs.size(); ++j)
      B0 += mpfr::abs(derivCoeffs[j]);
    if (B0 >= mpfr::abs(derivCoeffs[0])) {
      MatrixXq Cm(maxDegree - 1, maxDegree - 1);
      generateColleagueMatrix1stKind(Cm, derivCoeffs, true, prec);
      std::vector<mpfr::mpreal> eigenRoots;
      VectorXcq roots;
      determineEigenvalues(roots, Cm);
      getRealValues(eigenRoots, roots, ia, ib);
      changeOfVariable(eigenRoots, eigenRoots, subIntervals[i].first,
                       subIntervals[i].second);
      for (std::size_t j = 0u; j < eigenRoots.size(); ++j) {
        getError(B0, chebyBands, eigenRoots[j], a, prec);
        intervalExtremas[i].push_back(std::make_pair(eigenRoots[j], B0));
      }
    }
  }
  for (std::size_t i = 0u; i < intervalExtremas.size(); ++i)
    for (std::size_t j = 0u; j < intervalExtremas[i].size(); ++j)
      pExtremas.push_back(intervalExtremas[i][j]);

  std::sort(pExtremas.begin(), pExtremas.end(),
            [](const std::pair<mpfr::mpreal, mpfr::mpreal> &lhs,
               const std::pair<mpfr::mpreal, mpfr::mpreal> &rhs) {
              return lhs.first < rhs.first;
            });

  for (std::size_t i = 0u; i < pExtremas.size(); ++i) {
    if (norm.second < mpfr::abs(pExtremas[i].second))
      norm = pExtremas[i];
    norm.second = mpfr::abs(norm.second);
    for (std::size_t j{0u}; j < chebyBands.size(); ++j) {
      if (pExtremas[i].first >= chebyBands[j].start &&
          pExtremas[i].first <= chebyBands[j].stop)
        if (bandNorms[j].second < mpfr::abs(pExtremas[i].second)) {
          bandNorms[j] = pExtremas[i];
          bandNorms[j].second = mpfr::abs(bandNorms[j].second);
        }
    }
  }
}
