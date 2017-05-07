#include "filter/roots.h"
#include "filter/pm.h"
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <fstream>

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

  //for(auto& it : subIntervals)
  //  std::cout << "[" << it.first << ", " << it.second << "];\n";

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
                    std::vector<mpfr::mpreal> &refs,
                    std::vector<Band> &freqBands,
                    std::vector<Band> &chebyBands,
                    mpfr_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpfr::mpreal ia = -1;
  mpfr::mpreal ib = 1;

  // TODO: this is a little bit inefficient, it should
  // be rewritten!
  std::vector<Interval> subIntervals;
  std::vector<std::vector<mpfr::mpreal>> refPartition(chebyBands.size());
  for(std::size_t i{0u}; i < refs.size(); ++i)
      for(std::size_t j{0u}; j < chebyBands.size(); ++j)
      {
        if(refs[i] >= chebyBands[j].start && refs[i] <= chebyBands[j].stop)
            refPartition[j].push_back(refs[i]);
      }

  for(std::size_t i{0u}; i < chebyBands.size(); ++i)
  {
      if(refPartition[i][0] > chebyBands[i].start)
          subIntervals.push_back(std::make_pair(chebyBands[i].start, refPartition[i][0]));
      for(std::size_t j{0u}; j < refPartition[i].size() - 1; ++j)
          subIntervals.push_back(std::make_pair(refPartition[i][j], refPartition[i][j + 1u]));
      if(refPartition[i][refPartition[i].size() - 1u] < chebyBands[i].stop)
          subIntervals.push_back(std::make_pair(refPartition[i][refPartition[i].size() - 1u], chebyBands[i].stop));
  }

  std::size_t maxSize = 16;
  std::vector<mpfr::mpreal> chebyNodes(maxSize + 1);
  generateEquidistantNodes(chebyNodes, maxSize, prec);
  applyCos(chebyNodes, chebyNodes);

  std::vector<std::vector<mpfr::mpreal>> intervalZeros(subIntervals.size());
  #pragma omp parallel for
  for (std::size_t i = 0u; i < subIntervals.size(); ++i) {
    std::vector<mpfr::mpreal> siCN(maxSize + 1);
    changeOfVariable(siCN, chebyNodes, subIntervals[i].first,
                     subIntervals[i].second);

    std::vector<mpfr::mpreal> fx(maxSize + 1);
    for (std::size_t j = 0u; j < fx.size(); ++j)
      getError(fx[j], chebyBands, siCN[j], a, prec);

    std::vector<mpfr::mpreal> chebyCoeffs(maxSize + 1);
    generateChebyshevCoefficients(chebyCoeffs, fx, maxSize, prec);

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
  for (std::size_t i = 0u; i < intervalZeros.size(); ++i)
    for (std::size_t j = 0u; j < intervalZeros[i].size(); ++j)
      zeros.push_back(intervalZeros[i][j]);

  std::sort(zeros.begin(), zeros.end(),
            [](const mpfr::mpreal &lhs, const mpfr::mpreal &rhs) {
              return lhs < rhs;
            });

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

void generateGridPoints(std::vector<double> &grid, std::size_t degree,
                        std::vector<BandD> &freqBands) {

  double increment = M_PI;
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
}

void uniformSplit(std::vector<IntervalD> &subIntervals, const std::size_t N,
                  std::vector<BandD> &chebyBands) {


  double totalSize = 0;
  for (std::size_t i = 0u; i < chebyBands.size(); ++i)
    totalSize += (chebyBands[i].stop - chebyBands[i].start);

  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    double bandSI = (chebyBands[i].stop - chebyBands[i].start) / totalSize;
    bandSI *= (intervalDensity * N);
    bandSI = ceil(bandSI);
    std::size_t siCount = (std::size_t)bandSI;
    double leftEdge = chebyBands[i].start;

    double siWidth = (chebyBands[i].stop - chebyBands[i].start) / siCount;
    for (int j = 0; j < (int)siCount - 1; ++j) {
      subIntervals.push_back(std::make_pair(leftEdge, leftEdge + siWidth));
      leftEdge += siWidth;
    }
    subIntervals.push_back(std::make_pair(leftEdge, chebyBands[i].stop));
  }

}

void getError(double &error, std::vector<BandD> &chebyBands,
              double &x, std::vector<double> &a) {
  evaluateClenshaw(error, a, x);
  double D, W;
  computeIdealResponseAndWeight(D, W, x, chebyBands);
  error = W * (D - error);
}

void computeDenseNorm(double &normValue, std::vector<BandD> &chebyBands,
                      std::vector<double> &a,
                      std::vector<double> &grid) {

  double currentMax, currentValue;
  normValue = 0;
  for (auto &it : grid) {
    currentValue = cos(it);
    getError(currentMax, chebyBands, currentValue, a);
    currentMax = fabs(currentMax);
    if (currentMax > normValue)
      normValue = currentMax;
  }

}

void findEigenZeros(std::vector<double> &a,
                    std::vector<double> &zeros,
                    std::vector<BandD> &freqBands, std::vector<BandD> &chebyBands)
{
  double ia = -1;
  double ib = 1;

  std::vector<IntervalD> subIntervals;
  uniformSplit(subIntervals, a.size(), chebyBands);

  std::vector<double> chebyNodes(maxDegree + 1);
  generateEquidistantNodes(chebyNodes, maxDegree);
  applyCos(chebyNodes, chebyNodes);

  std::vector<std::vector<double>> intervalZeros(subIntervals.size());
#pragma omp parallel for
  for (std::size_t i = 0u; i < subIntervals.size(); ++i) {
    std::vector<double> siCN(maxDegree + 1);
    changeOfVariable(siCN, chebyNodes, subIntervals[i].first,
                     subIntervals[i].second);

    std::vector<double> fx(maxDegree + 1);
    for (std::size_t j = 0u; j < fx.size(); ++j)
      getError(fx[j], chebyBands, siCN[j], a);

    std::vector<double> chebyCoeffs(maxDegree + 1);
    generateChebyshevCoefficients(chebyCoeffs, fx, maxDegree);

    // zero-free subinterval testing
    double B0 = 0;
    for (std::size_t j = 1u; j < chebyCoeffs.size(); ++j)
      B0 += fabs(chebyCoeffs[j]);
    if (B0 >= fabs(chebyCoeffs[0])) {
      MatrixXd Cm(maxDegree, maxDegree);
      generateColleagueMatrix1stKind(Cm, chebyCoeffs, true);
      std::vector<double> eigenRoots;
      VectorXcd roots;
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
            [](const double &lhs, const double &rhs) {
              return lhs < rhs;
            });
}


void plotFunc(std::string &filename, std::function<double(double)> &f,
              double &a, double &b) {
  std::setprecision(std::numeric_limits<long double>::digits10 + 1);
  std::stringstream datFilename;
  datFilename << filename << "_0.dat";

  std::ofstream output;
  output.open(datFilename.str().c_str());
  double width = b - a;
  double buffer, bufferValue;
  std::size_t pointCount = 5000u;
  for (std::size_t i = 0u; i < pointCount; ++i) {
    buffer = a + (width * i) / pointCount;
    output << buffer << "\t";
    bufferValue = f(buffer);
    output << bufferValue << std::endl;
  }
  output << b << "\t";
  bufferValue = f(b);
  output << bufferValue << std::endl;

  output.close();

  std::stringstream gnuplotFile;
  gnuplotFile << filename << "_f.p";
  std::stringstream epsFilename;
  epsFilename << filename << "_f.eps";
  output.open(gnuplotFile.str().c_str());

  output << "set terminal postscript eps color\n"
         << R"(set out ")" << epsFilename.str() << R"(")" << std::endl
         << R"(set format x "%g")" << std::endl
         << R"(set format y "%g")" << std::endl
         << "set xrange [" << a << ":" << b << "]\n"
         << R"(plot ")" << datFilename.str() << R"(" using 1:2 with lines t "")"
         << std::endl;

  std::string gnuplotCommand = "gnuplot ";
  gnuplotCommand += gnuplotFile.str();
  system(gnuplotCommand.c_str());
}

void plotFuncEtVals(std::string &filename, std::function<double(double)> &f,
                    std::vector<std::pair<double, double>> &p, double &a,
                    double &b) {
  std::setprecision(std::numeric_limits<long double>::digits10 + 1);
  std::stringstream datFilename;
  datFilename << filename << "_0.dat";

  std::ofstream output;
  output.open(datFilename.str().c_str());
  double width = b - a;
  double buffer, bufferValue;
  std::size_t pointCount = 10000u;
  for (std::size_t i = 0u; i < pointCount; ++i) {
    buffer = a + (width * i) / pointCount;
    output << buffer << "\t";
    bufferValue = f(buffer);
    output << bufferValue << std::endl;
  }
  output << b << "\t";
  bufferValue = f(b);
  output << bufferValue << std::endl;

  output.close();

  std::stringstream pointFilename;
  pointFilename << filename << "_1.dat";

  std::ofstream output2;
  output2.open(pointFilename.str().c_str());
  for (std::size_t i{0u}; i < p.size(); ++i)
    output2 << p[i].first << "\t" << p[i].second << std::endl;

  output2.close();

  std::stringstream gnuplotFile;
  gnuplotFile << filename << "_f.p";
  std::stringstream epsFilename;
  epsFilename << filename << "_f.eps";
  output.open(gnuplotFile.str().c_str());

  output << "set terminal postscript eps color\n"
         << R"(set out ")" << epsFilename.str() << R"(")" << std::endl
         << R"(set format x "%g")" << std::endl
         << R"(set format y "%g")" << std::endl
         << "set xrange [" << a << ":" << b << "]\n"
         << R"(plot ")" << datFilename.str()
         << R"(" using 1:2 with lines t "", \)" << std::endl
         << "\t"
         << R"(")" << pointFilename.str() << R"(" using 1:2 with points t "")"
         << std::endl;

  std::string gnuplotCommand = "gnuplot ";
  gnuplotCommand += gnuplotFile.str();
  system(gnuplotCommand.c_str());
}

void plotFuncs(std::string &filename,
               std::vector<std::function<double(double)>> &fs, double &a,
               double &b) {

  std::setprecision(std::numeric_limits<long double>::digits10 + 1);
  std::vector<std::stringstream> datFilename(fs.size());
  for (std::size_t i{0u}; i < fs.size(); ++i) {
    datFilename[i] << filename << "_" << (i + 1u) << ".dat";
    std::ofstream output;
    output.open(datFilename[i].str().c_str());
    double width = b - a;
    double buffer, bufferValue;
    std::size_t pointCount = 5000u;
    for (std::size_t j{0u}; j < pointCount; ++j) {
      buffer = a + (width * j) / pointCount;
      output << buffer << "\t";
      bufferValue = fs[i](buffer);
      output << bufferValue << std::endl;
    }
    output << b << "\t";
    bufferValue = fs[i](b);
    output << bufferValue << std::endl;

    output.close();
  }

  std::stringstream gnuplotFile;
  gnuplotFile << filename << "_f.p";
  std::stringstream epsFilename;
  epsFilename << filename << "_f.eps";
  std::ofstream output;
  output.open(gnuplotFile.str().c_str());

  output << "set terminal postscript eps color\n"
         << R"(set out ")" << epsFilename.str() << R"(")" << std::endl
         << R"(set format x "%g")" << std::endl
         << R"(set format y "%g")" << std::endl
         << "set xrange [" << a << ":" << b << "]\n"
         << R"(plot ")" << datFilename[0].str()
         << R"(" using 1:2 with lines t "", \)" << std::endl;

  for (std::size_t i{1u}; i < fs.size() - 1; ++i) {
    output << "\t"
           << R"(")" << datFilename[i].str()
           << R"(" using 1:2 with lines t "", \)" << std::endl;
  }
  if (fs.size() > 1u)
    output << "\t"
           << R"(")" << datFilename[fs.size() - 1].str()
           << R"(" using 1:2 with lines t "")" << std::endl;

  output.close();

  std::string gnuplotCommand = "gnuplot ";
  gnuplotCommand += gnuplotFile.str();
  system(gnuplotCommand.c_str());
}

void plotChebPoly(std::string &filename, std::vector<double> &coeff, double &a,
                  double &b) {

  std::setprecision(std::numeric_limits<long double>::digits10 + 1);
  std::function<double(double)> f = [&](double x) -> double {
    double result;
    evaluateClenshaw(result, coeff, x);
    return result;
  };
  plotFunc(filename, f, a, b);
}

void plotChebPolys(std::string &filename,
                   std::vector<std::vector<double>> &coeffs, double &a,
                   double &b) {
  std::setprecision(std::numeric_limits<long double>::digits10 + 1);
  std::vector<std::function<double(double)>> fs(coeffs.size());

  for (std::size_t i = 0u; i < coeffs.size(); ++i) {
    fs[i] = [&](double x) -> double {
      double result;
      evaluateClenshaw(result, coeffs[i], x);
      return result;
    };
  }
  plotFuncs(filename, fs, a, b);
}

void plotVals(std::string &filename, std::vector<std::pair<double, double>> &p,
              double &a, double &b) {

  std::setprecision(std::numeric_limits<long double>::digits10 + 1);
  std::stringstream datFilename;
  datFilename << filename << "_0.dat";

  std::ofstream output;
  output.open(datFilename.str().c_str());
  for (std::size_t i{0u}; i < p.size(); ++i)
    output << p[i].first << "\t" << p[i].second << std::endl;

  output.close();

  std::stringstream gnuplotFile;
  gnuplotFile << filename << "_f.p";
  std::stringstream epsFilename;
  epsFilename << filename << "_f.eps";
  output.open(gnuplotFile.str().c_str());

  output << "set terminal postscript eps color\n"
         << R"(set out ")" << epsFilename.str() << R"(")" << std::endl
         << R"(set format x "%g")" << std::endl
         << R"(set format y "%g")" << std::endl
         << "set xrange [" << a << ":" << b << "]\n"
         << R"(plot ")" << datFilename.str() << R"(" using 1:2 with lines t "")"
         << std::endl;

  std::string gnuplotCommand = "gnuplot ";
  gnuplotCommand += gnuplotFile.str();
  system(gnuplotCommand.c_str());
}

void findEigenZeros(std::vector<double> &a,
                    std::vector<double> &zeros,
                    std::vector<double> &refs,
                    std::vector<BandD> &freqBands,
                    std::vector<BandD> &chebyBands)
{
  double ia = -1;
  double ib = 1;

  std::size_t maxSize = 4;
  std::vector<IntervalD> subIntervals;
  std::vector<std::vector<double>> refPartition(chebyBands.size());
  for(std::size_t i{0u}; i < refs.size(); ++i)
      for(std::size_t j{0u}; j < chebyBands.size(); ++j)
      {
        if(refs[i] >= chebyBands[j].start && refs[i] <= chebyBands[j].stop)
            refPartition[j].push_back(refs[i]);
      }

  for(std::size_t i{0u}; i < chebyBands.size(); ++i)
  {
      if(fabs(refPartition[i][0] - chebyBands[i].start) < 1e-12)
        refPartition[i][0] = chebyBands[i].start;
      if(fabs(refPartition[i][refPartition[i].size() - 1u] - chebyBands[i].stop) < 1e-12)
        refPartition[i][refPartition[i].size() - 1u] = chebyBands[i].stop;

      if(refPartition[i][0] > chebyBands[i].start)
          subIntervals.push_back(std::make_pair(chebyBands[i].start, refPartition[i][0]));
      for(std::size_t j{0u}; j < refPartition[i].size() - 1; ++j)
          subIntervals.push_back(std::make_pair(refPartition[i][j], refPartition[i][j + 1u]));
      if(refPartition[i][refPartition[i].size() - 1u] < chebyBands[i].stop)
          subIntervals.push_back(std::make_pair(refPartition[i][refPartition[i].size() - 1u], chebyBands[i].stop));
  }

  for(std::size_t i{0u}; i < subIntervals.size(); ++i)
  {
      if(fabs(subIntervals[i].first) < 1e-12)
          subIntervals[i].first = 0;
      if(fabs(subIntervals[i].second) < 1e-12)
          subIntervals[i].second = 0;
  }

  std::vector<double> chebyNodes(maxSize + 1);
  generateEquidistantNodes(chebyNodes, maxSize);
  applyCos(chebyNodes, chebyNodes);

  std::vector<std::vector<double>> intervalZeros(subIntervals.size());
  #pragma omp parallel for
  for (std::size_t i = 0u; i < subIntervals.size(); ++i) {

    std::vector<double> siCN(maxSize + 1);
    changeOfVariable(siCN, chebyNodes, subIntervals[i].first,
                     subIntervals[i].second);

    std::vector<double> fx(maxSize + 1);
    for (std::size_t j = 0u; j < fx.size(); ++j)
      getError(fx[j], chebyBands, siCN[j], a);

    std::vector<double> chebyCoeffs(maxSize + 1);
    generateChebyshevCoefficients(chebyCoeffs, fx, maxSize);

      MatrixXd Cm(maxSize, maxSize);
      generateColleagueMatrix1stKind(Cm, chebyCoeffs, true);
      std::vector<double> eigenRoots;
      VectorXcd roots;
      determineEigenvalues(roots, Cm);
      getRealValues(eigenRoots, roots, ia, ib);
      changeOfVariable(eigenRoots, eigenRoots, subIntervals[i].first,
                       subIntervals[i].second);
      for (std::size_t j = 0u; j < eigenRoots.size(); ++j)
        intervalZeros[i].push_back(eigenRoots[j]);
  }
  for (std::size_t i = 0u; i < intervalZeros.size(); ++i)
    for (std::size_t j = 0u; j < intervalZeros[i].size(); ++j)
      zeros.push_back(intervalZeros[i][j]);

  std::sort(zeros.begin(), zeros.end(),
            [](const double &lhs, const double &rhs) {
              return lhs < rhs;
            });

  std::vector<double> bufferZeros;
  bufferZeros.push_back(zeros[0]);
  for(std::size_t i{1u}; i < zeros.size(); ++i)
  {
    if(fabs(zeros[i] - bufferZeros[bufferZeros.size() - 1u]) > 1e-12)
      bufferZeros.push_back(zeros[i]);
  }

  zeros = bufferZeros;
}


void findEigenExtremas(std::vector<double> &a,
                       std::vector<double> &extremas,
                       std::vector<BandD> &freqBands,
                       std::vector<BandD> &chebyBands) {

  double ia = -1;
  double ib = 1;

  std::vector<IntervalD> subIntervals;
  uniformSplit(subIntervals, a.size(), chebyBands);

  std::vector<std::pair<double, double>> pExtremas;
  double edgeValues;
  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    getError(edgeValues, chebyBands, chebyBands[i].start, a);
    pExtremas.push_back(std::make_pair(chebyBands[i].start, edgeValues));
    getError(edgeValues, chebyBands, chebyBands[i].stop, a);
    pExtremas.push_back(std::make_pair(chebyBands[i].stop, edgeValues));
  }

  std::vector<double> chebyNodes(maxDegree + 1);
  generateEquidistantNodes(chebyNodes, maxDegree);
  applyCos(chebyNodes, chebyNodes);

  std::vector<std::vector<std::pair<double, double>>>
      intervalExtremas(subIntervals.size());
  #pragma omp parallel for
  for (std::size_t i = 0u; i < subIntervals.size(); ++i) {
    std::vector<double> siCN(maxDegree + 1);
    changeOfVariable(siCN, chebyNodes, subIntervals[i].first,
                     subIntervals[i].second);

    std::vector<double> fx(maxDegree + 1);
    for (std::size_t j = 0u; j < fx.size(); ++j)
      getError(fx[j], chebyBands, siCN[j], a);

    std::vector<double> chebyCoeffs(maxDegree + 1);
    generateChebyshevCoefficients(chebyCoeffs, fx, maxDegree);
    std::vector<double> derivCoeffs(maxDegree);
    derivativeCoefficients1stKind(derivCoeffs, chebyCoeffs);

    // zero-free subinterval testing
    double B0 = 0;
    for (std::size_t j = 1u; j < derivCoeffs.size(); ++j)
      B0 += fabs(derivCoeffs[j]);
    if (B0 >= fabs(derivCoeffs[0])) {
      MatrixXd Cm(maxDegree - 1, maxDegree - 1);
      generateColleagueMatrix1stKind(Cm, derivCoeffs, true);
      std::vector<double> eigenRoots;
      VectorXcd roots;
      determineEigenvalues(roots, Cm);
      getRealValues(eigenRoots, roots, ia, ib);
      changeOfVariable(eigenRoots, eigenRoots, subIntervals[i].first,
                       subIntervals[i].second);
      for (std::size_t j = 0u; j < eigenRoots.size(); ++j) {
        getError(B0, chebyBands, eigenRoots[j], a);
        intervalExtremas[i].push_back(std::make_pair(eigenRoots[j], B0));
      }
    }
  }
  for (std::size_t i = 0u; i < intervalExtremas.size(); ++i)
    for (std::size_t j = 0u; j < intervalExtremas[i].size(); ++j)
      pExtremas.push_back(intervalExtremas[i][j]);

  std::sort(pExtremas.begin(), pExtremas.end(),
            [](const std::pair<double, double> &lhs,
               const std::pair<double, double> &rhs) {
              return lhs.first < rhs.first;
            });

  // remove duplicates
  std::vector<std::pair<double, double>> pExtremas2;
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
  std::vector<std::pair<double, double>> alternatingExtrema;
  double minError = INT_MAX;
  double maxError = INT_MIN;
  double absError;

  while (extremaIt < pExtremas2.size()) {
    std::pair<double, double> maxErrorPoint;
    maxErrorPoint = pExtremas2[extremaIt];
    bool sgn1 = std::signbit(maxErrorPoint.second);
    bool sgn2 = std::signbit(pExtremas2[extremaIt + 1].second);
    while (extremaIt < pExtremas2.size() - 1 &&
           sgn1 == sgn2 &&
           monotonicity[extremaIt] == monotonicity[extremaIt + 1]) {
      ++extremaIt;
      if (mpfr::abs(maxErrorPoint.second) <
          mpfr::abs(pExtremas2[extremaIt].second))
        maxErrorPoint = pExtremas2[extremaIt];
    	sgn1 = std::signbit(maxErrorPoint.second);
    	sgn2 = std::signbit(pExtremas2[extremaIt + 1].second);
    }
    extremas.push_back(maxErrorPoint.first);
    ++extremaIt;
  }
}

void computeNorm(std::pair<double, double> &norm,
                 std::vector<std::pair<double, double>> &bandNorms,
                 std::vector<double> &a, std::vector<BandD> &freqBands,
                 std::vector<BandD> &chebyBands) {

  double ia = -1;
  double ib = 1;

  norm.second = 0;
  for (auto &bn : bandNorms)
    bn.second = 0;

  std::vector<IntervalD> subIntervals;
  uniformSplit(subIntervals, a.size(), chebyBands);

  std::vector<std::pair<double, double>> pExtremas;
  double edgeValues;
  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    getError(edgeValues, chebyBands, chebyBands[i].start, a);
    pExtremas.push_back(std::make_pair(chebyBands[i].start, edgeValues));
    getError(edgeValues, chebyBands, chebyBands[i].stop, a);
    pExtremas.push_back(std::make_pair(chebyBands[i].stop, edgeValues));
  }

  std::vector<double> chebyNodes(maxDegree + 1);
  generateEquidistantNodes(chebyNodes, maxDegree);
  applyCos(chebyNodes, chebyNodes);

  std::vector<std::vector<std::pair<double, double>>>
      intervalExtremas(subIntervals.size());
#pragma omp parallel for
  for (std::size_t i = 0u; i < subIntervals.size(); ++i) {
    std::vector<double> siCN(maxDegree + 1);
    changeOfVariable(siCN, chebyNodes, subIntervals[i].first,
                     subIntervals[i].second);

    std::vector<double> fx(maxDegree + 1);
    for (std::size_t j = 0u; j < fx.size(); ++j)
      getError(fx[j], chebyBands, siCN[j], a);

    std::vector<double> chebyCoeffs(maxDegree + 1);
    generateChebyshevCoefficients(chebyCoeffs, fx, maxDegree);
    std::vector<double> derivCoeffs(maxDegree);
    derivativeCoefficients1stKind(derivCoeffs, chebyCoeffs);

    // zero-free subinterval testing
    double B0 = 0;
    for (std::size_t j = 1u; j < derivCoeffs.size(); ++j)
      B0 += fabs(derivCoeffs[j]);
    if (B0 >= fabs(derivCoeffs[0])) {
      MatrixXd Cm(maxDegree - 1, maxDegree - 1);
      generateColleagueMatrix1stKind(Cm, derivCoeffs, true);
      std::vector<double> eigenRoots;
      VectorXcd roots;
      determineEigenvalues(roots, Cm);
      getRealValues(eigenRoots, roots, ia, ib);
      changeOfVariable(eigenRoots, eigenRoots, subIntervals[i].first,
                       subIntervals[i].second);
      for (std::size_t j = 0u; j < eigenRoots.size(); ++j) {
        getError(B0, chebyBands, eigenRoots[j], a);
        intervalExtremas[i].push_back(std::make_pair(eigenRoots[j], B0));
      }
    }
  }
  for (std::size_t i = 0u; i < intervalExtremas.size(); ++i)
    for (std::size_t j = 0u; j < intervalExtremas[i].size(); ++j)
      pExtremas.push_back(intervalExtremas[i][j]);

  std::sort(pExtremas.begin(), pExtremas.end(),
            [](const std::pair<double, double> &lhs,
               const std::pair<double, double> &rhs) {
              return lhs.first < rhs.first;
            });

  for (std::size_t i = 0u; i < pExtremas.size(); ++i) {
    if (norm.second < fabs(pExtremas[i].second))
      norm = pExtremas[i];
    norm.second = fabs(norm.second);
    for (std::size_t j{0u}; j < chebyBands.size(); ++j) {
      if (pExtremas[i].first >= chebyBands[j].start &&
          pExtremas[i].first <= chebyBands[j].stop)
        if (bandNorms[j].second < fabs(pExtremas[i].second)) {
          bandNorms[j] = pExtremas[i];
          bandNorms[j].second = fabs(bandNorms[j].second);
        }
    }
  }
}
