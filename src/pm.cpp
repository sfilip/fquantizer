#include "filter/pm.h"
#include "filter/band.h"
#include "filter/barycentric.h"
#include <fstream>
#include <set>

void initUniformExtremas(std::vector<mpfr::mpreal> &omega, std::vector<Band> &B,
                         mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  if (omega.size() <= B.size()) {

    std::size_t chosen = 2u;
    B[0].extremas = 1;
    B[B.size() - 1].extremas = 1;
    omega[0] = B[0].start;
    omega[omega.size() - 1] = B[B.size() - 1].stop;
    while (chosen < omega.size()) {
      omega[chosen - 1] = B[chosen - 1].start;
      B[chosen - 1].extremas = 1;
      ++chosen;
    }
  } else if (omega.size() <= 2 * B.size()) {
    for (std::size_t i{0u}; i < B.size(); ++i) {
      B[i].extremas = 1;
      omega[i] = B[i].start;
    }
    std::size_t chosen = B.size() + 1;
    omega[chosen - 1] = B[B.size() - 1].stop;
    ++B[B.size() - 1].extremas;
    while (chosen < omega.size()) {
      ++B[chosen - 1 - B.size()].extremas;
      omega[chosen] = B[chosen - 1 - B.size()].stop;
      ++chosen;
    }

  } else {

    mpfr::mpreal avgDistance = 0;

    std::vector<mpfr::mpreal> bandwidths(B.size());
    std::vector<std::size_t> nonPointBands;
    for (std::size_t i = 0; i < B.size(); ++i) {
      bandwidths[i] = B[i].stop - B[i].start;
      if (bandwidths[i] > 0.0) {
        nonPointBands.push_back(i);
        avgDistance += bandwidths[i];
      }
      B[i].extremas = 1u;
    }
    if (nonPointBands.empty()) {
      std::cerr << "All intervals are points!\n";
      exit(EXIT_FAILURE);
    }
    // TODO: error check
    avgDistance /= (omega.size() - B.size());

    B[nonPointBands[nonPointBands.size() - 1u]].extremas =
        omega.size() - (B.size() - nonPointBands.size());
    mpfr::mpreal buffer;
    buffer = bandwidths[nonPointBands[0]] / avgDistance;
    buffer += 0.5;

    if (nonPointBands.size() > 1) {
      B[nonPointBands[0]].extremas =
          mpfr_get_ui(buffer.mpfr_ptr(), GMP_RNDN) + 1;
      B[nonPointBands[nonPointBands.size() - 1u]].extremas -=
          B[nonPointBands[0]].extremas;
    }

    for (std::size_t i{1u}; i < nonPointBands.size() - 1; ++i) {
      buffer = bandwidths[nonPointBands[i]] / avgDistance;
      buffer += 0.5;
      B[nonPointBands[i]].extremas =
          mpfr_get_ui(buffer.mpfr_ptr(), GMP_RNDN) + 1;
      B[nonPointBands[nonPointBands.size() - 1u]].extremas -=
          B[nonPointBands[i]].extremas;
    }

    std::size_t startIndex = 0ul;
    for (std::size_t i{0ul}; i < B.size(); ++i) {
      if (B[i].extremas > 1u)
        buffer = bandwidths[i] / (B[i].extremas - 1);
      omega[startIndex + B[i].extremas - 1] = B[i].stop;
      omega[startIndex] = B[i].start;
      for (std::size_t j{1ul}; j < B[i].extremas - 1; ++j)
        omega[startIndex + j] = omega[startIndex + j - 1] + buffer;
      startIndex += B[i].extremas;
    }
  }

  mpreal::set_default_prec(prevPrec);
}

void referenceScaling(std::vector<mpfr::mpreal> &newX,
                      std::vector<Band> &newChebyBands,
                      std::vector<Band> &newFreqBands, std::size_t newXSize,
                      std::vector<mpfr::mpreal> &x,
                      std::vector<Band> &chebyBands,
                      std::vector<Band> &freqBands, mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  std::vector<std::size_t> newDistribution(chebyBands.size());
  for (std::size_t i{0u}; i < chebyBands.size(); ++i)
    newDistribution[i] = 0u;
  std::size_t multipointBands = 0u;
  std::size_t offset = 0u;
  int twoInt = 0;
  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    newX.push_back(x[offset]);
    //++newDistribution[i];
    if (chebyBands[i].extremas > 2u) {
      ++multipointBands;
      for (std::size_t j = 1u; j < chebyBands[i].extremas - 2u; ++j) {
        newX.push_back((x[offset + j] + x[offset + j + 1]) / 2);
        newX.push_back(x[offset + j]);
        // newDistribution[i] += 2u;
      }
      // if(chebyBands[i].extremas > 3u)
      //{
      //    newDistribution[i] += 1;
      //    newX.push_back(x[offset + chebyBands[i].extremas - 3u]);
      //}
      newX.push_back(x[offset + chebyBands[i].extremas - 2u]);
      newX.push_back(x[offset + chebyBands[i].extremas - 1u]);
      // newDistribution[i] += 2;
      twoInt += 2;
    } else if (chebyBands[i].extremas == 2u) {
      ++multipointBands;
      ++twoInt;
      newX.push_back(x[offset + 1u]);
      ++newDistribution[i];
    }
    offset += chebyBands[i].extremas;
  }
  int threeInt = newXSize - newX.size() - twoInt;
  offset = 0u;
  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    if (chebyBands[i].extremas > 1u) {
      if (threeInt > 0) {
        newX.push_back(x[offset] + (x[offset + 1] - x[offset]) / 3);
        mpfr::mpreal test = x[offset] + (x[offset + 1] - x[offset]) / 3 +
                            (x[offset + 1] - x[offset]) / 3;
        newX.push_back(test);
        // newDistribution[i] += 2u;
        threeInt--;
        twoInt--;
      } else if (twoInt > 0) {
        newX.push_back((x[offset] + x[offset + 1]) / 2);
        // newDistribution[i] += 1u;
        twoInt--;
      }
    }
    offset += chebyBands[i].extremas;
  }
  offset = 0;
  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    if (chebyBands[i].extremas > 2u) {
      if (threeInt > 0) {
        newX.push_back(x[offset + chebyBands[i].extremas - 2u] +
                       (x[offset + chebyBands[i].extremas - 1u] -
                        x[offset + chebyBands[i].extremas - 2u]) /
                           3);
        mpfr::mpreal test = x[offset + chebyBands[i].extremas - 2u] +
                            (x[offset + chebyBands[i].extremas - 1u] -
                             x[offset + chebyBands[i].extremas - 2u]) /
                                3 +
                            (x[offset + chebyBands[i].extremas - 1u] -
                             x[offset + chebyBands[i].extremas - 2u]) /
                                3;
        newX.push_back(test);
        // newDistribution[i] += 2u;
        threeInt--;
        twoInt--;
      } else if (twoInt > 0) {
        newX.push_back((x[offset + chebyBands[i].extremas - 2u] +
                        x[offset + chebyBands[i].extremas - 1u]) /
                       2);
        // newDistribution[i] += 1u;
        twoInt--;
      }
    }
    offset += chebyBands[i].extremas;
  }
  if (newXSize > newX.size()) {
    std::cerr << "Failed to do reference scaling\n";
    exit(EXIT_FAILURE);
  }
  // if (newX.size() > newXSize)
  //{
  //    std::size_t toRemove = newX.size() - newXSize;
  //    for(std::size_t i = newXSize; i < newX.size(); ++i)
  //    {
  //        for(std::size_t j = 0u; j < chebyBands.size(); ++j)
  //            if(newX[i] >= chebyBands[j].start && newX[i] <=
  //            chebyBands[j].stop)
  //                newDistribution[j]--;
  //    }
  //}
  newX.resize(newXSize);
  std::sort(newX.begin(), newX.end());
  std::size_t total = 0u;
  for (std::size_t i = 0ul; i < newX.size(); ++i) {
    for (std::size_t j = 0u; j < chebyBands.size(); ++j)
      if (newX[i] >= chebyBands[j].start && newX[i] <= chebyBands[j].stop) {
        newDistribution[j]++;
        ++total;
      }
  }
  if (total != newXSize) {
    std::cout << "Failed to find distribution!\n";
    exit(EXIT_FAILURE);
  }

  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    newFreqBands[freqBands.size() - 1u - i].extremas = newDistribution[i];
    newChebyBands[i].extremas = newDistribution[i];
  }
  mpreal::set_default_prec(prevPrec);
}

void splitInterval(std::vector<Interval> &subIntervals,
                   std::vector<Band> &chebyBands, std::vector<mpfr::mpreal> &x,
                   mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  // for(std::size_t i = 0u; i < chebyBands.size(); ++i)
  //    std::cout << "Band " << i << " with extremas " << chebyBands[i].extremas
  //    << std::endl;

  std::size_t bandOffset = 0u;
  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    // std::cout << "Band " << i << std::endl;
    if (bandOffset < x.size()) {
      mpfr::mpreal middleValA, middleValB;
      if (x[bandOffset] > chebyBands[i].start &&
          x[bandOffset] < chebyBands[i].stop) {
        middleValA = x[bandOffset];
        subIntervals.push_back(std::make_pair(chebyBands[i].start, middleValA));
      } else {
        middleValA = chebyBands[i].start;
      }
      if (chebyBands[i].extremas > 1) {
        for (std::size_t j = bandOffset;
             j < bandOffset + chebyBands[i].extremas - 2u; ++j) {
          middleValB = x[j + 1];
          subIntervals.push_back(std::make_pair(middleValA, middleValB));
          middleValA = middleValB;
        }
        if (middleValA != chebyBands[i].stop)
          subIntervals.push_back(
              std::make_pair(middleValA, chebyBands[i].stop));
      } // else {
      // subIntervals.push_back(
      // std::make_pair(middleValA, chebyBands[i].stop));
      //}
      bandOffset += chebyBands[i].extremas;
    }
  }
  mpreal::set_default_prec(prevPrec);
}

void findEigenExtrema(mpfr::mpreal &convergenceOrder, mpfr::mpreal &delta,
                      std::vector<mpfr::mpreal> &eigenExtrema,
                      std::vector<mpfr::mpreal> &x,
                      std::vector<Band> &chebyBands, int Nmax, mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  // 1.   Split the initial [-1, 1] interval in subintervals
  //      in order that we can use a reasonable size matrix
  //      eigenvalue solver on the subintervals
  std::vector<Interval> subIntervals;
  mpfr::mpreal a = -1;
  mpfr::mpreal b = 1;

  splitInterval(subIntervals, chebyBands, x, prec);

  // std::cout << "Number of subintervals: "
  //    << subIntervals.size() << std::endl;

  // 2.   Compute the barycentric variables (i.e. weights)
  //      needed for the current iteration

  std::vector<mpfr::mpreal> w(x.size());
  barycentricWeights(w, x, prec);

  computeDelta(delta, w, x, chebyBands, prec);
  //std::cout << "delta = " << delta << std::endl;

  std::vector<mpfr::mpreal> C(x.size());
  computeC(C, delta, x, chebyBands, prec);

  // 3.   Use an eigenvalue solver on each subinterval to find the
  //      local extrema that are located inside the frequency bands
  std::vector<mpfr::mpreal> chebyNodes(Nmax + 1u);
  generateEquidistantNodes(chebyNodes, Nmax, prec);
  applyCos(chebyNodes, chebyNodes);

  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> potentialExtrema;
  std::vector<mpfr::mpreal> pEx;
  mpfr::mpreal extremaErrorValueLeft;
  mpfr::mpreal extremaErrorValueRight;
  mpfr::mpreal extremaErrorValue;
  computeError(extremaErrorValue, chebyBands[0].start, delta, x, C, w,
               chebyBands);
  if(mpfr::abs(extremaErrorValue) >= mpfr::abs(delta))
    potentialExtrema.push_back(
        std::make_pair(chebyBands[0].start, extremaErrorValue));

  for (std::size_t i = 0u; i < chebyBands.size() - 1; ++i) {
    computeError(extremaErrorValueLeft, chebyBands[i].stop, delta, x, C, w,
                 chebyBands, prec);
    computeError(extremaErrorValueRight, chebyBands[i + 1].start, delta, x, C,
                 w, chebyBands, prec);
    int sgnLeft = mpfr::sgn(extremaErrorValueLeft);
    int sgnRight = mpfr::sgn(extremaErrorValueRight);
    if (sgnLeft * sgnRight < 0) {
      if(mpfr::abs(extremaErrorValueLeft) >= mpfr::abs(delta))
      potentialExtrema.push_back(
          std::make_pair(chebyBands[i].stop, extremaErrorValueLeft));
      if(mpfr::abs(extremaErrorValueRight) >= mpfr::abs(delta))
      potentialExtrema.push_back(
          std::make_pair(chebyBands[i + 1].start, extremaErrorValueRight));
    } else {
      mpfr::mpreal abs1 = mpfr::abs(extremaErrorValueLeft);
      mpfr::mpreal abs2 = mpfr::abs(extremaErrorValueRight);
      if (abs1 > abs2) {
        if(mpfr::abs(extremaErrorValueLeft) >= mpfr::abs(delta)) {
          potentialExtrema.push_back(
              std::make_pair(chebyBands[i].stop, extremaErrorValueLeft));
        } else {
          if(mpfr::abs(extremaErrorValueRight) >= mpfr::abs(delta)) {
          potentialExtrema.push_back(
              std::make_pair(chebyBands[i + 1].start, extremaErrorValueRight));
          }
        }
      }
    }
  }
  computeError(extremaErrorValue, chebyBands[chebyBands.size() - 1].stop, delta,
               x, C, w, chebyBands, prec);
  if(mpfr::abs(extremaErrorValue) >= mpfr::abs(delta))

  potentialExtrema.push_back(std::make_pair(
      chebyBands[chebyBands.size() - 1].stop, extremaErrorValue));
  // TODO (optimization): compute the error values at the
  // boundaries of the subintervals in order to optimize
  // the number of error evaluations needed per subinterval
  // (i.e. since the error value at the interval extremities
  // is used on average two times, we can pre-compute
  // them for an average saving of one error evaluation
  // per interval)

  std::vector<std::vector<mpfr::mpreal>> pExs(subIntervals.size());

#pragma omp parallel for
  for (std::size_t i = 0u; i < subIntervals.size(); ++i) {

    // find the Chebyshev nodes scaled to the current subinterval
    std::vector<mpfr::mpreal> siCN(Nmax + 1u);
    changeOfVariable(siCN, chebyNodes, subIntervals[i].first,
                     subIntervals[i].second);

    // compute the Chebyshev interpolation function values on the
    // current subinterval
    std::vector<mpfr::mpreal> fx(Nmax + 1u);
    for (std::size_t j = 0u; j < fx.size(); ++j) {
      computeError(fx[j], siCN[j], delta, x, C, w, chebyBands, prec);
    }

    // compute the values of the CI coefficients and those of its
    // derivative
    std::vector<mpfr::mpreal> chebyCoeffs(Nmax + 1u);
    generateChebyshevCoefficients(chebyCoeffs, fx, Nmax, prec);
    std::vector<mpfr::mpreal> derivCoeffs(Nmax);
    derivativeCoefficients2ndKind(derivCoeffs, chebyCoeffs);

    // solve the corresponding eigenvalue problem and determine the
    // local extrema situated in the current subinterval
    MatrixXq Cm(Nmax - 1u, Nmax - 1u);
    generateColleagueMatrix2ndKind(Cm, derivCoeffs, true, prec);

    std::vector<mpfr::mpreal> eigenRoots;
    VectorXcq roots;
    determineEigenvalues(roots, Cm);
    getRealValues(eigenRoots, roots, a, b);
    changeOfVariable(eigenRoots, eigenRoots, subIntervals[i].first,
                     subIntervals[i].second);
    for (std::size_t j = 0u; j < eigenRoots.size(); ++j)
      pExs[i].push_back(eigenRoots[j]);
    pExs[i].push_back(subIntervals[i].first);
    pExs[i].push_back(subIntervals[i].second);
  }
  for (std::size_t i = 0u; i < pExs.size(); ++i)
    for (std::size_t j = 0u; j < pExs[i].size(); ++j)
      pEx.push_back(pExs[i][j]);

  //std::size_t startingOffset = potentialExtrema.size();
  //potentialExtrema.resize(potentialExtrema.size() + pEx.size());
//#pragma omp parallel for
  for (std::size_t i = 0u; i < pEx.size(); ++i) {
    mpfr::mpreal valBuffer;
    computeError(valBuffer, pEx[i], delta, x, C, w, chebyBands, prec);
    //potentialExtrema[startingOffset + i] = std::make_pair(pEx[i], valBuffer);
    if(mpfr::abs(valBuffer) >= mpfr::abs(delta))
       potentialExtrema.push_back(std::make_pair(pEx[i], valBuffer));
  }

  // sort list of potential extrema in increasing order
  std::sort(potentialExtrema.begin(), potentialExtrema.end(),
            [](const std::pair<mpfr::mpreal, mpfr::mpreal> &lhs,
               const std::pair<mpfr::mpreal, mpfr::mpreal> &rhs) {
              return lhs.first < rhs.first;
            });

  eigenExtrema.clear();
  std::size_t extremaIt = 0u;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> alternatingExtrema;
  mpfr::mpreal minError = INT_MAX;
  mpfr::mpreal maxError = INT_MIN;
  mpfr::mpreal absError;

  while (extremaIt < potentialExtrema.size()) {
    std::pair<mpfr::mpreal, mpfr::mpreal> maxErrorPoint;
    maxErrorPoint = potentialExtrema[extremaIt];
    while (extremaIt < potentialExtrema.size() - 1 &&
           mpfr::sgn(maxErrorPoint.second) *
                   mpfr::sgn(potentialExtrema[extremaIt + 1].second) >
               0) {
      ++extremaIt;
      if (mpfr::abs(maxErrorPoint.second) <
          mpfr::abs(potentialExtrema[extremaIt].second))
        maxErrorPoint = potentialExtrema[extremaIt];
    }
    alternatingExtrema.push_back(maxErrorPoint);
    ++extremaIt;
  }
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bufferExtrema;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> backupExtrema = alternatingExtrema;

  //std::cout << "Alternating extrema: " << x.size() << " | "
  //          << alternatingExtrema.size() << std::endl;

  if (alternatingExtrema.size() < x.size()) {
    std::cerr << "TRIGGER: Not enough alternating extrema!\n"
              << "POSSIBLE CAUSE: Nmax too small\n";
    convergenceOrder = 2.0;
    mpreal::set_default_prec(prevPrec);
    return;
  } else if (alternatingExtrema.size() > x.size()) {
    std::size_t remSuperfluous = alternatingExtrema.size() - x.size();
    if (remSuperfluous % 2 != 0) {
      if (remSuperfluous == 1u) {
        std::vector<mpfr::mpreal> x1, x2;
        x1.push_back(alternatingExtrema[0u].first);
        for (std::size_t i = 1u; i < alternatingExtrema.size() - 1; ++i) {
          x1.push_back(alternatingExtrema[i].first);
          x2.push_back(alternatingExtrema[i].first);
        }
        x2.push_back(alternatingExtrema[alternatingExtrema.size() - 1u].first);
        mpfr::mpreal delta1, delta2;
        computeDelta(delta1, x1, chebyBands, prec);
        computeDelta(delta2, x2, chebyBands, prec);
        delta1 = mpfr::abs(delta1);
        delta2 = mpfr::abs(delta2);
        std::size_t sIndex = 1u;
        if (delta1 > delta2)
          sIndex = 0u;
        for (std::size_t i = sIndex;
             i < alternatingExtrema.size() + sIndex - 1u; ++i)
          bufferExtrema.push_back(alternatingExtrema[i]);
        alternatingExtrema = bufferExtrema;
        bufferExtrema.clear();
      } else {
        mpfr::mpreal abs1 = mpfr::abs(alternatingExtrema[0].second);
        mpfr::mpreal abs2 =
            mpfr::abs(alternatingExtrema[alternatingExtrema.size() - 1].second);
        std::size_t sIndex = 0u;
        if (abs1 < abs2)
          sIndex = 1u;
        for (std::size_t i = sIndex;
             i < alternatingExtrema.size() + sIndex - 1u; ++i)
          bufferExtrema.push_back(alternatingExtrema[i]);
        alternatingExtrema = bufferExtrema;
        bufferExtrema.clear();
      }
    }

    while (alternatingExtrema.size() > x.size()) {
      std::size_t toRemoveIndex = 0u;
      mpfr::mpreal minValToRemove =
          mpfr::min(mpfr::abs(alternatingExtrema[0].second),
                    mpfr::abs(alternatingExtrema[1].second));
      mpfr::mpreal removeBuffer;
      for (std::size_t i = 1u; i < alternatingExtrema.size() - 1; ++i) {
        removeBuffer = mpfr::min(mpfr::abs(alternatingExtrema[i].second),
                                 mpfr::abs(alternatingExtrema[i + 1].second));
        if (removeBuffer < minValToRemove) {
          minValToRemove = removeBuffer;
          toRemoveIndex = i;
        }
      }
      for (std::size_t i = 0u; i < toRemoveIndex; ++i)
        bufferExtrema.push_back(alternatingExtrema[i]);
      for (std::size_t i = toRemoveIndex + 2u; i < alternatingExtrema.size();
           ++i)
        bufferExtrema.push_back(alternatingExtrema[i]);
      alternatingExtrema = bufferExtrema;
      bufferExtrema.clear();
    }
  }
  if (alternatingExtrema.size() < x.size()) {
    std::cerr << "Trouble!\n";
    exit(EXIT_FAILURE);
  }

  //std::cout << "After removal: " << alternatingExtrema.size() << std::endl;
  for (auto &it : alternatingExtrema) {
    eigenExtrema.push_back(it.first);
    absError = mpfr::abs(it.second);
    minError = mpfr::min(minError, absError);
    maxError = mpfr::max(maxError, absError);
  }
  if(alternatingExtrema[0].second * backupExtrema[0].second < 0)
      delta = -delta;

  //std::cout << "Min error = " << minError << std::endl;
  //std::cout << "Max error = " << maxError << std::endl;
  convergenceOrder = (maxError - minError) / maxError;
  //std::cout << "Convergence order = " << convergenceOrder << std::endl;
  // update the extrema count in each frequency band
  std::size_t bIndex = 0u;
  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    chebyBands[i].extremas = 0;
  }
  for (auto &it : eigenExtrema) {
    if (chebyBands[bIndex].start <= it && it <= chebyBands[bIndex].stop) {
      ++chebyBands[bIndex].extremas;
    } else {
      ++bIndex;
      ++chebyBands[bIndex].extremas;
    }
  }

  mpreal::set_default_prec(prevPrec);
}

// TODO: remember that this routine assumes that the information
// pertaining to the reference x and the frequency bands (i.e. the
// number of reference values inside each band) is given at the
// beginning of the execution
PMOutput exchange(std::vector<mpfr::mpreal> &x, std::vector<Band> &chebyBands,
                  mpfr::mpreal eps, int Nmax, mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  PMOutput output;

  std::size_t degree = x.size() - 2u;
  std::sort(x.begin(), x.end(),
            [](const mpfr::mpreal &lhs, const mpfr::mpreal &rhs) {
              return lhs < rhs;
            });
  std::vector<mpfr::mpreal> startX{x};

  output.Q = 1;
  output.iter = 0u;
  do {
    ++output.iter;
    //std::cout << "*********ITERATION " << output.iter << " **********\n";
    findEigenExtrema(output.Q, output.delta, output.x, startX, chebyBands, Nmax,
                     prec);
    startX = output.x;
    if (output.Q > 1.0)
      break;

  } while (output.Q > eps && output.iter <= 100u);

  if (isnan(output.delta) || isnan(output.Q))
    std::cerr << "The exchange algorithm did not converge.\n"
              << "TRIGGER: numerical instability\n"
              << "POSSIBLE CAUSES: poor starting reference and/or "
              << "a too small value for Nmax.\n";

  if (output.iter > 101u)
    std::cerr << "The exchange algorithm did not converge.\n"
              << "TRIGGER: exceeded iteration threshold of 100\n"
              << "POSSIBLE CAUSES: poor starting reference and/or "
              << "a too small value for Nmax.\n";

  for (std::size_t i = 0u; i < chebyBands.size(); ++i)
    //std::cout << "Band " << i << " has " << chebyBands[i].extremas << std::endl;

  output.h.resize(degree + 1u);
  std::vector<mpfr::mpreal> finalC(output.x.size());
  std::vector<mpfr::mpreal> finalAlpha(output.x.size());
  barycentricWeights(finalAlpha, output.x, prec);
  mpfr::mpreal finalDelta = output.delta;
  output.delta = mpfr::abs(output.delta);
  //std::cout << "MINIMAX delta = " << output.delta << std::endl;
  computeC(finalC, finalDelta, output.x, chebyBands, prec);
  std::vector<mpfr::mpreal> finalChebyNodes(degree + 1);
  generateEquidistantNodes(finalChebyNodes, degree, prec);
  applyCos(finalChebyNodes, finalChebyNodes);
  std::vector<mpfr::mpreal> fv(degree + 1);

  for (std::size_t i = 0u; i < fv.size(); ++i)
    computeApprox(fv[i], finalChebyNodes[i], output.x, finalC, finalAlpha,
                  prec);

  generateChebyshevCoefficients(output.h, fv, degree, prec);
  mpreal::set_default_prec(prevPrec);

  return output;
}

// type I&II filters
PMOutput firpm(std::size_t n, std::vector<mpfr::mpreal> const &f,
               std::vector<mpfr::mpreal> const &a,
               std::vector<mpfr::mpreal> const &w, mpfr::mpreal eps, int Nmax,
               mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<mpfr::mpreal> h;
  if (n % 2 != 0) {
    if ((f[f.size() - 1u] == 1) && (a[a.size() - 1u] != 0)) {
      std::cout << "Warning: Gain at Nyquist frequency different from 0.\n"
                << "Increasing the number of taps by 1 and passing to a "
                << " type I filter\n"
                << std::endl;
      ++n;
    } else {
      std::size_t degree = n / 2u;
      // TODO: error checking code
      std::vector<Band> freqBands(w.size());
      std::vector<Band> chebyBands;
      for (std::size_t i = 0u; i < freqBands.size(); ++i) {
        freqBands[i].start = pi * f[2u * i];
        if (i < freqBands.size() - 1u)
          freqBands[i].stop = pi * f[2u * i + 1u];
        else {
          if (f[2u * i + 1u] == 1.0) {
            if (f[2u * i] < 0.9999)
              freqBands[i].stop = pi * 0.9999;
            else
              freqBands[i].stop = pi * ((f[2u * i] + 1) / 2);
          } else
            freqBands[i].stop = pi * f[2u * i + 1u];
        }
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].amplitude = [=](BandSpace bSpace,
                                     mpfr::mpreal x) -> mpfr::mpreal {
          if (a[2u * i] != a[2u * i + 1u]) {
            if (bSpace == BandSpace::CHEBY)
              x = acos(x);
            return (((x - freqBands[i].start) * a[2u * i + 1u] -
                     (x - freqBands[i].stop) * a[2u * i]) /
                    (freqBands[i].stop - freqBands[i].start)) /
                   mpfr::cos(x / 2);
          }
          if (bSpace == BandSpace::FREQ)
            return a[2u * i] / mpfr::cos(x / 2);
          else
            return a[2u * i] / mpfr::sqrt((x + 1) / 2);
        };
        freqBands[i].weight = [=](BandSpace bSpace,
                                  mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::FREQ)
            return mpfr::cos(x / 2) * w[i];
          else
            return mpfr::sqrt((x + 1) / 2) * w[i];
        };
      }
      std::vector<mpfr::mpreal> omega(degree + 2u);
      std::vector<mpfr::mpreal> x(degree + 2u);
      initUniformExtremas(omega, freqBands, prec);
      applyCos(x, omega);
      bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

      PMOutput output = exchange(x, chebyBands, eps, Nmax, prec);
      // std::string folder = ".";
      // std::string filename = "out";
      // plotHelper(folder, filename, output.h);

      h.resize(n + 1u);
      h[0] = h[n] = output.h[degree] / 4;
      h[degree] = h[degree + 1] = (output.h[0] * 2 + output.h[1]) / 4;
      for (std::size_t i{2u}; i < degree + 1; ++i)
        h[degree + 1 - i] = h[degree + i] =
            (output.h[i - 1] + output.h[i]) / 4u;
      output.h = h;
      mpreal::set_default_prec(prevPrec);
      return output;
    }
  }

  std::size_t degree = n / 2u;
  // TODO: error checking code
  std::vector<Band> freqBands(w.size());
  std::vector<Band> chebyBands;
  for (std::size_t i = 0u; i < freqBands.size(); ++i) {
    freqBands[i].start = pi * f[2u * i];
    freqBands[i].stop = pi * f[2u * i + 1u];
    freqBands[i].space = BandSpace::FREQ;
    freqBands[i].amplitude = [=](BandSpace bSpace,
                                 mpfr::mpreal x) -> mpfr::mpreal {
      if (a[2u * i] != a[2u * i + 1u]) {
        if (bSpace == BandSpace::CHEBY)
          x = mpfr::acos(x, MPFR_RNDN);
        return ((x - freqBands[i].start) * a[2u * i + 1u] -
                (x - freqBands[i].stop) * a[2u * i]) /
               (freqBands[i].stop - freqBands[i].start);
      }
      return a[2u * i];
    };
    freqBands[i].weight = [=](BandSpace bSpace,
                              mpfr::mpreal x) -> mpfr::mpreal { return w[i]; };
  }

  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  mpfr::mpreal finalDelta;
  std::vector<mpfr::mpreal> coeffs;
  std::vector<mpfr::mpreal> finalExtrema;
  mpfr::mpreal convergenceOrder;

  PMOutput output = exchange(x, chebyBands, eps, Nmax, prec);

  h.resize(n + 1u);
  h[degree] = output.h[0];
  for (std::size_t i{0u}; i < degree; ++i)
    h[i] = h[n - i] = output.h[degree - i] / 2u;
  output.h = h;
  mpreal::set_default_prec(prevPrec);
  return output;
}

PMOutput firpmRS(std::size_t n, std::vector<mpfr::mpreal> const &f,
                 std::vector<mpfr::mpreal> const &a,
                 std::vector<mpfr::mpreal> const &w, mpfr::mpreal eps,
                 std::size_t depth, int Nmax, mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  if (depth == 0u)
    return firpm(n, f, a, w, eps, Nmax, prec);
  std::vector<mpfr::mpreal> h;
  if (n % 2 != 0) {
    if ((f[f.size() - 1u] == 1) && (a[a.size() - 1u] != 0)) {
      std::cout << "Warning: Gain at Nyquist frequency different from 0.\n"
                << "Increasing the number of taps by 1 and passing to a "
                << " type I filter\n"
                << std::endl;
      ++n;
    } else {
      std::size_t degree = n / 2u;
      // TODO: error checking code
      std::vector<Band> freqBands(w.size());
      std::vector<Band> chebyBands;
      for (std::size_t i = 0u; i < freqBands.size(); ++i) {
        freqBands[i].start = pi * f[2u * i];
        if (i < freqBands.size() - 1u)
          freqBands[i].stop = pi * f[2u * i + 1u];
        else {
          if (f[2u * i + 1u] == 1.0) {
            if (f[2u * i] < 0.9999)
              freqBands[i].stop = pi * 0.9999;
            else
              freqBands[i].stop = pi * ((f[2u * i] + 1) / 2);
          } else
            freqBands[i].stop = pi * f[2u * i + 1u];
        }
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].amplitude = [=](BandSpace bSpace,
                                     mpfr::mpreal x) -> mpfr::mpreal {
          if (a[2u * i] != a[2u * i + 1u]) {
            if (bSpace == BandSpace::CHEBY)
              x = mpfr::acos(x);
            return (((x - freqBands[i].start) * a[2u * i + 1u] -
                     (x - freqBands[i].stop) * a[2u * i]) /
                    (freqBands[i].stop - freqBands[i].start)) /
                   mpfr::cos(x / 2);
          }
          if (bSpace == BandSpace::FREQ)
            return a[2u * i] / mpfr::cos(x / 2, MPFR_RNDN);
          else
            return a[2u * i] / mpfr::sqrt((x + 1) / 2, MPFR_RNDN);
        };
        freqBands[i].weight = [=](BandSpace bSpace,
                                  mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::FREQ)
            return mpfr::cos(x / 2) * w[i];
          else
            return mpfr::sqrt((x + 1) / 2) * w[i];
        };
      }

      std::vector<std::size_t> scaledDegrees(depth + 1u);
      scaledDegrees[depth] = degree;
      for (int i = depth - 1u; i >= 0; --i) {
        scaledDegrees[i] = scaledDegrees[i + 1] / 2;
      }

      std::vector<mpfr::mpreal> omega(scaledDegrees[0] + 2u);
      std::vector<mpfr::mpreal> x(scaledDegrees[0] + 2u);
      initUniformExtremas(omega, freqBands, prec);
      applyCos(x, omega);
      bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
      PMOutput output = exchange(x, chebyBands, eps, Nmax, prec);

      for (std::size_t i = 1u; i <= depth; ++i) {
        x.clear();
        referenceScaling(x, chebyBands, freqBands, scaledDegrees[i] + 2u,
                         output.x, chebyBands, freqBands, prec);
        output = exchange(x, chebyBands, eps, Nmax, prec);
      }

      // std::string folder = ".";
      // std::string filename = "out";
      // plotHelper(folder, filename, output.h);

      h.resize(n + 1u);
      h[0] = h[n] = output.h[degree] / 4;
      h[degree] = h[degree + 1] = (output.h[0] * 2 + output.h[1]) / 4;
      for (std::size_t i{2u}; i < degree + 1; ++i)
        h[degree + 1 - i] = h[degree + i] =
            (output.h[i - 1] + output.h[i]) / 4u;
      output.h = h;
      mpreal::set_default_prec(prevPrec);
      return output;
    }
  }

  std::size_t degree = n / 2u;
  // TODO: error checking code
  std::vector<Band> freqBands(w.size());
  std::vector<Band> chebyBands;
  for (std::size_t i = 0u; i < freqBands.size(); ++i) {
    freqBands[i].start = pi * f[2u * i];
    freqBands[i].stop = pi * f[2u * i + 1u];
    freqBands[i].space = BandSpace::FREQ;
    freqBands[i].amplitude = [=](BandSpace bSpace,
                                 mpfr::mpreal x) -> mpfr::mpreal {
      if (a[2u * i] != a[2u * i + 1u]) {
        if (bSpace == BandSpace::CHEBY)
          x = mpfr::acos(x);
        return ((x - freqBands[i].start) * a[2u * i + 1u] -
                (x - freqBands[i].stop) * a[2u * i]) /
               (freqBands[i].stop - freqBands[i].start);
      }
      return a[2u * i];
    };
    freqBands[i].weight = [=](BandSpace bSpace,
                              mpfr::mpreal x) -> mpfr::mpreal { return w[i]; };
  }

  std::vector<std::size_t> scaledDegrees(depth + 1u);
  scaledDegrees[depth] = degree;
  for (int i = depth - 1u; i >= 0; --i) {
    scaledDegrees[i] = scaledDegrees[i + 1] / 2;
  }

  std::vector<mpfr::mpreal> omega(scaledDegrees[0] + 2u);
  std::vector<mpfr::mpreal> x(scaledDegrees[0] + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
  PMOutput output = exchange(x, chebyBands, eps, Nmax, prec);

  for (std::size_t i = 1u; i <= depth; ++i) {
    x.clear();
    referenceScaling(x, chebyBands, freqBands, scaledDegrees[i] + 2u, output.x,
                     chebyBands, freqBands, prec);
    output = exchange(x, chebyBands, eps, Nmax, prec);
  }

  h.resize(n + 1u);
  h[degree] = output.h[0];
  for (std::size_t i{0u}; i < degree; ++i)
    h[i] = h[n - i] = output.h[degree - i] / 2u;
  output.h = h;
  mpreal::set_default_prec(prevPrec);
  return output;
}

// type III & IV filters
PMOutput firpm(std::size_t n, std::vector<mpfr::mpreal> const &f,
               std::vector<mpfr::mpreal> const &a,
               std::vector<mpfr::mpreal> const &w, ftype type, mpfr::mpreal eps,
               int Nmax, mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  PMOutput output;
  std::vector<mpfr::mpreal> h;
  switch (type) {
  case ftype::FIR_DIFFERENTIATOR: {
    std::size_t degree = n / 2u;
    // TODO: error checking code
    std::vector<mpfr::mpreal> fn = f;

    std::vector<Band> freqBands(w.size());
    std::vector<Band> chebyBands;
    mpfr::mpreal scaleFactor = a[1] / (f[1] * pi);
    if (n % 2 == 0) // Type III
    {
      if (f[0u] == 0.0l) {
        if (fn[1u] < 0.001l)
          fn[0u] = fn[1] / 2;
        else
          fn[0u] = 0.001l;
      }
      if (f[f.size() - 1u] == 1.0l) {
        if (f[f.size() - 2u] > 0.999l)
          fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
        else
          fn[f.size() - 1u] = 0.999l;
      }
      --degree;
      freqBands[0].start = pi * fn[0u];
      freqBands[0].stop = pi * fn[1u];
      freqBands[0].space = BandSpace::FREQ;
      freqBands[0].weight = [w](BandSpace bSpace,
                                mpfr::mpreal x) -> mpfr::mpreal {
        if (bSpace == BandSpace::FREQ) {
          return (mpfr::sin(x) / x) * w[0u];
        } else {
          return (mpfr::sqrt(mpfr::mpreal(1.0l) - x * x) / mpfr::acos(x)) *
                 w[0u];
        }
      };
      freqBands[0].amplitude = [scaleFactor](BandSpace bSpace,
                                             mpfr::mpreal x) -> mpfr::mpreal {
        if (bSpace == BandSpace::FREQ) {
          return (x / mpfr::sin(x)) * scaleFactor;
        } else {
          return (mpfr::acos(x) / mpfr::sqrt(mpfr::mpreal(1.0l) - x * x)) *
                 scaleFactor;
        }
      };
      for (std::size_t i = 1u; i < freqBands.size(); ++i) {
        freqBands[i].start = pi * fn[2u * i];
        freqBands[i].stop = pi * fn[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].weight = [w, i](BandSpace bSpace,
                                     mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::FREQ) {
            return mpfr::sin(x) * w[i];
          } else {
            return mpfr::sqrt(mpfr::mpreal(1.0l) - x * x) * w[i];
          }

        };
        freqBands[i].amplitude = [freqBands, a, i](
            BandSpace bSpace, mpfr::mpreal x) -> mpfr::mpreal {
          if (a[2u * i] != a[2u * i + 1u]) {
            if (bSpace == BandSpace::CHEBY)
              x = mpfr::acos(x);
            return ((x - freqBands[i].start) * a[2u * i + 1u] -
                    (x - freqBands[i].stop) * a[2u * i]) /
                   (freqBands[i].stop - freqBands[i].start);
          }
          return a[2u * i];

        };
      }

    } else { // Type IV
      if (f[0u] == 0.0l) {
        if (fn[1u] < 0.00001l)
          fn[0u] = fn[1] / 2;
        else
          fn[0u] = 0.00001l;
      }

      freqBands[0].start = pi * fn[0u];
      freqBands[0].stop = pi * fn[1u];
      freqBands[0].space = BandSpace::FREQ;
      freqBands[0].weight = [w](BandSpace bSpace,
                                mpfr::mpreal x) -> mpfr::mpreal {
        if (bSpace == BandSpace::FREQ) {
          return (mpfr::sin(x / 2) / x) * w[0u];
        } else {
          return (mpfr::sin(mpfr::acos(x) / 2) / mpfr::acos(x)) * w[0u];
        }
      };
      freqBands[0].amplitude = [scaleFactor](BandSpace bSpace,
                                             mpfr::mpreal x) -> mpfr::mpreal {
        if (bSpace == BandSpace::FREQ) {
          return (x / mpfr::sin(x / 2)) * scaleFactor;
        } else {
          return (mpfr::acos(x) / mpfr::sin(mpfr::acos(x) / 2)) * scaleFactor;
        }
      };
      for (std::size_t i = 1u; i < freqBands.size(); ++i) {
        freqBands[i].start = pi * fn[2u * i];
        freqBands[i].stop = pi * fn[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].weight = [w, i](BandSpace bSpace,
                                     mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::FREQ) {
            return mpfr::sin(x / 2) * w[i];
          } else {
            return (mpfr::sin(mpfr::acos(x) / 2)) * w[i];
          }

        };
        freqBands[i].amplitude = [freqBands, a, i](
            BandSpace bSpace, mpfr::mpreal x) -> mpfr::mpreal {
          if (a[2u * i] != a[2u * i + 1u]) {
            if (bSpace == BandSpace::CHEBY)
              x = mpfr::acos(x);
            return ((x - freqBands[i].start) * a[2u * i + 1u] -
                    (x - freqBands[i].stop) * a[2u * i]) /
                   (freqBands[i].stop - freqBands[i].start);
          }
          return a[2u * i];

        };
      }
    }

    std::vector<mpfr::mpreal> omega(degree + 2u);
    std::vector<mpfr::mpreal> x(degree + 2u);
    initUniformExtremas(omega, freqBands, prec);
    applyCos(x, omega);
    bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

    output = exchange(x, chebyBands, eps, Nmax, prec);

    h.resize(n + 1u);
    if (n % 2 == 0) {
      h[degree + 1u] = 0;
      h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
      h[degree + 2u] = -h[degree];
      h[1u] = output.h[degree - 1u] / 4;
      h[2u * degree + 1u] = -h[1u];
      h[0u] = output.h[degree] / 4;
      h[2u * (degree + 1u)] = -h[0u];
      for (std::size_t i{2u}; i < degree; ++i) {
        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
        h[degree + 1u + i] = -h[degree + 1u - i];
      }
    } else {
      ++degree;
      h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
      h[degree] = -h[degree - 1u];
      h[0u] = output.h[degree - 1u] / 4;
      h[2u * degree - 1u] = -h[0u];
      for (std::size_t i{2u}; i < degree; ++i) {
        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
        h[degree + i - 1u] = -h[degree - i];
      }
    }

  } break;
  default: // FIR_HILBERT
  {
    std::size_t degree = n / 2u;
    std::vector<mpfr::mpreal> fn = f;
    // TODO: error checking code
    std::vector<Band> freqBands(w.size());
    std::vector<Band> chebyBands;
    if (n % 2 == 0) // Type III
    {
      --degree;
      if (f[0u] == 0.0l) {
        if (fn[1u] < 0.00001l)
          fn[0u] = fn[1] / 2;
        else
          fn[0u] = 0.00001l;
      }
      if (f[f.size() - 1u] == 1.0l) {
        if (f[f.size() - 2u] > 0.9999l)
          fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
        else
          fn[f.size() - 1u] = 0.9999l;
      }

      for (std::size_t i = 0u; i < freqBands.size(); ++i) {
        freqBands[i].start = pi * fn[2u * i];
        freqBands[i].stop = pi * fn[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].amplitude = [=](BandSpace bSpace,
                                     mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::CHEBY)
            x = mpfr::acos(x);

          if (a[2u * i] != a[2u * i + 1u]) {
            return (((x - freqBands[i].start) * a[2u * i + 1u] -
                     (x - freqBands[i].stop) * a[2u * i]) /
                    (freqBands[i].stop - freqBands[i].start)) /
                   mpfr::sin(x);
          }
          return a[2u * i] / mpfr::sin(x);
        };
        freqBands[i].weight = [=](BandSpace bSpace,
                                  mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::FREQ)
            return w[i] * mpfr::sin(x);
          else
            return w[i] * (mpfr::sqrt(mpfr::mpreal(1.0l) - x * x));
        };
      }
    } else { // Type IV
      if (f[0u] == 0.0l) {
        if (f[1u] < 0.00001l)
          fn[0u] = fn[1u] / 2;
        else
          fn[0u] = 0.00001l;
      }
      for (std::size_t i = 0u; i < freqBands.size(); ++i) {
        freqBands[i].start = pi * fn[2u * i];
        freqBands[i].stop = pi * fn[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].amplitude = [=](BandSpace bSpace,
                                     mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::CHEBY)
            x = mpfr::acos(x);

          if (a[2u * i] != a[2u * i + 1u]) {
            return (((x - freqBands[i].start) * a[2u * i + 1u] -
                     (x - freqBands[i].stop) * a[2u * i]) /
                    (freqBands[i].stop - freqBands[i].start)) /
                   mpfr::sin(x / 2);
          }
          return a[2u * i] / mpfr::sin(x / 2);
        };
        freqBands[i].weight = [=](BandSpace bSpace,
                                  mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::FREQ)
            return w[i] * mpfr::sin(x / 2);
          else {
            x = mpfr::acos(x);
            return w[i] * (mpfr::sin(x / 2));
          }
        };
      }
    }
    std::vector<mpfr::mpreal> omega(degree + 2u);
    std::vector<mpfr::mpreal> x(degree + 2u);
    initUniformExtremas(omega, freqBands, prec);
    applyCos(x, omega);
    bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

    output = exchange(x, chebyBands, eps, Nmax, prec);

    h.resize(n + 1u);
    if (n % 2 == 0) {
      h[degree + 1u] = 0;
      h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
      h[degree + 2u] = -h[degree];
      h[1u] = output.h[degree - 1u] / 4;
      h[2u * degree + 1u] = -h[1u];
      h[0u] = output.h[degree] / 4;
      h[2u * (degree + 1u)] = -h[0u];
      for (std::size_t i{2u}; i < degree; ++i) {
        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
        h[degree + 1u + i] = -h[degree + 1u - i];
      }
    } else {
      ++degree;
      h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
      h[degree] = -h[degree - 1u];
      h[0u] = output.h[degree - 1u] / 4;
      h[2u * degree - 1u] = -h[0u];
      for (std::size_t i{2u}; i < degree; ++i) {
        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
        h[degree + i - 1u] = -h[degree - i];
      }
    }

  } break;
  }
  output.h = h;
  mpreal::set_default_prec(prevPrec);
  return output;
}

PMOutput firpmRS(std::size_t n, std::vector<mpfr::mpreal> const &f,
                 std::vector<mpfr::mpreal> const &a,
                 std::vector<mpfr::mpreal> const &w, ftype type,
                 mpfr::mpreal eps, std::size_t depth, int Nmax,
                 mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  if (depth == 0u)
    return firpm(n, f, a, w, type, eps, Nmax, prec);
  PMOutput output;
  std::vector<mpfr::mpreal> h;
  switch (type) {
  case ftype::FIR_DIFFERENTIATOR: {
    std::size_t degree = n / 2u;
    // TODO: error checking code
    std::vector<mpfr::mpreal> fn = f;

    std::vector<Band> freqBands(w.size());
    std::vector<Band> chebyBands;
    mpfr::mpreal scaleFactor = a[1] / (f[1] * pi);
    if (n % 2 == 0) // Type III
    {
      if (f[0u] == 0.0l) {
        if (fn[1u] < 0.001l)
          fn[0u] = fn[1] / 2;
        else
          fn[0u] = 0.001l;
      }
      if (f[f.size() - 1u] == 1.0l) {
        if (f[f.size() - 2u] > 0.999l)
          fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
        else
          fn[f.size() - 1u] = 0.999l;
      }
      --degree;
      freqBands[0].start = pi * fn[0u];
      freqBands[0].stop = pi * fn[1u];
      freqBands[0].space = BandSpace::FREQ;
      freqBands[0].weight = [w](BandSpace bSpace,
                                mpfr::mpreal x) -> mpfr::mpreal {
        if (bSpace == BandSpace::FREQ) {
          return (mpfr::sin(x) / x) * w[0u];
        } else {
          return (mpfr::sqrt(mpfr::mpreal(1.0) - x * x) / mpfr::acos(x)) *
                 w[0u];
        }
      };
      freqBands[0].amplitude = [scaleFactor](BandSpace bSpace,
                                             mpfr::mpreal x) -> mpfr::mpreal {
        if (bSpace == BandSpace::FREQ) {
          return (x / mpfr::sin(x)) * scaleFactor;
        } else {
          return (mpfr::acos(x) / mpfr::sqrt(mpfr::mpreal(1.0l) - x * x)) *
                 scaleFactor;
        }
      };
      for (std::size_t i = 1u; i < freqBands.size(); ++i) {
        freqBands[i].start = pi * fn[2u * i];
        freqBands[i].stop = pi * fn[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].weight = [w, i](BandSpace bSpace,
                                     mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::FREQ) {
            return mpfr::sin(x) * w[i];
          } else {
            return mpfr::sqrt(mpfr::mpreal(1.0l) - x * x) * w[i];
          }

        };
        freqBands[i].amplitude = [freqBands, a, i](
            BandSpace bSpace, mpfr::mpreal x) -> mpfr::mpreal {
          if (a[2u * i] != a[2u * i + 1u]) {
            if (bSpace == BandSpace::CHEBY)
              x = mpfr::acos(x);
            return ((x - freqBands[i].start) * a[2u * i + 1u] -
                    (x - freqBands[i].stop) * a[2u * i]) /
                   (freqBands[i].stop - freqBands[i].start);
          }
          return a[2u * i];

        };
      }

    } else { // Type IV
      if (f[0u] == 0.0l) {
        if (fn[1u] < 0.001l)
          fn[0u] = fn[1] / 2;
        else
          fn[0u] = 0.001l;
      }

      freqBands[0].start = pi * fn[0u];
      freqBands[0].stop = pi * fn[1u];
      freqBands[0].space = BandSpace::FREQ;
      freqBands[0].weight = [w](BandSpace bSpace,
                                mpfr::mpreal x) -> mpfr::mpreal {
        if (bSpace == BandSpace::FREQ) {
          return (mpfr::sin(x / 2) / x) * w[0u];
        } else {
          return (mpfr::sin(mpfr::acos(x) / 2) / mpfr::acos(x)) * w[0u];
        }
      };
      freqBands[0].amplitude = [scaleFactor](BandSpace bSpace,
                                             mpfr::mpreal x) -> mpfr::mpreal {
        if (bSpace == BandSpace::FREQ) {
          return (x / mpfr::sin(x / 2)) * scaleFactor;
        } else {
          return (mpfr::acos(x) / mpfr::sin(mpfr::acos(x) / 2)) * scaleFactor;
        }
      };
      for (std::size_t i = 1u; i < freqBands.size(); ++i) {
        freqBands[i].start = pi * fn[2u * i];
        freqBands[i].stop = pi * fn[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].weight = [w, i](BandSpace bSpace,
                                     mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::FREQ) {
            return mpfr::sin(x / 2) * w[i];
          } else {
            return (mpfr::sin(mpfr::acos(x) / 2)) * w[i];
          }

        };
        freqBands[i].amplitude = [freqBands, a, i](
            BandSpace bSpace, mpfr::mpreal x) -> mpfr::mpreal {
          if (a[2u * i] != a[2u * i + 1u]) {
            if (bSpace == BandSpace::CHEBY)
              x = mpfr::acos(x);
            return ((x - freqBands[i].start) * a[2u * i + 1u] -
                    (x - freqBands[i].stop) * a[2u * i]) /
                   (freqBands[i].stop - freqBands[i].start);
          }
          return a[2u * i];

        };
      }
    }

    std::vector<std::size_t> scaledDegrees(depth + 1u);
    scaledDegrees[depth] = degree;
    for (int i = depth - 1u; i >= 0; --i) {
      scaledDegrees[i] = scaledDegrees[i + 1] / 2;
    }

    std::vector<mpfr::mpreal> omega(scaledDegrees[0] + 2u);
    std::vector<mpfr::mpreal> x(scaledDegrees[0] + 2u);
    initUniformExtremas(omega, freqBands, prec);
    applyCos(x, omega);
    bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
    output = exchange(x, chebyBands, eps, Nmax, prec);

    for (std::size_t i = 1u; i <= depth; ++i) {
      x.clear();
      referenceScaling(x, chebyBands, freqBands, scaledDegrees[i] + 2u,
                       output.x, chebyBands, freqBands, prec);
      output = exchange(x, chebyBands, eps, Nmax, prec);
    }

    h.resize(n + 1u);
    if (n % 2 == 0) {
      h[degree + 1u] = 0;
      h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
      h[degree + 2u] = -h[degree];
      h[1u] = output.h[degree - 1u] / 4;
      h[2u * degree + 1u] = -h[1u];
      h[0u] = output.h[degree] / 4;
      h[2u * (degree + 1u)] = -h[0u];
      for (std::size_t i{2u}; i < degree; ++i) {
        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
        h[degree + 1u + i] = -h[degree + 1u - i];
      }
    } else {
      ++degree;
      h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
      h[degree] = -h[degree - 1u];
      h[0u] = output.h[degree - 1u] / 4;
      h[2u * degree - 1u] = -h[0u];
      for (std::size_t i{2u}; i < degree; ++i) {
        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
        h[degree + i - 1u] = -h[degree - i];
      }
    }

  } break;
  default: // FIR_HILBERT
  {
    std::size_t degree = n / 2u;
    std::vector<mpfr::mpreal> fn = f;
    // TODO: error checking code
    std::vector<Band> freqBands(w.size());
    std::vector<Band> chebyBands;
    if (n % 2 == 0) // Type III
    {
      --degree;
      if (f[0u] == 0.0l) {
        if (fn[1u] < 0.001l)
          fn[0u] = fn[1] / 2;
        else
          fn[0u] = 0.001l;
      }
      if (f[f.size() - 1u] == 1.0l) {
        if (f[f.size() - 2u] > 0.999l)
          fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
        else
          fn[f.size() - 1u] = 0.999l;
      }

      for (std::size_t i = 0u; i < freqBands.size(); ++i) {
        freqBands[i].start = pi * fn[2u * i];
        freqBands[i].stop = pi * fn[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].amplitude = [=](BandSpace bSpace,
                                     mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::CHEBY)
            x = mpfr::acos(x);

          if (a[2u * i] != a[2u * i + 1u]) {
            return (((x - freqBands[i].start) * a[2u * i + 1u] -
                     (x - freqBands[i].stop) * a[2u * i]) /
                    (freqBands[i].stop - freqBands[i].start)) /
                   mpfr::sin(x);
          }
          return a[2u * i] / mpfr::sin(x);
        };
        freqBands[i].weight = [=](BandSpace bSpace,
                                  mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::FREQ)
            return w[i] * mpfr::sin(x);
          else
            return w[i] * (mpfr::sqrt(mpfr::mpreal(1.0) - x * x));
        };
      }
    } else { // Type IV
      if (f[0u] == 0.0l) {
        if (f[1u] < 0.001l)
          fn[0u] = fn[1u] / 2;
        else
          fn[0u] = 0.001l;
      }
      for (std::size_t i = 0u; i < freqBands.size(); ++i) {
        freqBands[i].start = pi * fn[2u * i];
        freqBands[i].stop = pi * fn[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].amplitude = [=](BandSpace bSpace,
                                     mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::CHEBY)
            x = mpfr::acos(x);

          if (a[2u * i] != a[2u * i + 1u]) {
            return (((x - freqBands[i].start) * a[2u * i + 1u] -
                     (x - freqBands[i].stop) * a[2u * i]) /
                    (freqBands[i].stop - freqBands[i].start)) /
                   mpfr::sin(x / 2);
          }
          return a[2u * i] / mpfr::sin(x / 2);
        };
        freqBands[i].weight = [=](BandSpace bSpace,
                                  mpfr::mpreal x) -> mpfr::mpreal {
          if (bSpace == BandSpace::FREQ)
            return w[i] * mpfr::sin(x / 2);
          else {
            x = mpfr::acos(x);
            return w[i] * mpfr::sin(x / 2);
          }
        };
      }
    }

    std::vector<std::size_t> scaledDegrees(depth + 1u);
    scaledDegrees[depth] = degree;
    for (int i = depth - 1u; i >= 0; --i) {
      scaledDegrees[i] = scaledDegrees[i + 1] / 2;
    }

    std::vector<mpfr::mpreal> omega(scaledDegrees[0] + 2u);
    std::vector<mpfr::mpreal> x(scaledDegrees[0] + 2u);
    initUniformExtremas(omega, freqBands, prec);
    applyCos(x, omega);
    bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
    output = exchange(x, chebyBands, eps, Nmax, prec);

    for (std::size_t i = 1u; i <= depth; ++i) {
      x.clear();
      referenceScaling(x, chebyBands, freqBands, scaledDegrees[i] + 2u,
                       output.x, chebyBands, freqBands, prec);
      output = exchange(x, chebyBands, eps, Nmax, prec);
    }

    h.resize(n + 1u);
    if (n % 2 == 0) {
      h[degree + 1u] = 0;
      h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
      h[degree + 2u] = -h[degree];
      h[1u] = output.h[degree - 1u] / 4;
      h[2u * degree + 1u] = -h[1u];
      h[0u] = output.h[degree] / 4;
      h[2u * (degree + 1u)] = -h[0u];
      for (std::size_t i{2u}; i < degree; ++i) {
        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
        h[degree + 1u + i] = -h[degree + 1u - i];
      }
    } else {
      ++degree;
      h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
      h[degree] = -h[degree - 1u];
      h[0u] = output.h[degree - 1u] / 4;
      h[2u * degree - 1u] = -h[0u];
      for (std::size_t i{2u}; i < degree; ++i) {
        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
        h[degree + i - 1u] = -h[degree - i];
      }
    }

  } break;
  }
  output.h = h;
  mpreal::set_default_prec(prevPrec);
  return output;
}


typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;
typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> VectorXcd;

void generateVandermondeMatrix(MatrixXd& A, std::size_t degree, std::vector<double>& meshPoints,
        std::function<double(double)>& weightFunction)
{

    A.resize(degree + 1u, meshPoints.size());
    for(std::size_t i = 0u; i < meshPoints.size(); ++i)
    {
        double pointWeight = weightFunction(meshPoints[i]);
        A(0u, i) = 1;
        A(1u, i) = meshPoints[i];
        for(std::size_t j = 2u; j <= degree; ++j)
            A(j, i) = meshPoints[i] * A(j - 1u, i) * 2 - A(j - 2u, i);
        for(std::size_t j = 0u; j <= degree; ++j)
            A(j, i) *= pointWeight;
    }
}

// approximate Fekete points
void AFPPM(std::vector<double>& points, MatrixXd& A, std::vector<double>& meshPoints)
{
    VectorXd b = VectorXd::Ones(A.rows());
    b(0) = 2;
    VectorXd y = A.colPivHouseholderQr().solve(b);


    for(std::size_t i = 0u; i < y.rows(); ++i)
        if(y(i) != 0.0)
            points.push_back(meshPoints[i]);
    std::sort(points.begin(), points.end(),
            [](const double& lhs,
               const double& rhs) {
                return lhs < rhs;
            });

}

void bandCountPM(std::vector<BandD>& chebyBands, std::vector<double>& x)
{
    for(auto& it : chebyBands)
        it.extremas = 0u;
    std::size_t bandIt = 0u;
    for(std::size_t i = 0u; i < x.size(); ++i)
    {
        while(bandIt < chebyBands.size() && chebyBands[bandIt].stop < x[i])
            bandIt++;
        ++chebyBands[bandIt].extremas;
    }
}



void generateWAM(std::vector<double>& wam, std::vector<BandD>& chebyBands, std::size_t degree)
{
    std::vector<double> chebyNodes(degree + 2u);
    generateEquidistantNodes(chebyNodes, degree + 1u);
    applyCos(chebyNodes, chebyNodes);
    for(std::size_t i = 0u; i < chebyBands.size(); ++i)
    {
        if(chebyBands[i].start != chebyBands[i].stop)
        {
            std::vector<double> bufferNodes(degree + 2u);
            changeOfVariable(bufferNodes, chebyNodes,
                    chebyBands[i].start, chebyBands[i].stop);
            for(auto& it : bufferNodes)
                wam.push_back(it);
        }
        else
            wam.push_back(chebyBands[i].start);
    }
}


void initUniformExtremas(std::vector<double>& omega,
        std::vector<BandD>& B)
{
    double avgDistance = 0;

    std::vector<double> bandwidths(B.size());
    std::vector<std::size_t> nonPointBands;
    for(std::size_t i = 0; i < B.size(); ++i) {
        bandwidths[i] = B[i].stop - B[i].start;
        if(bandwidths[i] > 0.0)
        {
            nonPointBands.push_back(i);
            avgDistance += bandwidths[i];
        }
        B[i].extremas = 1u;
    }
    if(nonPointBands.empty())
    {
        std::cerr << "All intervals are points!\n";
        exit(EXIT_FAILURE);

    }
    // TODO: error check
    avgDistance /= (omega.size() - B.size());

    B[nonPointBands[nonPointBands.size() - 1u]].extremas = omega.size() - (B.size() - nonPointBands.size());
    double buffer;
    buffer = bandwidths[nonPointBands[0]] / avgDistance;
    buffer += 0.5;

        if (nonPointBands.size() > 1) {
            B[nonPointBands[0]].extremas = lrint(buffer) + 1;
            B[nonPointBands[nonPointBands.size() - 1u]].extremas -= B[nonPointBands[0]].extremas;
        }

        for(std::size_t i{1u}; i < nonPointBands.size() - 1; ++i) {
            buffer = bandwidths[nonPointBands[i]] / avgDistance;
            buffer += 0.5;
            B[nonPointBands[i]].extremas = lrint(buffer) + 1;
            B[nonPointBands[nonPointBands.size() - 1u]].extremas -= B[nonPointBands[i]].extremas;
        }


        std::size_t startIndex = 0ul;
        for(std::size_t i{0ul}; i < B.size(); ++i) {
            if(B[i].extremas > 1u)
                buffer = bandwidths[i] / (B[i].extremas - 1);
            omega[startIndex] = B[i].start;
            omega[startIndex + B[i].extremas - 1] = B[i].stop;
            for(std::size_t j{1ul}; j < B[i].extremas - 1; ++j)
                omega[startIndex + j] = omega[startIndex + j - 1] + buffer;
            startIndex += B[i].extremas;
        }
}

void referenceScaling(std::vector<double>& newX, std::vector<BandD>& newChebyBands,
        std::vector<BandD>& newFreqBands, std::size_t newXSize,
        std::vector<double>& x, std::vector<BandD>& chebyBands,
        std::vector<BandD>& freqBands)
{
        std::vector<std::size_t> newDistribution(chebyBands.size());
        for(std::size_t i{0u}; i < chebyBands.size(); ++i)
            newDistribution[i] = 0u;
        std::size_t multipointBands = 0u;
        std::size_t offset = 0u;
        int twoInt = 0;
        for(std::size_t i = 0u; i < chebyBands.size(); ++i)
        {
            newX.push_back(x[offset]);
            if(chebyBands[i].extremas > 2u)
            {
                ++multipointBands;
                for(std::size_t j = 1u; j < chebyBands[i].extremas - 2u; ++j)
                {
                    newX.push_back((x[offset + j] + x[offset + j + 1]) / 2);
                    newX.push_back(x[offset + j]);
                }
                newX.push_back(x[offset + chebyBands[i].extremas - 2u]);
                newX.push_back(x[offset + chebyBands[i].extremas - 1u]);
                twoInt += 2;
            }
            else if(chebyBands[i].extremas == 2u)
            {
                ++multipointBands;
                ++twoInt;
                newX.push_back(x[offset + 1u]);
                ++newDistribution[i];
            }
            offset += chebyBands[i].extremas;
        }
        int threeInt = newXSize - newX.size() - twoInt;
        offset = 0u;
        for(std::size_t i = 0u; i < chebyBands.size(); ++i)
        {
                if(chebyBands[i].extremas > 1u)
                {
                    if(threeInt > 0)
                    {
                        newX.push_back(x[offset] + (x[offset + 1] - x[offset]) / 3);
                        double secondValue = x[offset] + (x[offset + 1] - x[offset]) / 3
                            + (x[offset + 1] - x[offset]) / 3;
                        newX.push_back(secondValue);
                        threeInt--;
                        twoInt--;
                    }
                    else if (twoInt > 0)
                    {
                        newX.push_back((x[offset] + x[offset + 1]) / 2);
                        twoInt--;
                    }
                }
            offset += chebyBands[i].extremas;
        }
        offset = 0;
        for(std::size_t i = 0u; i < chebyBands.size(); ++i)
        {
                if(chebyBands[i].extremas > 2u)
                {
                    if(threeInt > 0)
                    {
                        newX.push_back(x[offset + chebyBands[i].extremas - 2u] +
                                (x[offset + chebyBands[i].extremas - 1u] -
                                 x[offset + chebyBands[i].extremas - 2u]) / 3);
                        double secondValue = x[offset + chebyBands[i].extremas - 2u] +
                            (x[offset + chebyBands[i].extremas - 1u] -
                             x[offset + chebyBands[i].extremas - 2u]) / 3 +
                            (x[offset + chebyBands[i].extremas - 1u] -
                             x[offset + chebyBands[i].extremas - 2u]) / 3;
                        newX.push_back(secondValue);
                        threeInt--;
                        twoInt--;
                    }
                    else if (twoInt > 0)
                    {
                        newX.push_back((x[offset + chebyBands[i].extremas - 2u] +
                                    x[offset + chebyBands[i].extremas - 1u]) / 2);
                        twoInt--;
                    }
                }
            offset += chebyBands[i].extremas;
        }
        if(newXSize > newX.size())
        {
            std::cerr << "Failed to do reference scaling\n";
            exit(EXIT_FAILURE);
        }
        newX.resize(newXSize);
        std::sort(newX.begin(), newX.end());
        std::size_t total = 0u;
        for(std::size_t i = 0ul; i < newX.size(); ++i)
        {
                for(std::size_t j = 0u; j < chebyBands.size(); ++j)
                    if(newX[i] >= chebyBands[j].start && newX[i] <= chebyBands[j].stop)
                    {
                        newDistribution[j]++;
                        ++total;
                    }
        }
        if(total != newXSize)
        {
            std::cout << "Failed to find distribution!\n";
            exit(EXIT_FAILURE);
        }


        for (std::size_t i = 0u; i < chebyBands.size(); ++i)
        {
            newFreqBands[freqBands.size() - 1u - i].extremas = newDistribution[i];
            newChebyBands[i].extremas = newDistribution[i];
        }
}




void splitInterval(std::vector<IntervalD>& subIntervals,
        std::vector<BandD>& chebyBands,
        std::vector<double> &x)
{
    std::size_t bandOffset = 0u;
    for(std::size_t i = 0u; i < chebyBands.size(); ++i)
    {
        if(bandOffset < x.size())
        {
            double middleValA, middleValB;
            if (x[bandOffset] > chebyBands[i].start
                && x[bandOffset] < chebyBands[i].stop)
            {
                middleValA = x[bandOffset];
                subIntervals.push_back(
                        std::make_pair(chebyBands[i].start, middleValA));
            } else {
                middleValA = chebyBands[i].start;
            }
            if(chebyBands[i].extremas > 1)
            {
                for(std::size_t j{bandOffset};
                    j < bandOffset + chebyBands[i].extremas - 1u; ++j)
                {
                    middleValB = x[j + 1];
                    subIntervals.push_back(std::make_pair(middleValA, middleValB));
                    middleValA = middleValB;
                }
                if(middleValA != chebyBands[i].stop)
                    subIntervals.push_back(
                        std::make_pair(middleValA, chebyBands[i].stop));
            }
            bandOffset += chebyBands[i].extremas;
        }
    }
}

void findEigenExtrema(double& convergenceOrder,
        double& delta, std::vector<double>& eigenExtrema,
        std::vector<double>& x, std::vector<BandD>& chebyBands,
        int Nmax)
{
    // 1.   Split the initial [-1, 1] interval in subintervals
    //      in order that we can use a reasonable size matrix
    //      eigenvalue solver on the subintervals
    std::vector<IntervalD> subIntervals;
    double a = -1;
    double b = 1;

    splitInterval(subIntervals, chebyBands, x);

    //std::cout << "Number of subintervals: "
    //    << subIntervals.size() << std::endl;

    // 2.   Compute the barycentric variables (i.e. weights)
    //      needed for the current iteration

    std::vector<double> w(x.size());
    barycentricWeights(w, x);


    computeDelta(delta, w, x, chebyBands);
    //std::cout << "delta = " << delta << std::endl;

    std::vector<double> C(x.size());
    computeC(C, delta, x, chebyBands);

    // 3.   Use an eigenvalue solver on each subinterval to find the
    //      local extrema that are located inside the frequency bands
    std::vector<double> chebyNodes(Nmax + 1u);
    generateEquidistantNodes(chebyNodes, Nmax);
    applyCos(chebyNodes, chebyNodes);


    std::vector<std::pair<double, double>> potentialExtrema;
    std::vector<double> pEx;
    double extremaErrorValueLeft;
    double extremaErrorValueRight;
    double extremaErrorValue;
    computeError(extremaErrorValue, chebyBands[0].start,
            delta, x, C, w, chebyBands);
    potentialExtrema.push_back(std::make_pair(
            chebyBands[0].start, extremaErrorValue));


    for (std::size_t i = 0u; i < chebyBands.size() - 1; ++i)
    {
        computeError(extremaErrorValueLeft, chebyBands[i].stop,
                delta, x, C, w, chebyBands);
        computeError(extremaErrorValueRight, chebyBands[i + 1].start,
                delta, x, C, w, chebyBands);
        bool sgnLeft = std::signbit(extremaErrorValueLeft);
        bool sgnRight = std::signbit(extremaErrorValueRight);
        if (sgnLeft != sgnRight) {
            potentialExtrema.push_back(std::make_pair(
                    chebyBands[i].stop, extremaErrorValueLeft));
            potentialExtrema.push_back(std::make_pair(
                    chebyBands[i + 1].start, extremaErrorValueRight));
        } else {
            double abs1 = fabs(extremaErrorValueLeft);
            double abs2 = fabs(extremaErrorValueRight);
            if(abs1 > abs2)
                potentialExtrema.push_back(std::make_pair(
                        chebyBands[i].stop, extremaErrorValueLeft));
            else
                potentialExtrema.push_back(std::make_pair(
                        chebyBands[i + 1].start, extremaErrorValueRight));
        }
    }
    computeError(extremaErrorValue,
            chebyBands[chebyBands.size() - 1].stop,
            delta, x, C, w, chebyBands);
    potentialExtrema.push_back(std::make_pair(
            chebyBands[chebyBands.size() - 1].stop,
            extremaErrorValue));


    std::vector<std::vector<double>> pExs(subIntervals.size());

    #pragma omp parallel for
    for (std::size_t i = 0u; i < subIntervals.size(); ++i)
    {

        // find the Chebyshev nodes scaled to the current subinterval
        std::vector<double> siCN(Nmax + 1u);
        changeOfVariable(siCN, chebyNodes, subIntervals[i].first,
                subIntervals[i].second);

        // compute the Chebyshev interpolation function values on the
        // current subinterval
        std::vector<double> fx(Nmax + 1u);
        for (std::size_t j = 0u; j < fx.size(); ++j)
        {
            computeError(fx[j], siCN[j], delta, x, C, w,
                    chebyBands);

        }

        // compute the values of the CI coefficients and those of its
        // derivative
        std::vector<double> chebyCoeffs(Nmax + 1u);
        generateChebyshevCoefficients(chebyCoeffs, fx, Nmax);
        std::vector<double> derivCoeffs(Nmax);
        derivativeCoefficients2ndKind(derivCoeffs, chebyCoeffs);

        // solve the corresponding eigenvalue problem and determine the
        // local extrema situated in the current subinterval
        MatrixXd Cm(Nmax - 1u, Nmax - 1u);
        generateColleagueMatrix2ndKind(Cm, derivCoeffs);

        std::vector<double> eigenRoots;
        VectorXcd roots;
        determineEigenvalues(roots, Cm);
        getRealValues(eigenRoots, roots, a, b);
        changeOfVariable(eigenRoots, eigenRoots,
                subIntervals[i].first, subIntervals[i].second);
        for (std::size_t j = 0u; j < eigenRoots.size(); ++j)
            pExs[i].push_back(eigenRoots[j]);
        pExs[i].push_back(subIntervals[i].first);
        pExs[i].push_back(subIntervals[i].second);
    }

    for(std::size_t i = 0u; i < pExs.size(); ++i)
        for(std::size_t j = 0u; j < pExs[i].size(); ++j)
            pEx.push_back(pExs[i][j]);

    std::size_t startingOffset = potentialExtrema.size();
    potentialExtrema.resize(potentialExtrema.size() + pEx.size());
    #pragma omp parallel for
    for(std::size_t i = 0u; i < pEx.size(); ++i)
    {
        double valBuffer;
        computeError(valBuffer, pEx[i],
                delta, x, C, w, chebyBands);
        potentialExtrema[startingOffset + i] = std::make_pair(pEx[i], valBuffer);
    }

    // sort list of potential extrema in increasing order
    std::sort(potentialExtrema.begin(), potentialExtrema.end(),
            [](const std::pair<double, double>& lhs,
               const std::pair<double, double>& rhs) {
                return lhs.first < rhs.first;
            });

    eigenExtrema.clear();
    std::size_t extremaIt = 0u;
    std::vector<std::pair<double, double>> alternatingExtrema;
    double minError = INT_MAX;
    double maxError = INT_MIN;
    double absError;

    while (extremaIt < potentialExtrema.size())
    {
        std::pair<double, double> maxErrorPoint;
        maxErrorPoint = potentialExtrema[extremaIt];
        while(extremaIt < potentialExtrema.size() - 1 &&
            (std::signbit(maxErrorPoint.second) ==
             std::signbit(potentialExtrema[extremaIt + 1].second)))
        {
            ++extremaIt;
            if (fabs(maxErrorPoint.second) < fabs(potentialExtrema[extremaIt].second))
                maxErrorPoint = potentialExtrema[extremaIt];
        }
        alternatingExtrema.push_back(maxErrorPoint);
        ++extremaIt;
    }
    std::vector<std::pair<double, double>> bufferExtrema;
    //std::cout << "Alternating extrema: " << x.size() << " | "
    //    << alternatingExtrema.size() << std::endl;

    if(alternatingExtrema.size() < x.size())
    {
        std::cerr << "The exchange algorithm did not converge.\n";
        std::cerr << "TRIGGER: Not enough alternating extrema!\n"
            << "POSSIBLE CAUSE: Nmax too small\n";
        convergenceOrder = 2.0;
        return;
    }
    else if (alternatingExtrema.size() > x.size())
    {
        std::size_t remSuperfluous = alternatingExtrema.size() - x.size();
        if (remSuperfluous % 2 != 0)
        {
            if(remSuperfluous == 1u)
            {
                std::vector<double> x1, x2;
                x1.push_back(alternatingExtrema[0u].first);
                for(std::size_t i{1u}; i < alternatingExtrema.size() - 1; ++i)
                {
                    x1.push_back(alternatingExtrema[i].first);
                    x2.push_back(alternatingExtrema[i].first);
                }
                x2.push_back(alternatingExtrema[alternatingExtrema.size() - 1u].first);
                double delta1, delta2;
                computeDelta(delta1, x1, chebyBands);
                computeDelta(delta2, x2, chebyBands);
                delta1 = fabsl(delta1);
                delta2 = fabsl(delta2);
                std::size_t sIndex = 1u;
                if(delta1 > delta2)
                    sIndex = 0u;
                for(std::size_t i = sIndex; i < alternatingExtrema.size() + sIndex - 1u; ++i)
                    bufferExtrema.push_back(alternatingExtrema[i]);
                alternatingExtrema = bufferExtrema;
                bufferExtrema.clear();
            }
            else
            {
                double abs1 = fabs(alternatingExtrema[0].second);
                double abs2 = fabs(alternatingExtrema[alternatingExtrema.size() - 1].second);
                std::size_t sIndex = 0u;
                if (abs1 < abs2)
                    sIndex = 1u;
                for(std::size_t i = sIndex; i < alternatingExtrema.size() + sIndex - 1u; ++i)
                    bufferExtrema.push_back(alternatingExtrema[i]);
                alternatingExtrema = bufferExtrema;
                bufferExtrema.clear();
            }
        }


        while (alternatingExtrema.size() > x.size())
        {
            std::size_t toRemoveIndex = 0u;
            double minValToRemove = fminl(fabsl(alternatingExtrema[0].second),
                                              fabsl(alternatingExtrema[1].second));
            double removeBuffer;
            for (std::size_t i{1u}; i < alternatingExtrema.size() - 1; ++i)
            {
                removeBuffer = fminl(fabsl(alternatingExtrema[i].second),
                                   fabsl(alternatingExtrema[i + 1].second));
                if (removeBuffer < minValToRemove)
                {
                    minValToRemove = removeBuffer;
                    toRemoveIndex  = i;
                }
            }
            for (std::size_t i{0u}; i < toRemoveIndex; ++i)
                bufferExtrema.push_back(alternatingExtrema[i]);
            for (std::size_t i{toRemoveIndex + 2u}; i < alternatingExtrema.size(); ++i)
                bufferExtrema.push_back(alternatingExtrema[i]);
            alternatingExtrema = bufferExtrema;
            bufferExtrema.clear();
        }


    }
    if (alternatingExtrema.size() < x.size())
    {
        std::cerr << "Trouble!\n";
        exit(EXIT_FAILURE);
    }

    //std::cout << "After removal: " << alternatingExtrema.size() << std::endl;
    for (auto& it : alternatingExtrema)
    {
        eigenExtrema.push_back(it.first);
        absError = fabs(it.second);
        minError = fmin(minError, absError);
        maxError = fmax(maxError, absError);
    }

    //std::cout << "Min error = " << minError << std::endl;
    //std::cout << "Max error = " << maxError << std::endl;
    convergenceOrder = (maxError - minError) / maxError;
    //std::cout << "Convergence order = " << convergenceOrder << std::endl;
    // update the extrema count in each frequency band
    std::size_t bIndex = 0u;
    for(std::size_t i = 0; i < chebyBands.size(); ++i)
    {
        chebyBands[i].extremas = 0;
    }
    for(auto &it : eigenExtrema)
    {
        if(chebyBands[bIndex].start <= it && it <= chebyBands[bIndex].stop)
        {
            ++chebyBands[bIndex].extremas;
        }
        else
        {
            ++bIndex;
            ++chebyBands[bIndex].extremas;
        }
    }
}


// TODO: remember that this routine assumes that the information
// pertaining to the reference x and the frequency bands (i.e. the
// number of reference values inside each band) is given at the
// beginning of the execution
PMOutputD exchange(std::vector<double>& x,
        std::vector<BandD>& chebyBands, double eps,
        int Nmax)
{
    PMOutputD output;

    std::size_t degree = x.size() - 2u;
    std::sort(x.begin(), x.end(),
            [](const double& lhs,
               const double& rhs) {
                return lhs < rhs;
            });
    std::vector<double> startX{x};
    std::cout.precision(20);

    output.Q = 1;
    output.iter = 0u;
    //double lastDelta = 1.0;
    do {
        ++output.iter;
        //std::cout << "*********ITERATION " << output.iter << " **********\n";
        findEigenExtrema(output.Q, output.delta,
                output.x, startX, chebyBands, Nmax);
        startX = output.x;
        if(output.Q > 1.0)
            break;
        //if(output.delta < lastDelta)
        //    break;
        //std::cout << "*********ITERATION " << output.iter << " **********\n";
    } while (output.Q > eps && output.iter <= 100u);

    if(std::isnan(output.delta) || std::isnan(output.Q))
        std::cerr << "The exchange algorithm did not converge.\n"
            << "TRIGGER: numerical instability\n"
            << "POSSIBLE CAUSES: poor starting reference and/or "
            << "a too small value for Nmax.\n";

    if(output.iter >= 101u)
        std::cerr << "The exchange algorithm did not converge.\n"
            << "TRIGGER: exceeded iteration threshold of 100\n"
            << "POSSIBLE CAUSES: poor starting reference and/or "
            << "a too small value for Nmax.\n";


    output.h.resize(degree + 1u);
    std::vector<double> finalC(output.x.size());
    std::vector<double> finalAlpha(output.x.size());
    barycentricWeights(finalAlpha, output.x);
    double finalDelta = output.delta;
    output.delta = fabsl(output.delta);
    //std::cout << "MINIMAX delta = " << output.delta << std::endl;
    computeC(finalC, finalDelta, output.x, chebyBands);
    std::vector<double> finalChebyNodes(degree + 1);
    generateEquidistantNodes(finalChebyNodes, degree);
    applyCos(finalChebyNodes, finalChebyNodes);
    std::vector<double> fv(degree + 1);

    for (std::size_t i{0u}; i < fv.size(); ++i)
        computeApprox(fv[i], finalChebyNodes[i], output.x,
                finalC, finalAlpha);

    generateChebyshevCoefficients(output.h, fv, degree);

    return output;
}


// type I&II filters
PMOutputD firpm(std::size_t n,
        std::vector<double>const& f,
        std::vector<double>const& a,
        std::vector<double>const& w,
        double eps,
        int Nmax)
{
    std::vector<double> h;
    if( n % 2 != 0)
    {
        if((f[f.size() - 1u] == 1) && (a[a.size() - 1u] != 0))
        {
            std::cout << "Warning: Gain at Nyquist frequency different from 0.\n"
                << "Increasing the number of taps by 1 and passing to a "
                << " type I filter\n" << std::endl;
            ++n;
        } else {
            std::size_t degree = n / 2u;
            // TODO: error checking code
            std::vector<BandD> freqBands(w.size());
            std::vector<BandD> chebyBands;
            for(std::size_t i{0u}; i < freqBands.size(); ++i)
            {
                freqBands[i].start = M_PI * f[2u * i];
                if(i < freqBands.size() - 1u)
                    freqBands[i].stop  = M_PI * f[2u * i + 1u];
                else
                {
                    if(f[2u * i + 1u] == 1.0)
                    {
                        if(f[2u * i] < 0.9999)
                            freqBands[i].stop = M_PI * 0.9999;
                        else
                            freqBands[i].stop = M_PI * ((f[2u * i] + 1) / 2);
                    }
                    else
                        freqBands[i].stop  = M_PI * f[2u * i + 1u];
                }
                freqBands[i].space = BandSpace::FREQ;
                freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                {
                    if (a[2u * i] != a[2u * i + 1u]) {
                        if(bSpace == BandSpace::CHEBY)
                            x = acos(x);
                        return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                (x - freqBands[i].stop) * a[2u * i]) /
                                (freqBands[i].stop - freqBands[i].start)) / cos(x / 2);
                    }
                    if(bSpace == BandSpace::FREQ)
                        return a[2u * i] / cos(x / 2);
                    else
                        return a[2u * i] / sqrt((x + 1) / 2);
                };
                freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                {
                    if (bSpace == BandSpace::FREQ)
                        return cos(x / 2) * w[i];
                    else
                        return sqrt((x + 1) / 2) * w[i];
                };
            }
            std::vector<double> omega(degree + 2u);
            std::vector<double> x(degree + 2u);
            initUniformExtremas(omega, freqBands);
            applyCos(x, omega);
            bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);


            PMOutputD output = exchange(x, chebyBands, eps, Nmax);

            h.resize(n + 1u);
            h[0] = h[n] = output.h[degree] / 4;
            h[degree] = h[degree + 1] = (output.h[0] * 2 + output.h[1]) / 4;
            for(std::size_t i{2u}; i < degree + 1; ++i)
                h[degree + 1 - i] = h[degree + i] = (output.h[i - 1] + output.h[i]) / 4u;
            output.h = h;
            return output;
        }
    }


    std::size_t degree = n / 2u;
    // TODO: error checking code
    std::vector<BandD> freqBands(w.size());
    std::vector<BandD> chebyBands;
    for(std::size_t i{0u}; i < freqBands.size(); ++i)
    {
        freqBands[i].start = M_PI * f[2u * i];
        freqBands[i].stop  = M_PI * f[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
        {
            if (a[2u * i] != a[2u * i + 1u]) {
                if(bSpace == BandSpace::CHEBY)
                    x = acosl(x);
                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                        (x - freqBands[i].stop) * a[2u * i]) /
                        (freqBands[i].stop - freqBands[i].start);
            }
            return a[2u * i];
        };
        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
        {
            return w[i];
        };
    }

    std::vector<double> omega(degree + 2u);
    std::vector<double> x(degree + 2u);
    initUniformExtremas(omega, freqBands);
    applyCos(x, omega);
    bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

    //double finalDelta;
    std::vector<double> coeffs;
    std::vector<double> finalExtrema;
    //double convergenceOrder;

    PMOutputD output = exchange(x, chebyBands, eps, Nmax);

    h.resize(n + 1u);
    h[degree] = output.h[0];
    for(std::size_t i{0u}; i < degree; ++i)
        h[i] = h[n - i] = output.h[degree - i] / 2u;
    output.h = h;
    return output;

}


PMOutputD firpmRS(std::size_t n,
        std::vector<double>const& f,
        std::vector<double>const& a,
        std::vector<double>const& w,
        double eps,
        std::size_t depth,
        int Nmax,
        RootSolver root)
{
    if (depth == 0u) return firpm(n, f, a, w, eps, Nmax);
    std::vector<double> h;
    if( n % 2 != 0)
    {
        if((f[f.size() - 1u] == 1) && (a[a.size() - 1u] != 0))
        {
            std::cout << "Warning: Gain at Nyquist frequency different from 0.\n"
                << "Increasing the number of taps by 1 and passing to a "
                << " type I filter\n" << std::endl;
            ++n;
        } else {
            std::size_t degree = n / 2u;
            // TODO: error checking code
            std::vector<BandD> freqBands(w.size());
            std::vector<BandD> chebyBands;
            for(std::size_t i{0u}; i < freqBands.size(); ++i)
            {
                freqBands[i].start = M_PI * f[2u * i];
                if(i < freqBands.size() - 1u)
                    freqBands[i].stop  = M_PI * f[2u * i + 1u];
                else
                {
                    if(f[2u * i + 1u] == 1.0)
                    {
                        if(f[2u * i] < 0.9999)
                            freqBands[i].stop = M_PI * 0.9999;
                        else
                            freqBands[i].stop = M_PI * ((f[2u * i] + 1) / 2);
                    }
                    else
                        freqBands[i].stop  = M_PI * f[2u * i + 1u];
                }
                freqBands[i].space = BandSpace::FREQ;
                freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                {
                    if (a[2u * i] != a[2u * i + 1u]) {
                        if(bSpace == BandSpace::CHEBY)
                            x = acos(x);
                        return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                (x - freqBands[i].stop) * a[2u * i]) /
                                (freqBands[i].stop - freqBands[i].start)) / cos(x / 2);
                    }
                    if(bSpace == BandSpace::FREQ)
                        return a[2u * i] / cos(x / 2);
                    else
                        return a[2u * i] / sqrt((x + 1) / 2);
                };
                freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                {
                    if (bSpace == BandSpace::FREQ)
                        return cos(x / 2) * w[i];
                    else
                        return sqrt((x + 1) / 2) * w[i];
                };
            }

            std::vector<std::size_t> scaledDegrees(depth + 1u);
            scaledDegrees[depth] = degree;
            for(int i = depth - 1u; i >=0; --i)
            {
                scaledDegrees[i] = scaledDegrees[i + 1] / 2;
            }

            std::vector<double> omega(scaledDegrees[0] + 2u);
            std::vector<double> x(scaledDegrees[0] + 2u);
            PMOutputD output;
            if(root == RootSolver::UNIFORM) {
                initUniformExtremas(omega, freqBands);
                applyCos(x, omega);
                bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
                output = exchange(x, chebyBands, eps, Nmax);

            } else {
                std::function<double(double)> weightFunction = [=](double x) -> double
                {
                    for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                        if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
                            return chebyBands[i].weight(BandSpace::CHEBY, x);
                    return 1.0; // default value
                };
                std::vector<double> wam;
                generateWAM(wam, chebyBands, degree);
                MatrixXd A;
                generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
                std::vector<double> afpX;
                AFPPM(afpX, A, wam);
                bandCountPM(chebyBands, afpX);

                output = exchange(afpX, chebyBands, eps, Nmax);
            }



            for(std::size_t i = 1u; i <= depth; ++i)
            {
                x.clear();
                referenceScaling(x, chebyBands, freqBands, scaledDegrees[i] + 2u,
                        output.x, chebyBands, freqBands);
                output = exchange(x, chebyBands, eps, Nmax);
            }

            h.resize(n + 1u);
            h[0] = h[n] = output.h[degree] / 4;
            h[degree] = h[degree + 1] = (output.h[0] * 2 + output.h[1]) / 4;
            for(std::size_t i{2u}; i < degree + 1; ++i)
                h[degree + 1 - i] = h[degree + i] = (output.h[i - 1] + output.h[i]) / 4u;
            output.h = h;
            return output;
        }
    }


    std::size_t degree = n / 2u;
    // TODO: error checking code
    std::vector<BandD> freqBands(w.size());
    std::vector<BandD> chebyBands;
    for(std::size_t i{0u}; i < freqBands.size(); ++i)
    {
        freqBands[i].start = M_PI * f[2u * i];
        freqBands[i].stop  = M_PI * f[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
        {
            if (a[2u * i] != a[2u * i + 1u]) {
                if(bSpace == BandSpace::CHEBY)
                    x = acosl(x);
                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                        (x - freqBands[i].stop) * a[2u * i]) /
                        (freqBands[i].stop - freqBands[i].start);
            }
            return a[2u * i];
        };
        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
        {
            return w[i];
        };
    }

    std::vector<std::size_t> scaledDegrees(depth + 1u);
    scaledDegrees[depth] = degree;
    for(int i = depth - 1u; i >= 0; --i)
    {
        scaledDegrees[i] = scaledDegrees[i + 1] / 2;
    }

    std::vector<double> omega(scaledDegrees[0] + 2u);
    std::vector<double> x(scaledDegrees[0] + 2u);
    PMOutputD output;
    if(root == RootSolver::UNIFORM) {
        initUniformExtremas(omega, freqBands);
        applyCos(x, omega);
        bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
        output = exchange(x, chebyBands, eps, Nmax);

    } else {
        std::function<double(double)> weightFunction = [=](double x) -> double
        {
            for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
                    return chebyBands[i].weight(BandSpace::CHEBY, x);
            return 1.0; // default value
        };
        std::vector<double> wam;
        generateWAM(wam, chebyBands, degree);
        MatrixXd A;
        generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
        std::vector<double> afpX;
        AFPPM(afpX, A, wam);
        bandCountPM(chebyBands, afpX);

        output = exchange(afpX, chebyBands, eps, Nmax);
    }


    for(std::size_t i = 1u; i <= depth; ++i)
    {
        x.clear();
        referenceScaling(x, chebyBands, freqBands, scaledDegrees[i] + 2u,
                output.x, chebyBands, freqBands);
        output = exchange(x, chebyBands, eps, Nmax);
    }


    h.resize(n + 1u);
    h[degree] = output.h[0];
    for(std::size_t i{0u}; i < degree; ++i)
        h[i] = h[n - i] = output.h[degree - i] / 2u;
    output.h = h;
    return output;

}

// type III & IV filters
PMOutputD firpm(std::size_t n,
        std::vector<double>const& f,
        std::vector<double>const& a,
        std::vector<double>const& w,
        ftype type,
        double eps,
        int Nmax)
{
    PMOutputD output;
    std::vector<double> h;
    switch(type) {
        case ftype::FIR_DIFFERENTIATOR :
            {
                std::size_t degree = n / 2u;
                // TODO: error checking code
                 std::vector<double> fn = f;

                std::vector<BandD> freqBands(w.size());
                std::vector<BandD> chebyBands;
                double scaleFactor = a[1] / (f[1] * M_PI);
                if(n % 2 == 0) // Type III
                {
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    if(f[f.size() - 1u] == 1.0l)
                    {
                        if(f[f.size() - 2u] > 0.9999l)
                            fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
                        else
                            fn[f.size() - 1u] = 0.9999l;
                    }
                    --degree;
                    freqBands[0].start = M_PI * fn[0u];
                    freqBands[0].stop  = M_PI * fn[1u];
                    freqBands[0].space = BandSpace::FREQ;
                    freqBands[0].weight = [w](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (sin(x) / x) * w[0u];
                        }
                        else
                        {
                            return (sqrt(1.0l - x * x) / acos(x)) * w[0u];
                        }
                    };
                    freqBands[0].amplitude = [scaleFactor](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (x / sin(x)) * scaleFactor;
                        }
                        else
                        {
                            return (acos(x) / sqrt(1.0l - x * x)) * scaleFactor;
                        }
                    };
                    for(std::size_t i{1u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].weight = [w, i](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                            {
                                return sin(x) * w[i];
                            }
                            else
                            {
                                return sqrt(1.0l - x * x) * w[i];
                            }

                        };
                        freqBands[i].amplitude = [freqBands, a, i](BandSpace bSpace, double x) -> double
                        {
                            if (a[2u * i] != a[2u * i + 1u]) {
                                if(bSpace == BandSpace::CHEBY)
                                    x = acos(x);
                                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start);
                            }
                            return a[2u * i];

                        };

                    }

                } else {       // Type IV
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }

                    freqBands[0].start = M_PI * fn[0u];
                    freqBands[0].stop  = M_PI * fn[1u];
                    freqBands[0].space = BandSpace::FREQ;
                    freqBands[0].weight = [w](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (sin(x / 2) / x) * w[0u];
                        }
                        else
                        {
                            return (sin(acos(x) / 2) / acos(x)) * w[0u];
                        }
                    };
                    freqBands[0].amplitude = [scaleFactor](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (x / sin(x / 2)) * scaleFactor;
                        }
                        else
                        {
                            return (acos(x) / sin(acos(x) / 2)) * scaleFactor;
                        }
                    };
                    for(std::size_t i{1u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].weight = [w,i](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                            {
                                return sin(x / 2) * w[i];
                            }
                            else
                            {
                                return (sin(acos(x) / 2)) * w[i];
                            }

                        };
                        freqBands[i].amplitude = [freqBands, a, i](BandSpace bSpace, double x) -> double
                        {
                            if (a[2u * i] != a[2u * i + 1u]) {
                                if(bSpace == BandSpace::CHEBY)
                                    x = acos(x);
                                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start);
                            }
                            return a[2u * i];

                        };

                    }

                }

                std::vector<double> omega(degree + 2u);
                std::vector<double> x(degree + 2u);
                initUniformExtremas(omega, freqBands);
                applyCos(x, omega);
                bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

                output = exchange(x, chebyBands, eps, Nmax);

                h.resize(n + 1u);
                if(n % 2 == 0)
                {
                    h[degree + 1u] = 0;
                    h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
                    h[degree + 2u] = -h[degree];
                    h[1u] = output.h[degree - 1u] / 4;
                    h[2u * degree + 1u] = -h[1u];
                    h[0u] =  output.h[degree] / 4;
                    h[2u * (degree + 1u)] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
                        h[degree + 1u + i] = -h[degree + 1u - i];
                    }
                } else {
                    ++degree;
                    h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
                    h[degree] = -h[degree - 1u];
                    h[0u] = output.h[degree - 1u] / 4;
                    h[2u * degree - 1u] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
                        h[degree + i - 1u] = -h[degree - i];
                    }
                }


            }
            break;
        default : // FIR_HILBERT
            {
                std::size_t degree = n / 2u;
                std::vector<double> fn = f;
                // TODO: error checking code
                std::vector<BandD> freqBands(w.size());
                std::vector<BandD> chebyBands;
                if(n % 2 == 0) // Type III
                {
                    --degree;
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    if(f[f.size() - 1u] == 1.0l)
                    {
                        if(f[f.size() - 2u] > 0.9999l)
                            fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
                        else
                            fn[f.size() - 1u] = 0.9999l;
                    }

                    for(std::size_t i{0u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::CHEBY)
                                x = acosl(x);

                            if (a[2u * i] != a[2u * i + 1u]) {
                                return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start)) / sinl(x);
                            }
                            return a[2u * i] / sin(x);
                        };
                        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                                return w[i] * sin(x);
                            else
                                return w[i] * (sqrt(1.0l - x * x));
                        };
                    }
                } else {       // Type IV
                    if(f[0u] == 0.0l)
                    {
                        if(f[1u] < 0.00001l)
                            fn[0u] = fn[1u] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    for(std::size_t i{0u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::CHEBY)
                                x = acos(x);

                            if (a[2u * i] != a[2u * i + 1u]) {
                                return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start)) / sinl(x / 2);
                            }
                            return a[2u * i] / sin(x / 2);
                        };
                        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                                return w[i] * sin(x / 2);
                            else
                            {
                                x = acos(x);
                                return w[i] * (sin(x / 2));
                            }
                        };
                    }
                }
                std::vector<double> omega(degree + 2u);
                std::vector<double> x(degree + 2u);
                initUniformExtremas(omega, freqBands);
                applyCos(x, omega);
                bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

                output = exchange(x, chebyBands, eps, Nmax);

                h.resize(n + 1u);
                if(n % 2 == 0)
                {
                    h[degree + 1u] = 0;
                    h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
                    h[degree + 2u] = -h[degree];
                    h[1u] = output.h[degree - 1u] / 4;
                    h[2u * degree + 1u] = -h[1u];
                    h[0u] =  output.h[degree] / 4;
                    h[2u * (degree + 1u)] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
                        h[degree + 1u + i] = -h[degree + 1u - i];
                    }
                } else {
                    ++degree;
                    h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
                    h[degree] = -h[degree - 1u];
                    h[0u] = output.h[degree - 1u] / 4;
                    h[2u * degree - 1u] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
                        h[degree + i - 1u] = -h[degree - i];
                    }
                }

            }
            break;
    }
    output.h = h;
    return output;
}


PMOutputD firpmRS(std::size_t n,
        std::vector<double>const& f,
        std::vector<double>const& a,
        std::vector<double>const& w,
        ftype type,
        double eps,
        std::size_t depth,
        int Nmax,
        RootSolver root)
{
    if (depth == 0u) return firpm(n, f, a, w, type, eps, Nmax);
    PMOutputD output;
    std::vector<double> h;
    switch(type) {
        case ftype::FIR_DIFFERENTIATOR :
            {
                std::size_t degree = n / 2u;
                // TODO: error checking code
                std::vector<double> fn = f;

                std::vector<BandD> freqBands(w.size());
                std::vector<BandD> chebyBands;
                double scaleFactor = a[1] / (f[1] * M_PI);
                if(n % 2 == 0) // Type III
                {
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    if(f[f.size() - 1u] == 1.0l)
                    {
                        if(f[f.size() - 2u] > 0.9999l)
                            fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
                        else
                            fn[f.size() - 1u] = 0.9999l;
                    }
                    --degree;
                    freqBands[0].start = M_PI * fn[0u];
                    freqBands[0].stop  = M_PI * fn[1u];
                    freqBands[0].space = BandSpace::FREQ;
                    freqBands[0].weight = [w](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (sin(x) / x) * w[0u];
                        }
                        else
                        {
                            return (sqrt(1.0l - x * x) / acos(x)) * w[0u];
                        }
                    };
                    freqBands[0].amplitude = [scaleFactor](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (x / sin(x)) * scaleFactor;
                        }
                        else
                        {
                            return (acos(x) / sqrt(1.0l - x * x)) * scaleFactor;
                        }
                    };
                    for(std::size_t i{1u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].weight = [w, i](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                            {
                                return sin(x) * w[i];
                            }
                            else
                            {
                                return sqrt(1.0l - x * x) * w[i];
                            }

                        };
                        freqBands[i].amplitude = [freqBands, a, i](BandSpace bSpace, double x) -> double
                        {
                            if (a[2u * i] != a[2u * i + 1u]) {
                                if(bSpace == BandSpace::CHEBY)
                                    x = acos(x);
                                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start);
                            }
                            return a[2u * i];

                        };

                    }

                } else {       // Type IV
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }

                    freqBands[0].start = M_PI * fn[0u];
                    freqBands[0].stop  = M_PI * fn[1u];
                    freqBands[0].space = BandSpace::FREQ;
                    freqBands[0].weight = [w](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (sin(x / 2) / x) * w[0u];
                        }
                        else
                        {
                            return (sin(acos(x) / 2) / acos(x)) * w[0u];
                        }
                    };
                    freqBands[0].amplitude = [scaleFactor](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (x / sin(x / 2)) * scaleFactor;
                        }
                        else
                        {
                            return (acos(x) / sin(acos(x) / 2)) * scaleFactor;
                        }
                    };
                    for(std::size_t i{1u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].weight = [w,i](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                            {
                                return sin(x / 2) * w[i];
                            }
                            else
                            {
                                return (sin(acos(x) / 2)) * w[i];
                            }

                        };
                        freqBands[i].amplitude = [freqBands, a, i](BandSpace bSpace, double x) -> double
                        {
                            if (a[2u * i] != a[2u * i + 1u]) {
                                if(bSpace == BandSpace::CHEBY)
                                    x = acos(x);
                                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start);
                            }
                            return a[2u * i];

                        };

                    }

                }

            std::vector<std::size_t> scaledDegrees(depth + 1u);
            scaledDegrees[depth] = degree;
            for(int i = depth - 1u; i >=0; --i)
            {
                scaledDegrees[i] = scaledDegrees[i + 1] / 2;
            }

            std::vector<double> omega(scaledDegrees[0] + 2u);
            std::vector<double> x(scaledDegrees[0] + 2u);
            PMOutputD output;
            if(root == RootSolver::UNIFORM) {
                initUniformExtremas(omega, freqBands);
                applyCos(x, omega);
                bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
                output = exchange(x, chebyBands, eps, Nmax);

            } else {
                std::function<double(double)> weightFunction = [=](double x) -> double
                {
                    for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                        if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
                            return chebyBands[i].weight(BandSpace::CHEBY, x);
                    return 1.0; // default value
                };
                std::vector<double> wam;
                generateWAM(wam, chebyBands, degree);
                MatrixXd A;
                generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
                std::vector<double> afpX;
                AFPPM(afpX, A, wam);
                bandCountPM(chebyBands, afpX);

                output = exchange(afpX, chebyBands, eps, Nmax);
            }


            for(std::size_t i = 1u; i <= depth; ++i)
            {
                x.clear();
                referenceScaling(x, chebyBands, freqBands, scaledDegrees[i] + 2u,
                        output.x, chebyBands, freqBands);
                output = exchange(x, chebyBands, eps, Nmax);
            }

                h.resize(n + 1u);
                if(n % 2 == 0)
                {
                    h[degree + 1u] = 0;
                    h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
                    h[degree + 2u] = -h[degree];
                    h[1u] = output.h[degree - 1u] / 4;
                    h[2u * degree + 1u] = -h[1u];
                    h[0u] =  output.h[degree] / 4;
                    h[2u * (degree + 1u)] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
                        h[degree + 1u + i] = -h[degree + 1u - i];
                    }
                } else {
                    ++degree;
                    h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
                    h[degree] = -h[degree - 1u];
                    h[0u] = output.h[degree - 1u] / 4;
                    h[2u * degree - 1u] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
                        h[degree + i - 1u] = -h[degree - i];
                    }
                }


            }
            break;
        default : // FIR_HILBERT
            {
                std::size_t degree = n / 2u;
                std::vector<double> fn = f;
                // TODO: error checking code
                std::vector<BandD> freqBands(w.size());
                std::vector<BandD> chebyBands;
                if(n % 2 == 0) // Type III
                {
                    --degree;
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    if(f[f.size() - 1u] == 1.0l)
                    {
                        if(f[f.size() - 2u] > 0.9999l)
                            fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
                        else
                            fn[f.size() - 1u] = 0.9999l;
                    }

                    for(std::size_t i{0u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::CHEBY)
                                x = acosl(x);

                            if (a[2u * i] != a[2u * i + 1u]) {
                                return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start)) / sinl(x);
                            }
                            return a[2u * i] / sin(x);
                        };
                        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                                return w[i] * sin(x);
                            else
                                return w[i] * (sqrt(1.0l - x * x));
                        };
                    }
                } else {       // Type IV
                    if(f[0u] == 0.0l)
                    {
                        if(f[1u] < 0.00001l)
                            fn[0u] = fn[1u] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    for(std::size_t i{0u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::CHEBY)
                                x = acos(x);

                            if (a[2u * i] != a[2u * i + 1u]) {
                                return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start)) / sinl(x / 2);
                            }
                            return a[2u * i] / sin(x / 2);
                        };
                        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                                return w[i] * sin(x / 2);
                            else
                            {
                                x = acos(x);
                                return w[i] * (sin(x / 2));
                            }
                        };
                    }
                }



                std::vector<std::size_t> scaledDegrees(depth + 1u);
                scaledDegrees[depth] = degree;
                for(int i = depth - 1u; i >=0; --i)
                {
                    scaledDegrees[i] = scaledDegrees[i + 1] / 2;
                }

                std::vector<double> omega(scaledDegrees[0] + 2u);
                std::vector<double> x(scaledDegrees[0] + 2u);
                PMOutputD output;
                if(root == RootSolver::UNIFORM) {
                    initUniformExtremas(omega, freqBands);
                    applyCos(x, omega);
                    bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
                    output = exchange(x, chebyBands, eps, Nmax);

                } else {
                    std::function<double(double)> weightFunction = [=](double x) -> double
                    {
                        for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                            if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
                                return chebyBands[i].weight(BandSpace::CHEBY, x);
                        return 1.0; // default value
                    };
                    std::vector<double> wam;
                    generateWAM(wam, chebyBands, degree);
                    MatrixXd A;
                    generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
                    std::vector<double> afpX;
                    AFPPM(afpX, A, wam);
                    bandCountPM(chebyBands, afpX);

                    output = exchange(afpX, chebyBands, eps, Nmax);
                }



                for(std::size_t i = 1u; i <= depth; ++i)
                {
                    x.clear();
                    referenceScaling(x, chebyBands, freqBands, scaledDegrees[i] + 2u,
                            output.x, chebyBands, freqBands);
                    output = exchange(x, chebyBands, eps, Nmax);
                }

                h.resize(n + 1u);
                if(n % 2 == 0)
                {
                    h[degree + 1u] = 0;
                    h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
                    h[degree + 2u] = -h[degree];
                    h[1u] = output.h[degree - 1u] / 4;
                    h[2u * degree + 1u] = -h[1u];
                    h[0u] =  output.h[degree] / 4;
                    h[2u * (degree + 1u)] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
                        h[degree + 1u + i] = -h[degree + 1u - i];
                    }
                } else {
                    ++degree;
                    h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
                    h[degree] = -h[degree - 1u];
                    h[0u] = output.h[degree - 1u] / 4;
                    h[2u * degree - 1u] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
                        h[degree + i - 1u] = -h[degree - i];
                    }
                }

            }
            break;
    }
    output.h = h;
    return output;
}



// type I&II filters
PMOutputD firpmAFPPM(std::size_t n,
        std::vector<double>const& f,
        std::vector<double>const& a,
        std::vector<double>const& w,
        double eps,
        int Nmax)
{
    std::vector<double> h;
    if( n % 2 != 0)
    {
        if((f[f.size() - 1u] == 1) && (a[a.size() - 1u] != 0))
        {
            std::cout << "Warning: Gain at Nyquist frequency different from 0.\n"
                << "Increasing the number of taps by 1 and passing to a "
                << " type I filter\n" << std::endl;
            ++n;
        } else {
            std::size_t degree = n / 2u;
            // TODO: error checking code
            std::vector<BandD> freqBands(w.size());
            std::vector<BandD> chebyBands;
            for(std::size_t i{0u}; i < freqBands.size(); ++i)
            {
                freqBands[i].start = M_PI * f[2u * i];
                if(i < freqBands.size() - 1u)
                    freqBands[i].stop  = M_PI * f[2u * i + 1u];
                else
                {
                    if(f[2u * i + 1u] == 1.0)
                    {
                        if(f[2u * i] < 0.9999)
                            freqBands[i].stop = M_PI * 0.9999;
                        else
                            freqBands[i].stop = M_PI * ((f[2u * i] + 1) / 2);
                    }
                    else
                        freqBands[i].stop  = M_PI * f[2u * i + 1u];
                }
                freqBands[i].space = BandSpace::FREQ;
                freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                {
                    if (a[2u * i] != a[2u * i + 1u]) {
                        if(bSpace == BandSpace::CHEBY)
                            x = acos(x);
                        return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                (x - freqBands[i].stop) * a[2u * i]) /
                                (freqBands[i].stop - freqBands[i].start)) / cos(x / 2);
                    }
                    if(bSpace == BandSpace::FREQ)
                        return a[2u * i] / cos(x / 2);
                    else
                        return a[2u * i] / sqrt((x + 1) / 2);
                };
                freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                {
                    if (bSpace == BandSpace::FREQ)
                        return cos(x / 2) * w[i];
                    else
                        return sqrt((x + 1) / 2) * w[i];
                };
            }
            bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
            std::function<double(double)> weightFunction = [=](double x) -> double
            {
                for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                    if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
                        return chebyBands[i].weight(BandSpace::CHEBY, x);
                return 1.0; // default value
            };
            std::vector<double> wam;
            generateWAM(wam, chebyBands, degree);
            MatrixXd A;
            generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
            std::vector<double> afpX;
            AFPPM(afpX, A, wam);
            bandCountPM(chebyBands, afpX);


            PMOutputD output = exchange(afpX, chebyBands, eps, Nmax);

            h.resize(n + 1u);
            h[0] = h[n] = output.h[degree] / 4;
            h[degree] = h[degree + 1] = (output.h[0] * 2 + output.h[1]) / 4;
            for(std::size_t i{2u}; i < degree + 1; ++i)
                h[degree + 1 - i] = h[degree + i] = (output.h[i - 1] + output.h[i]) / 4u;
            output.h = h;
            return output;
        }
    }


    std::size_t degree = n / 2u;
    // TODO: error checking code
    std::vector<BandD> freqBands(w.size());
    std::vector<BandD> chebyBands;
    for(std::size_t i{0u}; i < freqBands.size(); ++i)
    {
        freqBands[i].start = M_PI * f[2u * i];
        freqBands[i].stop  = M_PI * f[2u * i + 1u];
        freqBands[i].space = BandSpace::FREQ;
        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
        {
            if (a[2u * i] != a[2u * i + 1u]) {
                if(bSpace == BandSpace::CHEBY)
                    x = acosl(x);
                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                        (x - freqBands[i].stop) * a[2u * i]) /
                        (freqBands[i].stop - freqBands[i].start);
            }
            return a[2u * i];
        };
        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
        {
            return w[i];
        };
    }

    bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
    std::function<double(double)> weightFunction = [=](double x) -> double
    {
        for(std::size_t i = 0u; i < chebyBands.size(); ++i)
            if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
                return chebyBands[i].weight(BandSpace::CHEBY, x);
        return 1.0; // default value
    };
    std::vector<double> wam;
    generateWAM(wam, chebyBands, degree);
    MatrixXd A;
    generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
    std::vector<double> afpX;
    AFPPM(afpX, A, wam);
    bandCountPM(chebyBands, afpX);



    //double finalDelta;
    std::vector<double> coeffs;
    std::vector<double> finalExtrema;
    //double convergenceOrder;

    PMOutputD output = exchange(afpX, chebyBands, eps, Nmax);

    h.resize(n + 1u);
    h[degree] = output.h[0];
    for(std::size_t i{0u}; i < degree; ++i)
        h[i] = h[n - i] = output.h[degree - i] / 2u;
    output.h = h;
    return output;

}

// type III & IV filters
PMOutputD firpmAFPPM(std::size_t n,
        std::vector<double>const& f,
        std::vector<double>const& a,
        std::vector<double>const& w,
        ftype type,
        double eps,
        int Nmax)
{
    PMOutputD output;
    std::vector<double> h;
    switch(type) {
        case ftype::FIR_DIFFERENTIATOR :
            {
                std::size_t degree = n / 2u;
                // TODO: error checking code
                 std::vector<double> fn = f;

                std::vector<BandD> freqBands(w.size());
                std::vector<BandD> chebyBands;
                double scaleFactor = a[1] / (f[1] * M_PI);
                if(n % 2 == 0) // Type III
                {
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    if(f[f.size() - 1u] == 1.0l)
                    {
                        if(f[f.size() - 2u] > 0.9999l)
                            fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
                        else
                            fn[f.size() - 1u] = 0.9999l;
                    }
                    --degree;
                    freqBands[0].start = M_PI * fn[0u];
                    freqBands[0].stop  = M_PI * fn[1u];
                    freqBands[0].space = BandSpace::FREQ;
                    freqBands[0].weight = [w](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (sin(x) / x) * w[0u];
                        }
                        else
                        {
                            return (sqrt(1.0l - x * x) / acos(x)) * w[0u];
                        }
                    };
                    freqBands[0].amplitude = [scaleFactor](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (x / sin(x)) * scaleFactor;
                        }
                        else
                        {
                            return (acos(x) / sqrt(1.0l - x * x)) * scaleFactor;
                        }
                    };
                    for(std::size_t i{1u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].weight = [w, i](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                            {
                                return sin(x) * w[i];
                            }
                            else
                            {
                                return sqrt(1.0l - x * x) * w[i];
                            }

                        };
                        freqBands[i].amplitude = [freqBands, a, i](BandSpace bSpace, double x) -> double
                        {
                            if (a[2u * i] != a[2u * i + 1u]) {
                                if(bSpace == BandSpace::CHEBY)
                                    x = acos(x);
                                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start);
                            }
                            return a[2u * i];

                        };

                    }

                } else {       // Type IV
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }

                    freqBands[0].start = M_PI * fn[0u];
                    freqBands[0].stop  = M_PI * fn[1u];
                    freqBands[0].space = BandSpace::FREQ;
                    freqBands[0].weight = [w](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (sin(x / 2) / x) * w[0u];
                        }
                        else
                        {
                            return (sin(acos(x) / 2) / acos(x)) * w[0u];
                        }
                    };
                    freqBands[0].amplitude = [scaleFactor](BandSpace bSpace, double x) -> double
                    {
                        if(bSpace == BandSpace::FREQ)
                        {
                            return (x / sin(x / 2)) * scaleFactor;
                        }
                        else
                        {
                            return (acos(x) / sin(acos(x) / 2)) * scaleFactor;
                        }
                    };
                    for(std::size_t i{1u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].weight = [w,i](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                            {
                                return sin(x / 2) * w[i];
                            }
                            else
                            {
                                return (sin(acos(x) / 2)) * w[i];
                            }

                        };
                        freqBands[i].amplitude = [freqBands, a, i](BandSpace bSpace, double x) -> double
                        {
                            if (a[2u * i] != a[2u * i + 1u]) {
                                if(bSpace == BandSpace::CHEBY)
                                    x = acos(x);
                                return ((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start);
                            }
                            return a[2u * i];

                        };

                    }

                }

                bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
                std::function<double(double)> weightFunction = [=](double x) -> double
                {
                    for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                        if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
                            return chebyBands[i].weight(BandSpace::CHEBY, x);
                    return 1.0; // default value
                };
                std::vector<double> wam;
                generateWAM(wam, chebyBands, degree);
                MatrixXd A;
                generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
                std::vector<double> afpX;
                AFPPM(afpX, A, wam);
                bandCountPM(chebyBands, afpX);


                output = exchange(afpX, chebyBands, eps, Nmax);

                h.resize(n + 1u);
                if(n % 2 == 0)
                {
                    h[degree + 1u] = 0;
                    h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
                    h[degree + 2u] = -h[degree];
                    h[1u] = output.h[degree - 1u] / 4;
                    h[2u * degree + 1u] = -h[1u];
                    h[0u] =  output.h[degree] / 4;
                    h[2u * (degree + 1u)] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
                        h[degree + 1u + i] = -h[degree + 1u - i];
                    }
                } else {
                    ++degree;
                    h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
                    h[degree] = -h[degree - 1u];
                    h[0u] = output.h[degree - 1u] / 4;
                    h[2u * degree - 1u] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
                        h[degree + i - 1u] = -h[degree - i];
                    }
                }


            }
            break;
        default : // FIR_HILBERT
            {
                std::size_t degree = n / 2u;
                std::vector<double> fn = f;
                // TODO: error checking code
                std::vector<BandD> freqBands(w.size());
                std::vector<BandD> chebyBands;
                if(n % 2 == 0) // Type III
                {
                    --degree;
                    if(f[0u] == 0.0l)
                    {
                        if(fn[1u] < 0.00001l)
                            fn[0u] = fn[1] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    if(f[f.size() - 1u] == 1.0l)
                    {
                        if(f[f.size() - 2u] > 0.9999l)
                            fn[f.size() - 1u] = (1.0 + f[f.size() - 2u]) / 2;
                        else
                            fn[f.size() - 1u] = 0.9999l;
                    }

                    for(std::size_t i{0u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::CHEBY)
                                x = acosl(x);

                            if (a[2u * i] != a[2u * i + 1u]) {
                                return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start)) / sinl(x);
                            }
                            return a[2u * i] / sin(x);
                        };
                        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                                return w[i] * sin(x);
                            else
                                return w[i] * (sqrt(1.0l - x * x));
                        };
                    }
                } else {       // Type IV
                    if(f[0u] == 0.0l)
                    {
                        if(f[1u] < 0.00001l)
                            fn[0u] = fn[1u] / 2;
                        else
                            fn[0u] = 0.00001l;
                    }
                    for(std::size_t i{0u}; i < freqBands.size(); ++i)
                    {
                        freqBands[i].start = M_PI * fn[2u * i];
                        freqBands[i].stop  = M_PI * fn[2u * i + 1u];
                        freqBands[i].space = BandSpace::FREQ;
                        freqBands[i].amplitude = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::CHEBY)
                                x = acos(x);

                            if (a[2u * i] != a[2u * i + 1u]) {
                                return (((x - freqBands[i].start) * a[2u * i + 1u] -
                                        (x - freqBands[i].stop) * a[2u * i]) /
                                        (freqBands[i].stop - freqBands[i].start)) / sinl(x / 2);
                            }
                            return a[2u * i] / sin(x / 2);
                        };
                        freqBands[i].weight = [=](BandSpace bSpace, double x) -> double
                        {
                            if(bSpace == BandSpace::FREQ)
                                return w[i] * sin(x / 2);
                            else
                            {
                                x = acos(x);
                                return w[i] * (sin(x / 2));
                            }
                        };
                    }
                }
                bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);
                std::function<double(double)> weightFunction = [=](double x) -> double
                {
                    for(std::size_t i = 0u; i < chebyBands.size(); ++i)
                        if(chebyBands[i].start <= x && x <= chebyBands[i].stop)
                            return chebyBands[i].weight(BandSpace::CHEBY, x);
                    return 1.0; // default value
                };
                std::vector<double> wam;
                generateWAM(wam, chebyBands, degree);
                MatrixXd A;
                generateVandermondeMatrix(A, degree + 1u, wam, weightFunction);
                std::vector<double> afpX;
                AFPPM(afpX, A, wam);
                bandCountPM(chebyBands, afpX);


                output = exchange(afpX, chebyBands, eps, Nmax);

                h.resize(n + 1u);
                if(n % 2 == 0)
                {
                    h[degree + 1u] = 0;
                    h[degree] = (output.h[0u] * 2.0l - output.h[2]) / 4u;
                    h[degree + 2u] = -h[degree];
                    h[1u] = output.h[degree - 1u] / 4;
                    h[2u * degree + 1u] = -h[1u];
                    h[0u] =  output.h[degree] / 4;
                    h[2u * (degree + 1u)] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree + 1u - i] = (output.h[i - 1u] - output.h[i + 1u]) / 4;
                        h[degree + 1u + i] = -h[degree + 1u - i];
                    }
                } else {
                    ++degree;
                    h[degree - 1u] = (output.h[0u] * 2.0l - output.h[1u]) / 4;
                    h[degree] = -h[degree - 1u];
                    h[0u] = output.h[degree - 1u] / 4;
                    h[2u * degree - 1u] = -h[0u];
                    for(std::size_t i{2u}; i < degree; ++i)
                    {
                        h[degree - i] = (output.h[i - 1u] - output.h[i]) / 4;
                        h[degree + i - 1u] = -h[degree - i];
                    }
                }

            }
            break;
    }
    output.h = h;
    return output;
}
