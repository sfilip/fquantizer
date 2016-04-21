#include "filter/afp.h"
#include "filter/band.h"
#include "filter/barycentric.h"
#include "filter/cheby.h"
#include "filter/conv.h"
#include "filter/eigenvalue.h"
#include "filter/plotting.h"
#include "filter/pm.h"
#include "filter/roots.h"
#include "filter/util.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <vector>

TEST(quantization_test, A35_8) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(2);

  freqBands[0].start = pi * 0.0;
  freqBands[0].stop = pi * 0.4;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.5;
  freqBands[1].stop = pi * 1.0;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  std::size_t degree = 17;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 128;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(quantization_test, A45_8) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(2);

  freqBands[0].start = pi * 0.0;
  freqBands[0].stop = pi * 0.4;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.5;
  freqBands[1].stop = pi * 1.0;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  std::size_t degree = 22;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 128;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(quantization_test, A125_21) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(2);

  freqBands[0].start = pi * 0.0;
  freqBands[0].stop = pi * 0.4;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.5;
  freqBands[1].stop = pi * 1.0;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  std::size_t degree = 62;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 1048576;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1.5;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  // slight weight change
  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1.5;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(quantization_test, B35_9) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(2);

  freqBands[0].start = pi * 0.0;
  freqBands[0].stop = pi * 0.4;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.5;
  freqBands[1].stop = pi * 1.0;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(10);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  std::size_t degree = 17;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 256;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 30;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  // slight weight change
  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 30;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(quantization_test, B45_9) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(2);

  freqBands[0].start = pi * 0.0;
  freqBands[0].stop = pi * 0.4;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.5;
  freqBands[1].stop = pi * 1.0;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(10);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  std::size_t degree = 22;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 256;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 12;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  // NO weight change
  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 10;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(quantization_test, B125_9) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(2);

  freqBands[0].start = pi * 0.0;
  freqBands[0].stop = pi * 0.4;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.5;
  freqBands[1].stop = pi * 1.0;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(10);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  std::size_t degree = 62;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 2097152;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 10;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  // NO weight change
  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 10;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(quantization_test, C35_8) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(3);

  freqBands[0].start = pi * 0.0;
  freqBands[0].stop = pi * 0.24;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.4;
  freqBands[1].stop = pi * 0.68;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  freqBands[2].start = pi * 0.84;
  freqBands[2].stop = pi;
  freqBands[2].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[2].space = BandSpace::FREQ;
  freqBands[2].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  std::size_t degree = 17;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 128;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  // NO weight change
  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(quantization_test, C45_8) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(3);

  freqBands[0].start = pi * 0.0;
  freqBands[0].stop = pi * 0.24;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.4;
  freqBands[1].stop = pi * 0.68;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  freqBands[2].start = pi * 0.84;
  freqBands[2].stop = pi;
  freqBands[2].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[2].space = BandSpace::FREQ;
  freqBands[2].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  std::size_t degree = 22;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 128;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  // NO weight change
  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(quantization_test, C125_21) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(3);

  freqBands[0].start = pi * 0.0;
  freqBands[0].stop = pi * 0.24;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.4;
  freqBands[1].stop = pi * 0.68;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  freqBands[2].start = pi * 0.84;
  freqBands[2].stop = pi;
  freqBands[2].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[2].space = BandSpace::FREQ;
  freqBands[2].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  std::size_t degree = 62;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 1048576;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  // NO weight change
  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(quantization_test, D35_9) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(3);

  freqBands[0].start = pi * 0.0;
  freqBands[0].stop = pi * 0.24;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.4;
  freqBands[1].stop = pi * 0.68;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(10);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  freqBands[2].start = pi * 0.84;
  freqBands[2].stop = pi;
  freqBands[2].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[2].space = BandSpace::FREQ;
  freqBands[2].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  std::size_t degree = 17;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 256;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[1].start <= points[i] && chebyBands[1].stop >= points[i])
      weights[i] = 10;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  // NO weight change
  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[1].start <= points[i] && chebyBands[1].stop >= points[i])
      weights[i] = 10;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(quantization_test, D45_9) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(3);

  freqBands[0].start = pi * 0.0;
  freqBands[0].stop = pi * 0.24;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.4;
  freqBands[1].stop = pi * 0.68;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(10);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  freqBands[2].start = pi * 0.84;
  freqBands[2].stop = pi;
  freqBands[2].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[2].space = BandSpace::FREQ;
  freqBands[2].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  std::size_t degree = 22;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 256;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[1].start <= points[i] && chebyBands[1].stop >= points[i])
      weights[i] = 10;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  // NO weight change
  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[1].start <= points[i] && chebyBands[1].stop >= points[i])
      weights[i] = 10;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(quantization_test, D125_22) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(3);

  freqBands[0].start = pi * 0.0;
  freqBands[0].stop = pi * 0.24;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.4;
  freqBands[1].stop = pi * 0.68;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(10);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  freqBands[2].start = pi * 0.84;
  freqBands[2].stop = pi;
  freqBands[2].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[2].space = BandSpace::FREQ;
  freqBands[2].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  std::size_t degree = 62;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 2097152;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[1].start <= points[i] && chebyBands[1].stop >= points[i])
      weights[i] = 10;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  // NO weight change
  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[1].start <= points[i] && chebyBands[1].stop >= points[i])
      weights[i] = 10;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(quantization_test, E35_8) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(2);

  freqBands[0].start = pi * 0.02;
  freqBands[0].stop = pi * 0.42;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.52;
  freqBands[1].stop = pi * 0.98;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  std::size_t degree = 17;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 128;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  // NO weight change
  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(firpm_quantization_test, E45_8) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(2);

  freqBands[0].start = pi * 0.02;
  freqBands[0].stop = pi * 0.42;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.52;
  freqBands[1].stop = pi * 0.98;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  std::size_t degree = 22;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 128;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  // NO weight change
  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}

TEST(firpm_quantization_test, E125_21) {
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(2);

  freqBands[0].start = pi * 0.02;
  freqBands[0].stop = pi * 0.42;
  freqBands[0].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[0].space = BandSpace::FREQ;
  freqBands[0].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };

  freqBands[1].start = pi * 0.52;
  freqBands[1].stop = pi * 0.98;
  freqBands[1].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(1);
  };
  freqBands[1].space = BandSpace::FREQ;
  freqBands[1].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
    return mpfr::mpreal(0);
  };

  std::size_t degree = 62;
  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  PMOutput output = exchange(x, chebyBands);
  ASSERT_LT(output.Q.toDouble(), 0.1e-1);

  std::cout << "Parks-McClellan delta = " << output.delta << std::endl;

  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;

  mpfr::mpreal scalingFactor = 1048576;

  std::vector<mpfr::mpreal> zeros;
  findEigenZeros(output.h, zeros, freqBands, chebyBands, prec);

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);

  std::vector<mpfr::mpreal> points;
  points = zeros;

  for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);
  }

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1.3;
    else
      weights[i] = 1;
  }

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, zeroLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  std::pair<mpfr::mpreal, mpfr::mpreal> norm;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bandNorms(
      chebyBands.size());
  computeNorm(norm, bandNorms, zeroLLL, freqBands, chebyBands, prec);
  std::cout << "ZERO error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

  std::function<mpfr::mpreal(mpfr::mpreal)> weight =
      [=](mpfr::mpreal x) -> mpfr::mpreal {
    for (std::size_t i{0u}; i < chebyBands.size(); ++i)
      if (x <= chebyBands[i].stop && x >= chebyBands[i].start)
        return chebyBands[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<mpfr::mpreal> chebyMesh;
  chebyMeshGeneration(chebyMesh, chebyBands, degree, prec);

  MatrixXq Acheby;
  generateAFPMatrix(Acheby, degree + 1u, chebyMesh, weight);
  std::vector<mpfr::mpreal> afpXcheby;
  AFP(afpXcheby, Acheby, chebyMesh);

  points = afpXcheby;
  weights.resize(points.size());

  // weight change
  for (std::size_t i = 0u; i < points.size(); ++i) {
    if (chebyBands[0].start <= points[i] && chebyBands[0].stop >= points[i])
      weights[i] = 1.3;
    else
      weights[i] = 1;
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, output.h,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;
}
