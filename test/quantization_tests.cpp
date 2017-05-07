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

using namespace std::chrono;

void testLatticeBasedQuantization(std::vector<double> vBands,
        std::vector<double> vAmplitude, std::vector<double> vWeights,
        std::size_t degree, mpfr::mpreal scalingFactor)
{
  using mpfr::mpreal;
  std::size_t prec = 200ul;
  mpreal::set_default_prec(prec);
  mpfr::mpreal pi = mpfr::const_pi(prec);

  std::vector<Band> freqBands(vBands.size() / 2);

  for(std::size_t i{0u}; i < vBands.size() / 2; ++i)
  {
      freqBands[i].start = pi * vBands[2*i];
      freqBands[i].stop = pi * vBands[2*i+1];
      freqBands[i].weight = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
        return mpfr::mpreal(vWeights[i]);
      };
      freqBands[i].space = BandSpace::FREQ;
      freqBands[i].amplitude = [=](BandSpace, mpfr::mpreal) -> mpfr::mpreal {
        return mpfr::mpreal(vAmplitude[i]);
      };
  }


  std::vector<mpfr::mpreal> a;
  mpfr::mpreal finalDelta;
  std::vector<Band> chebyBands;
  std::vector<mpfr::mpreal> omega(degree + 2u);
  std::vector<mpfr::mpreal> x(degree + 2u);
  initUniformExtremas(omega, freqBands, prec);
  applyCos(x, omega);
  std::vector<mpfr::mpreal> secondPass = x;
  bandConversion(chebyBands, freqBands, ConversionDirection::FROMFREQ);

  double piD = M_PI;

  std::vector<BandD> freqBandsD(vBands.size() / 2);

  for(std::size_t i{0u}; i < vBands.size() / 2; ++i)
  {
      freqBandsD[i].start = piD * vBands[2*i];
      freqBandsD[i].stop = piD * vBands[2*i+1];
      freqBandsD[i].weight = [=](BandSpace, double) -> double {
        return double(vWeights[i]);
      };
      freqBandsD[i].space = BandSpace::FREQ;
      freqBandsD[i].amplitude = [=](BandSpace, double) -> double {
        return double(vAmplitude[i]);
      };
  }

  std::vector<BandD> chebyBandsD;
  std::vector<double> omegaD(degree + 2u);
  std::vector<double> xD(degree + 2u);
  initUniformExtremas(omegaD, freqBandsD);
  applyCos(xD, omegaD);
  bandConversion(chebyBandsD, freqBandsD, ConversionDirection::FROMFREQ);

  PMOutputD outputD = exchange(xD, chebyBandsD, 0.0001, 16);
  ASSERT_LT(outputD.Q, 0.1e-1);

  std::cout << "Parks-McClellan delta = " << outputD.delta << std::endl;


  std::vector<mpfr::mpreal> zeroLLL;
  mpfr::mpreal error = 1u;
  mpfr::mpreal currentError;
  mpfr::mpreal currentFactor;


  a.resize(outputD.h.size());
  for(std::size_t i{0u}; i < a.size(); ++i)
    a[i] = outputD.h[i];


  std::vector<mpfr::mpreal> zeros;


  std::vector<double> zerosD;
  findEigenZeros(outputD.h, zerosD, outputD.x, freqBandsD, chebyBandsD);

  zeros.resize(zerosD.size());
  for(std::size_t i{0u}; i < zerosD.size(); ++i)
      zeros[i] = zerosD[i];

  std::vector<mpfr::mpreal> fixedA(0);
  std::vector<mpfr::mpreal> fullA = a;
  for (auto &it : fixedA)
    fullA.push_back(it);


  std::vector<mpfr::mpreal> points;
  points = zeros;

  std::size_t bandIndex{0u};
  while(bandIndex < chebyBands.size() - 1u && points.size() < degree + 1u)
  {
    points.push_back((chebyBands[bandIndex].stop + chebyBands[bandIndex + 1u].start) / 2);
    ++bandIndex;
  }

  /*for (auto &it : chebyBands) {
    points.push_back(it.start);
    points.push_back(it.stop);

  }*/

  std::vector<mpfr::mpreal> weights(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    weights[i] = 1;
    bandIndex = 0u;
    bool found = false;
    while((bandIndex < chebyBands.size()) && !found ) {
        if (chebyBands[bandIndex].start <= points[i]
                && chebyBands[bandIndex].stop >= points[i]) {
            weights[i] = chebyBands[bandIndex].weight(BandSpace::CHEBY, points[i]);
            found = true;
        }
        ++bandIndex;
    }
  }


  // replace output.h with a
  fpminimaxWithNeighborhoodSearchDiscreteMinimax(currentError, zeroLLL, a,
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


  points.clear();
  for(auto & it :outputD.x)
      points.push_back(mpfr::mpreal(it));

  weights.resize(points.size());

  // update the points
  for(std::size_t i{0u}; i < points.size(); ++i)
  {
      for(std::size_t j{0u}; j < chebyBands.size() - 1; ++j) {
        if (mpfr::abs(points[i] - chebyBands[j].stop) < 1e-14)
            points[i] = chebyBands[j].stop;
        if (mpfr::abs(points[i] - chebyBands[j + 1].start) < 1e-14)
          points[i] = chebyBands[j + 1].start;
      }

  }

  for (std::size_t i = 0u; i < points.size(); ++i) {
    weights[i] = 1;
    bandIndex = 0u;
    bool found = false;
    while((bandIndex < chebyBands.size()) && !found ) {
        if (chebyBands[bandIndex].start <= points[i]
                && chebyBands[bandIndex].stop >= points[i]) {
            weights[i] = chebyBands[bandIndex].weight(BandSpace::CHEBY, points[i]);
            found = true;
        }
        ++bandIndex;
    }
  }

  std::vector<mpfr::mpreal> extremaLLL;

  // replace output.h with a
  fpminimaxWithNeighborhoodSearchDiscrete(currentError, extremaLLL, a,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, extremaLLL, freqBands, chebyBands, prec);
  std::cout << "EXTREMA error = " << norm.second << std::endl;
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

  std::function<double(double)> weightD =
      [=](double x) -> double {
    for (std::size_t i{0u}; i < chebyBandsD.size(); ++i)
      if (x <= chebyBandsD[i].stop && x >= chebyBandsD[i].start)
        return chebyBandsD[i].weight(BandSpace::CHEBY, x);
    return 1;
  };

  std::vector<double> chebyMeshD;
  chebyMeshGeneration(chebyMeshD, chebyBandsD, degree);

  MatrixXd AchebyD;
  generateAFPMatrix(AchebyD, degree + 1u, chebyMeshD, weightD);
  std::vector<double> afpXchebyD;
  AFP(afpXchebyD, AchebyD, chebyMeshD);
  points.resize(afpXchebyD.size());
  for(std::size_t i{0u}; i < afpXchebyD.size(); ++i) {
      points[i] = afpXchebyD[i];
  }

  // update the points
  for(std::size_t i{0u}; i < points.size(); ++i)
  {
      for(std::size_t j{0u}; j < chebyBands.size() - 1; ++j) {
        if (mpfr::abs(points[i] - chebyBands[j].stop) < 1e-14)
            points[i] = chebyBands[j].stop;
        if (mpfr::abs(points[i] - chebyBands[j + 1].start) < 1e-14)
          points[i] = chebyBands[j + 1].start;
      }

  }

  weights.resize(points.size());

  for (std::size_t i = 0u; i < points.size(); ++i) {
    weights[i] = 1;
    bandIndex = 0u;
    bool found = false;
    while((bandIndex < chebyBands.size()) && !found ) {
        if (chebyBands[bandIndex].start <= points[i]
                && chebyBands[bandIndex].stop >= points[i]) {
            weights[i] = chebyBands[bandIndex].weight(BandSpace::CHEBY, points[i]);
          found = true;
        }
        ++bandIndex;
    }
  }

  std::vector<mpfr::mpreal> afpLLL;

  fpminimaxWithNeighborhoodSearchDiscrete(currentError, afpLLL, a,
                                          fixedA, points, freqBands, weights,
                                          scalingFactor, prec);

  computeNorm(norm, bandNorms, afpLLL, freqBands, chebyBands, prec);
  std::cout << "AFP error = " << norm.second << std::endl;
  for (std::size_t i = 0u; i < bandNorms.size(); ++i)
    std::cout << "Band " << i << " error = " << bandNorms[i].second
              << std::endl;

}


TEST(quantization_test, A35_8) {

  std::size_t degree = 17;
  mpfr::mpreal scalingFactor = 128;

  testLatticeBasedQuantization({0.0, 0.4, 0.5, 1.0}, {1.0, 0.0},
        {1.0, 1.0}, degree, scalingFactor);
}


TEST(quantization_test, A45_8) {

  std::size_t degree = 22;
  mpfr::mpreal scalingFactor = 128;

  testLatticeBasedQuantization({0.0, 0.4, 0.5, 1.0}, {1.0, 0.0},
        {1.0, 1.0}, degree, scalingFactor);
}

TEST(quantization_test, A125_21) {

  std::size_t degree = 62;
  mpfr::mpreal scalingFactor = 1048576;

  testLatticeBasedQuantization({0.0, 0.4, 0.5, 1.0}, {1.0, 0.0},
        {1.0, 1.0}, degree, scalingFactor);
}

TEST(quantization_test, B35_9) {

  std::size_t degree = 17;
  mpfr::mpreal scalingFactor = 256;

  testLatticeBasedQuantization({0.0, 0.4, 0.5, 1.0}, {1.0, 0.0},
        {1.0, 10.0}, degree, scalingFactor);
}

TEST(quantization_test, B45_9) {

  std::size_t degree = 22;
  mpfr::mpreal scalingFactor = 256;

  testLatticeBasedQuantization({0.0, 0.4, 0.5, 1.0}, {1.0, 0.0},
        {1.0, 10.0}, degree, scalingFactor);
}

TEST(quantization_test, B125_22) {

  std::size_t degree = 62;
  mpfr::mpreal scalingFactor = 2097152;

  testLatticeBasedQuantization({0.0, 0.4, 0.5, 1.0}, {1.0, 0.0},
        {1.0, 10.0}, degree, scalingFactor);
}


TEST(quantization_test, C35_8) {

  std::size_t degree = 17;
  mpfr::mpreal scalingFactor = 128;

  testLatticeBasedQuantization({0.0, 0.24, 0.4, 0.68, 0.84, 1.0}, {1.0, 0.0, 1.0},
        {1.0, 1.0, 1.0}, degree, scalingFactor);
}

TEST(quantization_test, C45_8) {

  std::size_t degree = 22;
  mpfr::mpreal scalingFactor = 128;

  testLatticeBasedQuantization({0.0, 0.24, 0.4, 0.68, 0.84, 1.0}, {1.0, 0.0, 1.0},
        {1.0, 1.0, 1.0}, degree, scalingFactor);

}


TEST(quantization_test, C125_21) {

  std::size_t degree = 62;
  mpfr::mpreal scalingFactor = 1048576;

  testLatticeBasedQuantization({0.0, 0.24, 0.4, 0.68, 0.84, 1.0}, {1.0, 0.0, 1.0},
        {1.0, 1.0, 1.0}, degree, scalingFactor);

}

TEST(quantization_test, D35_9) {

  std::size_t degree = 17;
  mpfr::mpreal scalingFactor = 256;

  testLatticeBasedQuantization({0.0, 0.24, 0.4, 0.68, 0.84, 1.0}, {1.0, 0.0, 1.0},
        {1.0, 10.0, 1.0}, degree, scalingFactor);
}


TEST(quantization_test, D45_9) {

  std::size_t degree = 22;
  mpfr::mpreal scalingFactor = 256;

  testLatticeBasedQuantization({0.0, 0.24, 0.4, 0.68, 0.84, 1.0}, {1.0, 0.0, 1.0},
        {1.0, 10.0, 1.0}, degree, scalingFactor);

}

TEST(quantization_test, D125_22) {

  std::size_t degree = 62;
  mpfr::mpreal scalingFactor = 2097152;

  testLatticeBasedQuantization({0.0, 0.24, 0.4, 0.68, 0.84, 1.0}, {1.0, 0.0, 1.0},
        {1.0, 10.0, 1.0}, degree, scalingFactor);

}


TEST(quantization_test, E35_8) {

  std::size_t degree = 17;
  mpfr::mpreal scalingFactor = 128;

  testLatticeBasedQuantization({0.02, 0.42, 0.52, 0.98}, {1.0, 0.0},
        {1.0, 1.0}, degree, scalingFactor);
}


TEST(quantization_test, E45_8) {

  std::size_t degree = 22;
  mpfr::mpreal scalingFactor = 128;

  testLatticeBasedQuantization({0.02, 0.42, 0.52, 0.98}, {1.0, 0.0},
        {1.0, 1.0}, degree, scalingFactor);
}

TEST(quantization_test, E125_21) {

  std::size_t degree = 62;
  mpfr::mpreal scalingFactor = 1048576;

  testLatticeBasedQuantization({0.02, 0.42, 0.52, 0.98}, {1.0, 0.0},
        {1.0, 1.0}, degree, scalingFactor);
}
