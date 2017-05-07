#include "filter/grid.h"

void generateGrid(std::vector<GridPoint> &grid, std::size_t degree,
                  std::vector<Band> &freqBands, std::size_t density,
                  mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  mpfr::mpreal increment = mpfr::const_pi();
  increment /= (degree * density);
  int bandIndex = 0;
  GridPoint buffer;
  mpfr::mpreal omega, x, D, W;
  while (bandIndex < freqBands.size()) {
    omega = freqBands[bandIndex].start;
    x = mpfr::cos(omega);
    computeIdealResponseAndWeight(D, W, omega, freqBands);
    buffer.omega = omega.toDouble();
    buffer.x = x.toDouble();
    buffer.D = D.toDouble();
    buffer.W = W.toDouble();
    grid.push_back(buffer);
    while (grid.back().omega <= freqBands[bandIndex].stop) {
      omega += increment;
      x = mpfr::cos(omega);
      computeIdealResponseAndWeight(D, W, omega, freqBands);
      buffer.omega = omega.toDouble();
      buffer.x = x.toDouble();
      buffer.D = D.toDouble();
      buffer.W = W.toDouble();
      grid.push_back(buffer);
    }
    omega = freqBands[bandIndex].stop;
    x = mpfr::cos(omega);
    grid[grid.size() - 1].omega = omega.toDouble();
    grid[grid.size() - 1].x = x.toDouble();
    computeIdealResponseAndWeight(D, W, omega, freqBands);
    grid[grid.size() - 1u].D = D.toDouble();
    grid[grid.size() - 1u].W = W.toDouble();
    ++bandIndex;
  }

  mpreal::set_default_prec(prevPrec);
}


void getError(double &error, GridPoint &p,
              std::vector<double> &a) {
  evaluateClenshaw(error, a, p.x);
  //std::cout << p.x << "\t" << error << std::endl;
  error = p.W * (p.D - error);
}

void computeDenseNorm(double &normValue,
                      std::vector<double> &bandNorms,
                      std::vector<Band> &chebyBands,
                      std::vector<GridPoint> &grid,
                      std::vector<double> &a) {

  normValue = 0;
  double currentError;
  for (auto &it : bandNorms)
    it = 0;
  for (std::size_t i = 0u; i < grid.size(); ++i) {
    getError(currentError, grid[i], a);
    currentError = fabs(currentError);
    if (currentError > normValue)
      normValue = currentError;
    for (std::size_t j = 0u; j < chebyBands.size(); ++j) {
      if (grid[i].x >= chebyBands[j].start.toDouble(GMP_RNDD)
        && grid[i].x <= chebyBands[j].stop.toDouble(GMP_RNDU))
        if (bandNorms[j] < currentError)
          bandNorms[j] = currentError;
    }
  }
}
