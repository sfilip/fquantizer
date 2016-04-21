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
  while (bandIndex < freqBands.size()) {
    buffer.omega = freqBands[bandIndex].start;
    buffer.x = mpfr::cos(buffer.omega);
    computeIdealResponseAndWeight(buffer.D, buffer.W, buffer.omega, freqBands);
    grid.push_back(buffer);
    while (grid.back().omega <= freqBands[bandIndex].stop) {
      buffer.omega += increment;
      buffer.x = mpfr::cos(buffer.omega);
      computeIdealResponseAndWeight(buffer.D, buffer.W, buffer.omega,
                                    freqBands);
      grid.push_back(buffer);
    }
    grid[grid.size() - 1].omega = freqBands[bandIndex].stop;
    grid[grid.size() - 1].x = mpfr::cos(grid[grid.size() - 1].omega);
    computeIdealResponseAndWeight(grid[grid.size() - 1u].D,
                                  grid[grid.size() - 1u].W,
                                  grid[grid.size() - 1u].omega, freqBands);
    ++bandIndex;
  }

  mpreal::set_default_prec(prevPrec);
}

void getError(mpfr::mpreal &error, std::vector<Band> &chebyBands, GridPoint &p,
              std::vector<mpfr::mpreal> &a, mp_prec_t prec) {
  evaluateClenshaw(error, a, p.x, prec);
  error = p.W * (p.D - error);
}

void computeDenseNorm(mpfr::mpreal &normValue,
                      std::vector<mpfr::mpreal> &bandNorms,
                      std::vector<Band> &chebyBands,
                      std::vector<GridPoint> &grid,
                      std::vector<mpfr::mpreal> &a, mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  normValue = 0;
  mpfr::mpreal currentError;
  for (auto &it : bandNorms)
    it = 0;
  for (std::size_t i = 0u; i < grid.size(); ++i) {
    getError(currentError, chebyBands, grid[i], a, prec);
    currentError = mpfr::abs(currentError);
    if (currentError > normValue)
      normValue = currentError;
    for (std::size_t j = 0u; j < chebyBands.size(); ++j) {
      if (grid[i].x >= chebyBands[j].start && grid[i].x <= chebyBands[j].stop)
        if (bandNorms[j] < currentError)
          bandNorms[j] = currentError;
    }
  }

  mpreal::set_default_prec(prevPrec);
}
