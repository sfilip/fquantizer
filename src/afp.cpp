#include "filter/afp.h"

void generateAFPMatrix(
    MatrixXq &A, std::size_t degree, std::vector<mpfr::mpreal> &meshPoints,
    std::function<mpfr::mpreal(mpfr::mpreal)> &weightFunction) {

  A.resize(degree + 1u, meshPoints.size());
  for (std::size_t i = 0u; i < meshPoints.size(); ++i) {
    mpfr::mpreal pointWeight = weightFunction(meshPoints[i]);
    A(0u, i) = 1;
    A(1u, i) = meshPoints[i];
    for (std::size_t j = 2u; j <= degree; ++j)
      A(j, i) = meshPoints[i] * A(j - 1u, i) * 2 - A(j - 2u, i);
    for (std::size_t j = 0u; j <= degree; ++j)
      A(j, i) *= pointWeight;
  }
}

// approximate Fekete points
void AFP(std::vector<mpfr::mpreal> &points, MatrixXq &A,
         std::vector<mpfr::mpreal> &meshPoints) {
  VectorXq b = VectorXq::Ones(A.rows());
  b(1) = 2;
  VectorXq y = A.colPivHouseholderQr().solve(b);

  for (std::size_t i = 0u; i < y.rows(); ++i)
    if (y(i) != 0.0)
      points.push_back(meshPoints[i]);
  std::sort(points.begin(), points.end(),
            [](const mpfr::mpreal &lhs, const mpfr::mpreal &rhs) {
              return lhs < rhs;
            });
}

void generateVandermondeMatrix(
    MatrixXq &Vm, std::vector<mpfr::mpreal> &grid,
    std::function<mpfr::mpreal(mpfr::mpreal)> &weightFunction,
    std::size_t degree, mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  for (std::size_t i = 0u; i < grid.size(); ++i) {
    Vm(i, 0) = 1.0;
    Vm(i, 1) = grid[i];
  }
  for (std::size_t j = 2u; j <= degree; ++j)
    for (std::size_t i = 0u; i < grid.size(); ++i)
      Vm(i, j) = Vm(i, j - 1) * (grid[i] << 1u) - Vm(i, j - 2);

  for (std::size_t j = 0u; j <= degree; ++j)
    for (std::size_t i = 0u; i < grid.size(); ++i)
      Vm(i, j) *= weightFunction(grid[i]);

  mpreal::set_default_prec(prevPrec);
}

void linspace(std::vector<mpfr::mpreal> &points, mpfr::mpreal &a,
              mpfr::mpreal &b, std::size_t N, mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpfr::mpreal step = (b - a) / (N - 1u);
  points.push_back(a);
  for (std::size_t i = 1u; i <= N - 2u; ++i)
    points.push_back(a + step * i);
  points.push_back(b);

  mpreal::set_default_prec(prevPrec);
}

void bandCount(std::vector<Band> &chebyBands, std::vector<mpfr::mpreal> &x) {
  for (auto &it : chebyBands)
    it.extremas = 0u;
  std::size_t bandIt = 0u;
  for (std::size_t i = 0u; i < x.size(); ++i) {
    while (bandIt < chebyBands.size() && chebyBands[bandIt].stop < x[i])
      bandIt++;
    ++chebyBands[bandIt].extremas;
  }
  std::size_t index = 0u;
  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    x[index] = chebyBands[i].start;
    x[index + chebyBands[i].extremas - 1u] = chebyBands[i].stop;
    index += chebyBands[i].extremas;
  }
}

void chebyMeshGeneration(std::vector<mpfr::mpreal> &chebyMesh,
                         std::vector<Band> &chebyBands, std::size_t degree,
                         mp_prec_t prec) {
  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    std::vector<mpfr::mpreal> bufferPoints(degree + 2u);
    generateEquidistantNodes(bufferPoints, degree + 1u, prec);
    applyCos(bufferPoints, bufferPoints);
    changeOfVariable(bufferPoints, bufferPoints, chebyBands[i].start,
                     chebyBands[i].stop);
    for (auto &it : bufferPoints)
      chebyMesh.push_back(it);
  }
}


void generateAFPMatrix(
    MatrixXd &A, std::size_t degree, std::vector<double> &meshPoints,
    std::function<double(double)> &weightFunction) {

  A.resize(degree + 1u, meshPoints.size());
  for (std::size_t i = 0u; i < meshPoints.size(); ++i) {
    double pointWeight = weightFunction(meshPoints[i]);
    A(0u, i) = 1;
    A(1u, i) = meshPoints[i];
    for (std::size_t j = 2u; j <= degree; ++j)
      A(j, i) = meshPoints[i] * A(j - 1u, i) * 2 - A(j - 2u, i);
    for (std::size_t j = 0u; j <= degree; ++j)
      A(j, i) *= pointWeight;
  }
}

// approximate Fekete points
void AFP(std::vector<double> &points, MatrixXd &A,
         std::vector<double> &meshPoints) {
  VectorXd b = VectorXd::Ones(A.rows());
  b(1) = 2;
  VectorXd y = A.colPivHouseholderQr().solve(b);

  for (std::size_t i = 0u; i < y.rows(); ++i)
    if (y(i) != 0.0)
      points.push_back(meshPoints[i]);
  std::sort(points.begin(), points.end(),
            [](const double &lhs, const double &rhs) {
              return lhs < rhs;
            });
}

void generateVandermondeMatrix(
    MatrixXd &Vm, std::vector<double> &grid,
    std::function<double(double)> &weightFunction,
    std::size_t degree) {

  for (std::size_t i = 0u; i < grid.size(); ++i) {
    Vm(i, 0) = 1.0;
    Vm(i, 1) = grid[i];
  }
  for (std::size_t j = 2u; j <= degree; ++j)
    for (std::size_t i = 0u; i < grid.size(); ++i)
      Vm(i, j) = Vm(i, j - 1) * (grid[i] * 2) - Vm(i, j - 2);

  for (std::size_t j = 0u; j <= degree; ++j)
    for (std::size_t i = 0u; i < grid.size(); ++i)
      Vm(i, j) *= weightFunction(grid[i]);
}

void linspace(std::vector<double> &points, double &a,
              double &b, std::size_t N) {

  double step = (b - a) / (N - 1u);
  points.push_back(a);
  for (std::size_t i = 1u; i <= N - 2u; ++i)
    points.push_back(a + step * i);
  points.push_back(b);
}

void bandCount(std::vector<BandD> &chebyBands, std::vector<double> &x) {
  for (auto &it : chebyBands)
    it.extremas = 0u;
  std::size_t bandIt = 0u;
  for (std::size_t i = 0u; i < x.size(); ++i) {
    while (bandIt < chebyBands.size() && chebyBands[bandIt].stop < x[i])
      bandIt++;
    ++chebyBands[bandIt].extremas;
  }
  std::size_t index = 0u;
  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    x[index] = chebyBands[i].start;
    x[index + chebyBands[i].extremas - 1u] = chebyBands[i].stop;
    index += chebyBands[i].extremas;
  }
}

void chebyMeshGeneration(std::vector<double> &chebyMesh,
                         std::vector<BandD> &chebyBands, std::size_t degree) {
  for (std::size_t i = 0u; i < chebyBands.size(); ++i) {
    std::vector<double> bufferPoints(degree + 2u);
    generateEquidistantNodes(bufferPoints, degree + 1u);
    applyCos(bufferPoints, bufferPoints);
    changeOfVariable(bufferPoints, bufferPoints, chebyBands[i].start,
                     chebyBands[i].stop);
    for (auto &it : bufferPoints)
      chebyMesh.push_back(it);
  }
}
