#include "filter/barycentric.h"

void barycentricWeights(std::vector<mpfr::mpreal> &w,
                        std::vector<mpfr::mpreal> &x, mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  std::size_t step = (x.size() - 2) / 15 + 1;
  mpreal one = 1u;
  for (std::size_t i = 0u; i < x.size(); ++i) {
    mpreal denom = 1.0;
    mpreal xi = x[i];
    for (std::size_t j = 0u; j < step; ++j) {
      for (std::size_t k = j; k < x.size(); k += step)
        if (k != i)
          denom *= ((xi - x[k]) << 1);
    }
    w[i] = one / denom;
  }

  mpreal::set_default_prec(prevPrec);
}

void computeIdealResponseAndWeight(mpfr::mpreal &D, mpfr::mpreal &W,
                                   const mpfr::mpreal &xVal,
                                   std::vector<Band> &bands) {
  for (auto &it : bands) {
    if (xVal >= it.start && xVal <= it.stop) {

      D = it.amplitude(it.space, xVal);
      W = it.weight(it.space, xVal);
      return;
    }
  }
}

void computeDelta(mpfr::mpreal &delta, std::vector<mpfr::mpreal> &x,
                  std::vector<Band> &bands, mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  std::vector<mpfr::mpreal> w(x.size());
  barycentricWeights(w, x);

  mpfr::mpreal num, denom, D, W, buffer;
  num = denom = D = W = 0;
  for (std::size_t i = 0u; i < w.size(); ++i) {
    computeIdealResponseAndWeight(D, W, x[i], bands);
    buffer = w[i];
    num = fma(buffer, D, num);
    buffer = w[i] / W;
    if (i % 2 == 0)
      buffer = -buffer;
    denom += buffer;
  }

  delta = num / denom;

  mpreal::set_default_prec(prevPrec);
}

void computeDelta(mpfr::mpreal &delta, std::vector<mpfr::mpreal> &w,
                  std::vector<mpfr::mpreal> &x, std::vector<Band> &bands,
                  mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  mpfr::mpreal num, denom, D, W, buffer;
  num = denom = D = W = 0;
  for (std::size_t i = 0u; i < w.size(); ++i) {
    computeIdealResponseAndWeight(D, W, x[i], bands);
    buffer = w[i];
    num = fma(buffer, D, num);
    buffer = w[i] / W;
    if (i % 2 == 0)
      buffer = -buffer;
    denom += buffer;
  }
  delta = num / denom;
  mpreal::set_default_prec(prevPrec);
}

void computeC(std::vector<mpfr::mpreal> &C, mpfr::mpreal &delta,
              std::vector<mpfr::mpreal> &omega, std::vector<Band> &bands,
              mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  mpfr::mpreal D, W;
  D = W = 0;
  for (std::size_t i = 0u; i < omega.size(); ++i) {
    computeIdealResponseAndWeight(D, W, omega[i], bands);
    if (i % 2 != 0)
      W = -W;
    C[i] = D + (delta / W);
  }
  mpreal::set_default_prec(prevPrec);
}

void computeApprox(mpfr::mpreal &Pc, const mpfr::mpreal &omega,
                   std::vector<mpfr::mpreal> &x, std::vector<mpfr::mpreal> &C,
                   std::vector<mpfr::mpreal> &w, mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  mpfr::mpreal num, denom;
  mpfr::mpreal buff;
  num = denom = 0;

  Pc = omega;
  std::size_t r = x.size();
  for (std::size_t i = 0u; i < r; ++i) {
    if (Pc == x[i]) {
      Pc = C[i];
      mpreal::set_default_prec(prevPrec);
      return;
    }
    buff = w[i] / (Pc - x[i]);
    num = fma(buff, C[i], num);
    denom += buff;
  }
  Pc = num / denom;
  mpreal::set_default_prec(prevPrec);
}

void computeError(mpfr::mpreal &error, const mpfr::mpreal &xVal,
                  mpfr::mpreal &delta, std::vector<mpfr::mpreal> &x,
                  std::vector<mpfr::mpreal> &C, std::vector<mpfr::mpreal> &w,
                  std::vector<Band> &bands, mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  for (std::size_t i = 0u; i < x.size(); ++i) {
    if (xVal == x[i]) {
      if (i % 2 == 0)
        error = delta;
      else
        error = -delta;
      mpreal::set_default_prec(prevPrec);
      return;
    }
  }

  mpfr::mpreal D, W;
  D = W = 0;
  computeIdealResponseAndWeight(D, W, xVal, bands);
  computeApprox(error, xVal, x, C, w);
  error -= D;
  error *= W;
  mpreal::set_default_prec(prevPrec);
}

void barycentricWeights(std::vector<double>& w,
        std::vector<double>& x)
{
    if(x.size() > 500u)
    {
        for(std::size_t i = 0u; i < x.size(); ++i)
        {
            double one = 1;
            double denom = 0.0;
            double xi = x[i];
            for(std::size_t j = 0u; j < x.size(); ++j)
            {
                if (j != i) {
                    denom += log(((xi - x[j] > 0) ? (xi - x[j]) : (x[j] - xi)));
                    one *= ((xi - x[j] > 0) ? 1 : -1);
                }
            }
            w[i] = one / exp(denom + log(2.0)* (x.size() - 1));
        }
    }
    else
    {
        std::size_t step = (x.size() - 2) / 15 + 1;
        double one = 1u;
        for(std::size_t i = 0u; i < x.size(); ++i)
        {
            double denom = 1.0;
            double xi = x[i];
            for(std::size_t j = 0u; j < step; ++j)
            {
                for(std::size_t k = j; k < x.size(); k += step)
                    if (k != i)
                        denom *= ((xi - x[k]) * 2);
            }
            w[i] = one / denom;
        }
    }
}


void computeIdealResponseAndWeight(double &D, double &W,
        const double &xVal, std::vector<BandD> &bands)
{
    for (auto &it : bands) {
        if (xVal >= it.start && xVal <= it.stop) {
            D = it.amplitude(it.space, xVal);
            W = it.weight(it.space, xVal);
            return;
        }
    }
}

void computeDelta(double &delta, std::vector<double>& x,
        std::vector<BandD> &bands)
{
    std::vector<double> w(x.size());
    barycentricWeights(w, x);

    double num, denom, D, W, buffer;
    num = denom = D = W = 0;
    for (std::size_t i = 0u; i < w.size(); ++i)
    {
        computeIdealResponseAndWeight(D, W, x[i], bands);
        buffer = w[i];
        num += buffer * D;
        buffer = w[i] / W;
        if (i % 2 == 0)
            buffer = -buffer;
        denom += buffer;
    }

    delta = num / denom;
}

void computeDelta(double &delta, std::vector<double>& w,
        std::vector<double>& x, std::vector<BandD> &bands)
{
    double num, denom, D, W, buffer;
    num = denom = D = W = 0;
    for (std::size_t i = 0u; i < w.size(); ++i)
    {
        computeIdealResponseAndWeight(D, W, x[i], bands);
        buffer = w[i];
        num += buffer * D;
        buffer = w[i] / W;
        if (i % 2 == 0)
            buffer = -buffer;
        denom += buffer;
    }
    delta = num / denom;
}


void computeC(std::vector<double> &C, double &delta,
        std::vector<double> &omega, std::vector<BandD> &bands)
{
    double D, W;
    D = W = 0;
    for (std::size_t i = 0u; i < omega.size(); ++i)
    {
        computeIdealResponseAndWeight(D, W, omega[i], bands);
        if (i % 2 != 0)
            W = -W;
        C[i] = D + (delta / W);
    }
}

void computeApprox(double &Pc, const double &omega,
        std::vector<double> &x, std::vector<double> &C,
        std::vector<double> &w)
{
    double num, denom;
    double buff;
    num = denom = 0;

    Pc = omega;
    std::size_t r = x.size();
    for (std::size_t i = 0u; i < r; ++i)
    {
        if (Pc == x[i]) {
            Pc = C[i];
            return;
        }
        buff = w[i] / (Pc - x[i]);
        num += buff * C[i];
        denom += buff;
    }
    Pc = num / denom;
}

void computeError(double &error, const double &xVal,
        double &delta, std::vector<double> &x,
        std::vector<double> &C, std::vector<double> &w,
        std::vector<BandD> &bands)
{
    for (std::size_t i = 0u; i < x.size(); ++i)
    {
        if (xVal == x[i]) {
            if (i % 2 == 0)
                error = delta;
            else
                error = -delta;
            return;
        }
    }

    double D, W;
    D = W = 0;
    computeIdealResponseAndWeight(D, W, xVal, bands);
    computeApprox(error, xVal, x, C, w);
    error -= D;
    error *= W;
}
