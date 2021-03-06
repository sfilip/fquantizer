#include "filter/cheby.h"

void applyCos(std::vector<mpfr::mpreal> &out,
              std::vector<mpfr::mpreal> const &in) {
  for (std::size_t i = 0u; i < in.size(); ++i)
    out[i] = mpfr::cos(in[i]);
}

void changeOfVariable(std::vector<mpfr::mpreal> &out,
                      std::vector<mpfr::mpreal> const &in, mpfr::mpreal &a,
                      mpfr::mpreal &b) {
  using mpfr::mpreal;
  for (std::size_t i = 0u; i < in.size(); ++i)
    out[i] = fma((b - a) / 2, in[i], (b + a) / 2);
}

void evaluateClenshaw(mpfr::mpreal &result, std::vector<mpfr::mpreal> &p,
                      mpfr::mpreal &x, mpfr::mpreal &a, mpfr::mpreal &b,
                      mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpreal bn1, bn2, bn;
  mpreal buffer;

  bn1 = 0;
  bn2 = 0;

  // compute the value of (2*x - b - a)/(b - a) in the temporary
  // variable buffer
  buffer = (x * 2 - b - a) / (b - a);

  int n = (int)p.size() - 1;
  for (int k = n; k >= 0; --k) {
    bn = buffer * 2;
    bn = bn * bn1 - bn2 + p[k];
    // update values
    bn2 = bn1;
    bn1 = bn;
  }

  // set the value for the result (line 8 which outputs the value
  // of the CI at x)
  result = bn1 - buffer * bn2;
  mpreal::set_default_prec(prevPrec);
}

void evaluateClenshaw(mpfr::mpreal &result, std::vector<mpfr::mpreal> &p,
                      mpfr::mpreal &x, mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpreal bn1, bn2, bn;

  int n = (int)p.size() - 1;
  bn2 = 0;
  bn1 = p[n];
  for (int k = n - 1; k >= 1; --k) {
    bn = x * 2;
    bn = bn * bn1 - bn2 + p[k];
    // update values
    bn2 = bn1;
    bn1 = bn;
  }

  result = x * bn1 - bn2 + p[0];
  mpreal::set_default_prec(prevPrec);
}

void evaluateClenshaw2ndKind(mpfr::mpreal &result, std::vector<mpfr::mpreal> &p,
                             mpfr::mpreal &x, mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpreal bn1, bn2, bn;

  int n = (int)p.size() - 1;
  bn2 = 0;
  bn1 = p[n];
  for (int k = n - 1; k >= 1; --k) {
    bn = x * 2;
    bn = bn * bn1 - bn2 + p[k];
    // update values
    bn2 = bn1;
    bn1 = bn;
  }

  result = (x << 1) * bn1 - bn2 + p[0];
  mpreal::set_default_prec(prevPrec);
}

void generateEquidistantNodes(std::vector<mpfr::mpreal> &v, std::size_t n,
                              mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpreal pi = mpfr::const_pi(prec);

  // store the points in the vector v as v[i] = i * pi / n
  for (std::size_t i = 0; i <= n; ++i) {
    v[i] = pi * i;
    v[i] /= n;
  }
  mpreal::set_default_prec(prevPrec);
}

void generateChebyshevPoints(std::vector<mpfr::mpreal> &x, std::size_t n,
                             mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  mpreal pi = mpfr::const_pi(prec);

  // n is the number of points - 1
  x.reserve(n + 1u);
  if (n > 0u) {
    for (int k = n; k >= -n; k -= 2)
      x.push_back(sin(pi * k / (n * 2)));
  } else {
    x.push_back(mpfr::mpreal(0));
  }
  mpreal::set_default_prec(prevPrec);
}

void generateChebyshevCoefficients(std::vector<mpfr::mpreal> &c,
                                   std::vector<mpfr::mpreal> &fv, std::size_t n,
                                   mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  std::vector<mpreal> v(n + 1);
  generateEquidistantNodes(v, n, prec);

  mpreal buffer;

  // halve the first and last coefficients
  mpfr::mpreal oldValue1 = fv[0];
  mpfr::mpreal oldValue2 = fv[n];
  fv[0] /= 2;
  fv[n] /= 2;

  for (std::size_t i = 0u; i <= n; ++i) {
    buffer = mpfr::cos(v[i]); // compute the actual value at the Chebyshev
                              // node cos(i * pi / n)

    evaluateClenshaw(c[i], fv, buffer,
                     prec); // evaluate the current coefficient
                            // using Clenshaw
    if (i == 0u || i == n) {
      c[i] /= n;
    } else {
      c[i] <<= 1;
      c[i] /= n;
    }
  }
  fv[0] = oldValue1;
  fv[n] = oldValue2;

  mpreal::set_default_prec(prevPrec);
}

// function that generates the coefficients of the derivative of a given CI
void derivativeCoefficients1stKind(std::vector<mpfr::mpreal> &derivC,
                                   std::vector<mpfr::mpreal> &c) {
  using mpfr::mpreal;
  int n = c.size() - 2;
  derivC[n] = c[n + 1] * (2 * (n + 1));
  derivC[n - 1] = c[n] * (2 * n);
  for (int i = n - 2; i >= 0; --i) {
    derivC[i] = 2 * (i + 1);
    derivC[i] = fma(derivC[i], c[i + 1], derivC[i + 2]);
  }
  derivC[0] >>= 1;
}

// use the formula (T_n(x))' = n * U_{n-1}(x)
void derivativeCoefficients2ndKind(std::vector<mpfr::mpreal> &derivC,
                                   std::vector<mpfr::mpreal> &c) {
  std::size_t n = c.size() - 1;
  for (std::size_t i = n; i > 0u; --i)
    derivC[i - 1] = c[i] * i;
}

void applyCos(std::vector<double>& out,
        std::vector<double> const& in)
{
    for (std::size_t i{0u}; i < in.size(); ++i)
        out[i] = cosl(in[i]);
}

void changeOfVariable(std::vector<double>& out,
        std::vector<double> const& in,
        double& a, double& b)
{
    for (std::size_t i{0u}; i < in.size(); ++i)
        out[i] = (b + a) / 2 + in[i] * (b - a) / 2;
}

void evaluateClenshaw(double &result, std::vector<double> &p,
        double &x, double &a, double &b)
{
    double bn1, bn2, bn;
    double buffer;

    bn1 = 0;
    bn2 = 0;

    // compute the value of (2*x - b - a)/(b - a) in the temporary
    // variable buffer
    buffer = (x * 2 - b - a) / (b - a);

    int n = (int)p.size() - 1;
    for(int k{n}; k >= 0; --k) {
        bn = buffer * 2;
        bn = bn * bn1 - bn2 + p[k];
        // update values
        bn2 = bn1;
        bn1 = bn;
    }

    // set the value for the result
    // (i.e. the CI value at x)
    result = bn1 - buffer * bn2;
}

void evaluateClenshaw(double &result, std::vector<double> &p,
                            double &x)
{
    double bn1, bn2, bn;

    int n = (int)p.size() - 1;
    bn2 = 0;
    bn1 = p[n];
    for(int k{n - 1}; k >= 1; --k) {
        bn = x * 2;
        bn = bn * bn1 - bn2 + p[k];
        // update values
        bn2 = bn1;
        bn1 = bn;
    }

    result = x * bn1 - bn2 + p[0];
}

void evaluateClenshaw2ndKind(double &result, std::vector<double> &p,
                            double &x)
{
    double bn1, bn2, bn;

    int n = (int)p.size() - 1;
    bn2 = 0;
    bn1 = p[n];
    for(int k{n - 1}; k >= 1; --k) {
        bn = x * 2;
        bn = bn * bn1 - bn2 + p[k];
        // update values
        bn2 = bn1;
        bn1 = bn;
    }

    result = (x * 2) * bn1 - bn2 + p[0];
}


void generateEquidistantNodes(std::vector<double>& v, std::size_t n)
{
    // store the points in the vector v as v[i] = i * pi / n
    for(std::size_t i{0u}; i <= n; ++i) {
        v[i] = M_PI * i;
        v[i] /= n;
    }
}

void generateChebyshevPoints(std::vector<double>& x, std::size_t n)
{
    // n is the number of points - 1
    x.reserve(n + 1u);
    if(n > 0u)
    {
        for(int k = n; k >= -n; k -= 2)
            x.push_back(sin(M_PI * k / (n * 2)));
    }
    else
    {
        x.push_back(0);
    }
}

// this function computes the values of the coefficients of the CI when
// Chebyshev nodes of the second kind are used
void generateChebyshevCoefficients(std::vector<double>& c,
        std::vector<double>& fv, std::size_t n)
{
    std::vector<double> v(n + 1);
    generateEquidistantNodes(v, n);

    double buffer;

    // halve the first and last coefficients
    double oldValue1 = fv[0];
    double oldValue2 = fv[n];
    fv[0] /= 2;
    fv[n] /= 2;

    for(std::size_t i{0u}; i <= n; ++i) {
        buffer = cos(v[i]);         // compute the actual value at the Chebyshev
                                    // node cos(i * pi / n)

        evaluateClenshaw(c[i], fv, buffer);

        if(i == 0u || i == n) {
            c[i] /= n;
        } else {
            c[i] *= 2;
            c[i] /= n;
        }
    }
    fv[0] = oldValue1;
    fv[n] = oldValue2;

}

// function that generates the coefficients of the derivative of a given CI
void derivativeCoefficients1stKind(std::vector<double>& derivC,
                                        std::vector<double>& c)
{
    int n = c.size() - 2;
    derivC[n] = c[n + 1] * (2 * (n + 1));
    derivC[n - 1] = c[n] * (2 * n);
    for(int i{n - 2}; i >= 0; --i) {
        derivC[i] = 2 * (i + 1);
        derivC[i] = derivC[i] * c[i + 1] + derivC[i + 2];
    }
    derivC[0] /= 2;
}

// use the formula (T_n(x))' = n * U_{n-1}(x)
void derivativeCoefficients2ndKind(std::vector<double>& derivC,
        std::vector<double>& c)
{
    std::size_t n = c.size() - 1;
    for(std::size_t i{n}; i > 0u; --i)
        derivC[i - 1] = c[i] * i;
}