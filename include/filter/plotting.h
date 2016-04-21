#ifndef PLOTTING_H
#define PLOTTING_H

#include "util.h"
#include "conv.h"
#include "band.h"
#include "eigenvalue.h"
#include "barycentric.h"
#include "pm.h"
#include "cheby.h"
#include "fpminimax.h"

void plotPolyFixedData(std::string& filename, std::vector<mpfr::mpreal>& a,
        mpfr::mpreal& start, mpfr::mpreal& stop, std::vector<mpfr::mpreal>& points,
        mpfr::mpreal& position, mp_prec_t prec = 100ul);

void plotPolyDynamicData(std::string& filename, std::vector<mpfr::mpreal>& a,
        mpfr::mpreal& start, mpfr::mpreal& stop, std::vector<mpfr::mpreal>& points,
        mp_prec_t prec = 100ul);

void plotPolys(std::string& filename, std::vector<std::vector<mpfr::mpreal>>& a,
        mpfr::mpreal& start, mpfr::mpreal& stop, mp_prec_t prec = 100ul);

void plotAll(std::string& filename, std::vector<std::vector<mpfr::mpreal>>& a,
        mpfr::mpreal& start, mpfr::mpreal& stop, std::vector<mpfr::mpreal>& points,
        mp_prec_t prec = 100ul);


#endif
