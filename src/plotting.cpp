#include "filter/plotting.h"
#include <cstdlib>
#include <fstream>

void plotPolyFixedData(std::string &filename, std::vector<mpfr::mpreal> &a,
                       mpfr::mpreal &start, mpfr::mpreal &stop,
                       std::vector<mpfr::mpreal> &points,
                       mpfr::mpreal &position, mp_prec_t prec) {
  using namespace mpfr;
  std::stringstream fullFilename, datFilename;
  fullFilename << filename << "_0.sollya";
  datFilename << filename << "_0.dat";

  std::ofstream output;
  output.open(fullFilename.str().c_str());
  output << "I=[" << start.toString("%.30RNf") << "*pi,"
         << stop.toString("%.30RNf") << "*pi];\n"
         << "n=" << a.size() - 1 << ";\n"
         << "a=[||];\n"
         << "figureOut=\"" << filename << "_0"
         << "\";\n"
         << "//-----------------------------------\n";
  for (std::size_t i{0u}; i < a.size(); ++i)
    output << "a[" << i << "]=" << a[i].toString("%.30RNf") << ";\n";
  output << "//-----------------------------------\n"
         << "prec=500;\n"
         << "points=1000;\n"
         << "ChebyPolys=[|1,cos(x)|];\n"
         << "for i from 1 to n do {\n"
         << "ChebyPolys[i+1]=cos((i+1)*x);\n"
         << "};\n"
         << "P=a[0];\n"
         << "for i from 1 to n do {\n"
         << "P=P+a[i]*ChebyPolys[i];\n"
         << "};\n"
         << "plot(P, I, postscriptfile, figureOut);\n";
  output.close();

  std::string sollyaCommand = "sollya ";
  sollyaCommand += fullFilename.str();
  system(sollyaCommand.c_str());

  std::stringstream pointDataFile;
  pointDataFile << filename << "_1.dat";
  std::ofstream dataOutput;
  dataOutput.open(pointDataFile.str().c_str());
  for (std::size_t i{0u}; i < points.size(); ++i) {
    mpfr::mpreal pt, ptVal;
    pt = mpfr::acos(points[i]);
    evaluateClenshaw(ptVal, a, points[i], prec);
    dataOutput << pt.toString("%.30RNf") << "\t" << position << std::endl;
  }
  dataOutput.close();

  std::stringstream gnuplotFile;
  gnuplotFile << filename << "_f.p";
  std::stringstream epsFilename;
  epsFilename << filename << "_f.eps";
  output.open(gnuplotFile.str().c_str());
  mpfr::mpreal startPlot = start * mpfr::const_pi();
  mpfr::mpreal stopPlot = stop * mpfr::const_pi();

  output << "set terminal postscript eps color\n"
         << R"(set out ")" << epsFilename.str() << R"(")" << std::endl
         << R"(set format x "%g")" << std::endl
         << R"(set format y "%g")" << std::endl
         << "set xrange [" << startPlot.toString("%.30RNf") << ":"
         << stopPlot.toString("%.30RNf") << "]\n"
         << R"(plot ")" << datFilename.str()
         << R"(" using 1:2 with lines t "", \)" << std::endl
         << "\t"
         << R"(")" << pointDataFile.str() << R"(" using 1:2 with points t "")"
         << std::endl;

  std::string gnuplotCommand = "gnuplot ";
  gnuplotCommand += gnuplotFile.str();
  system(gnuplotCommand.c_str());
}

void plotPolyDynamicData(std::string &filename, std::vector<mpfr::mpreal> &a,
                         mpfr::mpreal &start, mpfr::mpreal &stop,
                         std::vector<mpfr::mpreal> &points, mp_prec_t prec) {
  using namespace mpfr;
  std::stringstream datFilename;
  datFilename << filename << "_0.dat";

  std::ofstream output;
  output.open(datFilename.str().c_str());
  mpfr::mpreal startValue = mpfr::cos(stop * mpfr::const_pi());
  mpfr::mpreal stopValue = mpfr::cos(start * mpfr::const_pi());
  mpfr::mpreal width = stopValue - startValue;
  mpfr::mpreal buffer, bufferValue;
  std::size_t pointCount = 5000u;
  for (std::size_t i = 0u; i < pointCount; ++i) {
    buffer = startValue + (width * i) / pointCount;
    output << buffer.toString("%.80RNf") << "\t";
    evaluateClenshaw(bufferValue, a, buffer, prec);
    output << bufferValue.toString("%.80RNf") << std::endl;
  }
  output << stopValue.toString("%.80RNf") << "\t";
  evaluateClenshaw(bufferValue, a, stopValue, prec);
  output << bufferValue.toString("%.80RNf") << std::endl;

  output.close();

  std::stringstream pointDataFile;
  pointDataFile << filename << "_1.dat";
  std::ofstream dataOutput;
  dataOutput.open(pointDataFile.str().c_str());
  for (std::size_t i{0u}; i < points.size(); ++i) {
    mpfr::mpreal ptVal;
    evaluateClenshaw(ptVal, a, points[i], prec);
    dataOutput << points[i].toString("%.80RNf") << "\t"
               << ptVal.toString("%.80RNf") << std::endl;
  }
  dataOutput.close();

  std::stringstream gnuplotFile;
  gnuplotFile << filename << "_f.p";
  std::stringstream epsFilename;
  epsFilename << filename << "_f.eps";
  output.open(gnuplotFile.str().c_str());

  output << "set terminal postscript eps color\n"
         << R"(set out ")" << epsFilename.str() << R"(")" << std::endl
         << R"(set format x "%g")" << std::endl
         << R"(set format y "%g")" << std::endl
         << "set xrange [" << startValue.toString("%.80RNf") << ":"
         << stopValue.toString("%.80RNf") << "]\n"
         << R"(plot ")" << datFilename.str()
         << R"(" using 1:2 with lines t "", \)" << std::endl
         << "\t"
         << R"(")" << pointDataFile.str() << R"(" using 1:2 with points t "")"
         << std::endl;

  std::string gnuplotCommand = "gnuplot ";
  gnuplotCommand += gnuplotFile.str();
  system(gnuplotCommand.c_str());
}

void plotPolys(std::string &filename, std::vector<std::vector<mpfr::mpreal>> &a,
               mpfr::mpreal &start, mpfr::mpreal &stop, mp_prec_t prec) {
  using namespace mpfr;
  std::vector<std::stringstream> fullFilename(a.size());
  std::vector<std::stringstream> datFilename(a.size());
  for (std::size_t i{0u}; i < a.size(); ++i) {
    fullFilename[i] << filename << "_" << (i + 1u) << ".sollya";
    datFilename[i] << filename << "_" << (i + 1u) << ".dat";

    std::ofstream output;
    output.open(fullFilename[i].str().c_str());
    output << "I=[" << start.toString("%.30RNf") << "*pi,"
           << stop.toString("%.30RNf") << "*pi];\n"
           << "n=" << a[i].size() - 1 << ";\n"
           << "a=[||];\n"
           << "figureOut=\"" << filename << "_" << (i + 1) << "\";\n"
           << "//-----------------------------------\n";
    for (std::size_t j{0u}; j < a[i].size(); ++j)
      output << "a[" << j << "]=" << a[i][j].toString("%.30RNf") << ";\n";
    output << "//-----------------------------------\n"
           << "prec=500;\n"
           << "points=2000;\n"
           << "ChebyPolys=[|1,cos(x)|];\n"
           << "for i from 1 to n do {\n"
           << "ChebyPolys[i+1]=cos((i+1)*x);\n"
           << "};\n"
           << "P=a[0];\n"
           << "for i from 1 to n do {\n"
           << "P=P+a[i]*ChebyPolys[i];\n"
           << "};\n"
           << "plot(P, I, postscriptfile, figureOut);\n";
    output.close();

    std::string sollyaCommand = "sollya ";
    sollyaCommand += fullFilename[i].str();
    system(sollyaCommand.c_str());
  }

  std::stringstream gnuplotFile;
  gnuplotFile << filename << "_f.p";
  std::stringstream epsFilename;
  epsFilename << filename << "_f.eps";
  std::ofstream output;
  output.open(gnuplotFile.str().c_str());
  mpfr::mpreal startPlot = start * mpfr::const_pi();
  mpfr::mpreal stopPlot = stop * mpfr::const_pi();

  output << "set terminal postscript eps color\n"
         << R"(set out ")" << epsFilename.str() << R"(")" << std::endl
         << R"(set format x "%g")" << std::endl
         << R"(set format y "%g")" << std::endl
         << "set xrange [" << startPlot.toString("%.30RNf") << ":"
         << stopPlot.toString("%.30RNf") << "]\n"
         << R"(plot ")" << datFilename[0].str()
         << R"(" using 1:2 with lines t "", \)" << std::endl;

  for (std::size_t i{1u}; i < a.size() - 1; ++i) {
    output << "\t"
           << R"(")" << datFilename[i].str()
           << R"(" using 1:2 with lines t "", \)" << std::endl;
  }
  output << "\t"
         << R"(")" << datFilename[a.size() - 1].str()
         << R"(" using 1:2 with lines t "")" << std::endl;
  output.close();

  std::string gnuplotCommand = "gnuplot ";
  gnuplotCommand += gnuplotFile.str();
  system(gnuplotCommand.c_str());
}

void plotAll(std::string &filename, std::vector<std::vector<mpfr::mpreal>> &a,
             mpfr::mpreal &start, mpfr::mpreal &stop,
             std::vector<mpfr::mpreal> &points, mp_prec_t prec) {
  using namespace mpfr;
  std::vector<std::stringstream> fullFilename(a.size());
  std::vector<std::stringstream> datFilename(a.size());
  for (std::size_t i{0u}; i < a.size(); ++i) {
    fullFilename[i] << filename << "_" << (i + 1u) << ".sollya";
    datFilename[i] << filename << "_" << (i + 1u) << ".dat";

    std::ofstream output;
    output.open(fullFilename[i].str().c_str());
    output << "I=[" << start.toString("%.30RNf") << "*pi,"
           << stop.toString("%.30RNf") << "*pi];\n"
           << "n=" << a[i].size() - 1 << ";\n"
           << "a=[||];\n"
           << "figureOut=\"" << filename << "_" << (i + 1) << "\";\n"
           << "//-----------------------------------\n";
    for (std::size_t j{0u}; j < a[i].size(); ++j)
      output << "a[" << j << "]=" << a[i][j].toString("%.30RNf") << ";\n";
    output << "//-----------------------------------\n"
           << "prec=500;\n"
           << "ChebyPolys=[|1,cos(x)|];\n"
           << "for i from 1 to n do {\n"
           << "ChebyPolys[i+1]=cos((i+1)*x);\n"
           << "};\n"
           << "P=a[0];\n"
           << "for i from 1 to n do {\n"
           << "P=P+a[i]*ChebyPolys[i];\n"
           << "};\n"
           //<< "P=P - 1/128 * ChebyPolys[n-1] - 1/128 *ChebyPolys[n];"
           << "plot(P, I, postscriptfile, figureOut);\n";
    output.close();

    std::string sollyaCommand = "sollya ";
    sollyaCommand += fullFilename[i].str();
    system(sollyaCommand.c_str());
  }

  std::stringstream pointDataFile;
  pointDataFile << filename << "_p.dat";
  std::ofstream dataOutput;
  dataOutput.open(pointDataFile.str().c_str());
  for (std::size_t i{0u}; i < points.size(); ++i) {
    mpfr::mpreal pt, ptVal;
    pt = mpfr::acos(points[i]);
    evaluateClenshaw(ptVal, a[0], points[i], prec);
    dataOutput << pt.toString("%.30RNf") << "\t" << ptVal.toString("%.30RNf")
               << std::endl;
  }
  dataOutput.close();

  std::stringstream gnuplotFile;
  gnuplotFile << filename << "_f.p";
  std::stringstream epsFilename;
  epsFilename << filename << "_f.eps";
  std::ofstream output;
  output.open(gnuplotFile.str().c_str());
  mpfr::mpreal startPlot = start * mpfr::const_pi();
  mpfr::mpreal stopPlot = stop * mpfr::const_pi();

  output << "set terminal postscript eps color\n"
         << R"(set out ")" << epsFilename.str() << R"(")" << std::endl
         << R"(set format x "%g")" << std::endl
         << R"(set format y "%g")" << std::endl
         << "set xrange [" << startPlot.toString("%.30RNf") << ":"
         << stopPlot.toString("%.30RNf") << "]\n"
         << R"(plot ")" << datFilename[0].str()
         << R"(" using 1:2 with lines t "", \)" << std::endl;

  for (std::size_t i{1u}; i < a.size() - 1; ++i) {
    output << "\t"
           << R"(")" << datFilename[i].str()
           << R"(" using 1:2 with lines t "", \)" << std::endl;
  }
  if (a.size() > 1u)
    output << "\t"
           << R"(")" << datFilename[a.size() - 1].str()
           << R"(" using 1:2 with lines t "", \)" << std::endl;

  output << "\t"
         << R"(")" << pointDataFile.str() << R"(" using 1:2 with points t "")"
         << std::endl;
  output.close();

  std::string gnuplotCommand = "gnuplot ";
  gnuplotCommand += gnuplotFile.str();
  system(gnuplotCommand.c_str());
}
