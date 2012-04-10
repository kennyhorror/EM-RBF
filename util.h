#include "classifier_interface.h"
#include <vector>

const double kEps = 1e-10;

class BadMatrix: public std::exception {
public:
  virtual const char* what() const throw() {
    return "Non positive definit matrix";
  }
};


std::vector< std::vector<long double> > GetInverse(std::vector< std::vector<long double> > matrix);
long double GetDet(std::vector< std::vector<long double> > matrix);