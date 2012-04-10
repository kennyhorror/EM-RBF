#ifndef __EM__
#define __EM__
#include <cmath>
#include <vector>
#include "classifier_interface.h"

using std::vector;

const double kPi = acos(-1.0);
const double kInf = 1e300;
const int kMaxSteps = 1000;

class ExpectationMaximization {
public:
  ExpectationMaximization(int count = -1);
  ExpectationMaximization(const vector< vector<Feature> > &x, long double delta, int count = -1);
  void GetAproximation(vector< vector<Feature> > x);
  void kMeans(const vector< vector<Feature> > &data, int clusters, bool last = false);
  long double Maximize(const vector< vector<Feature> > &x, long double delta);
  long double GetProbability(const vector<Feature> &x);
  long double GetProbabilityOnComponent(const vector<Feature> &data, size_t id);
private:
  int count_;
  vector<long double> w_;
  vector<long double> det_;
  vector< vector<long double> > mu_;
  vector< vector< vector<long double> > > sigma_;
};
#endif
