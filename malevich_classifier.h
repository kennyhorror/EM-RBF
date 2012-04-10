#include <vector>
#include "classifier_interface.h"
#include "expectation_maximization.h"

using std::vector;

class MalevichClassifier: public ClassifierInterface {
public:
  MalevichClassifier(int clusters = -1);
  virtual void Learn(const Dataset& dataset);
  virtual void Classify(Dataset* dataset);
private:
  int clusters_;
  vector<long double> prior_;
  vector<ExpectationMaximization> prob_;
};