#include "malevich_classifier.h"

MalevichClassifier::MalevichClassifier(int clusters) : clusters_(clusters) {
}

void MalevichClassifier::Learn(const Dataset& dataset) {
  prior_.clear();
  //Calculating prior
  for (size_t i = 0; i < dataset.size(); ++i) {
    if (dataset[i].class_label >= (int)prior_.size()) {
      prior_.resize(dataset[i].class_label + 1);
    }
    ++prior_[dataset[i].class_label];
  }
  for (size_t i = 0; i < prior_.size(); ++i) {
    prior_[i] /= dataset.size();
  }

  //Preparing data for EM.
  vector< vector< vector<Feature> > > data(prior_.size());
  for (size_t i = 0; i < dataset.size(); ++i) {
    data[dataset[i].class_label].push_back(dataset[i].features);
  }
  prob_.resize(prior_.size());
  for (size_t i = 0; i < prior_.size(); ++i) {
    prob_[i] = ExpectationMaximization(data[i], 1e-12, clusters_);
  }
}

void MalevichClassifier::Classify(Dataset* dataset) {
  for (size_t i = 0; i < dataset->size(); ++i) {
    long double maxValue = -1;
    int bestClass = 0;
    for (size_t j = 0; j < prob_.size(); ++j) {
      // Assumed equal costs lambda
      long double value = prior_[j] * prob_[j].GetProbability(dataset->at(i).features);
      if (maxValue < value) {
        maxValue = value;
        bestClass = (int)j;
      }
    }
    dataset->at(i).class_label = bestClass;
  }
}