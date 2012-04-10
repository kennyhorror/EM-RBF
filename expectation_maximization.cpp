#include <iostream>
#include <algorithm>
#include "expectation_maximization.h"
#include "util.h"

long double Sqr(const long double &x) {
  return x * x;
}

vector<long double> Convert(const vector<Feature> &data) {
  vector<long double> result(data.size());
  for (size_t j = 0; j < result.size(); ++j) {
    result[j] = data[j];
  }
  return result;
}

vector<long double> GetMean(const vector< vector<long double> > &x) {
  vector<long double> result(x[0].size());
  for (size_t i = 0; i < x.size(); ++i) {
    for (size_t j = 0; j < x[i].size(); ++j) {
      result[j] += x[i][j];
    }
  }
  for (size_t j = 0; j < result.size(); ++j) {
    result[j] /= x.size();
  }
  return result;
}

vector< vector<long double> > GetDisperse(const vector< vector<long double> > &x,
    const vector<long double> &mu) {
  vector< vector<long double> > result(x[0].size(), vector<long double>(x[0].size()));
  for (size_t i = 0; i < x.size(); ++i) {
    for (size_t j = 0; j < x[i].size(); ++j) {
      for (size_t k = 0; k < x[i].size(); ++k) {
        result[j][k] += (x[i][j] - mu[j]) * (x[i][k] - mu[k]);
      }
    }
  }
  for (size_t j = 0; j < result.size(); ++j) {
    for (size_t k = 0; k < result[0].size(); ++k) {
      result[j][k] /= x.size();
    }
  }
  return result;
}

long double GetSquaredDistance(const vector<Feature> &f, const vector<long double> &s) {
  long double ret = 0;
  for (size_t i = 0; i < f.size(); ++i) {
    ret += Sqr(f[i] - s[i]);
  }
  return ret;
}

ExpectationMaximization::ExpectationMaximization(int count) : count_(count) {
}

ExpectationMaximization::ExpectationMaximization(const vector< vector<Feature> > &x,
    long double delta, int count) : count_(count) {
  GetAproximation(x);
  Maximize(x, delta);
}

void ExpectationMaximization::kMeans(const vector< vector<Feature> > &data, int clusters,
    bool last) {
  //Preparation
  vector<size_t> cluster(data.size());
  vector< vector<long double> > sum(clusters, vector<long double>(data[0].size()));
  vector<int> count(clusters);

  //Getting initial centers
  mu_.clear();
  for (int i = 0; i < clusters; ++i) {
    mu_.push_back(Convert(data[rand() % data.size()]));
  }

  //Calculations
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < data.size(); ++i) {
      long double best = kInf;
      size_t bestCluster = 0;
      for (size_t j = 0; j < mu_.size(); ++j) {
        long double dist = GetSquaredDistance(data[i], mu_[j]);
        if (dist < best) {
          best = dist;
          bestCluster = j;
        }
      }
      if (bestCluster != cluster[i]) {
        cluster[i] = bestCluster;
        changed = true;
      }
    }

    for (size_t i = 0; i < sum.size(); ++i) {
      count[i] = 0;
      for (size_t j = 0; j < sum[0].size(); ++j) {
        sum[i][j] = 0;
      }
    }
    for (size_t i = 0; i < data.size(); ++i) {
      ++count[cluster[i]];
      for (size_t j = 0; j < data[i].size(); ++j) {
        sum[cluster[i]][j] += data[i][j];
      }
    }
    for (size_t i = 0; i < sum.size(); ++i) {
      if (count[i]) {
        for (size_t j = 0; j < sum[0].size(); ++j) {
          mu_[i][j] = sum[i][j] / count[i];
        }
      } else {
        //Useless one
        mu_.erase(mu_.begin() + i);
        sum.erase(sum.begin() + i);
        count.erase(count.begin() + i);
        --clusters;
        --i;
      }
    }
  }

  if (last) {
    //Getting approximations
    w_.resize(clusters);
    det_.resize(clusters);
    sigma_.resize(clusters);
    vector< vector< vector<long double> > > cls(clusters);

    for (size_t i = 0; i < data.size(); ++i) {
      cls[cluster[i]].push_back(Convert(data[i]));
    }
    for (size_t i = 0; i < sum.size(); ++i) {
      w_[i] = (long double)cls[i].size() / data.size();
      sigma_[i] = GetDisperse(cls[i], mu_[i]);
      det_[i] = GetDet(sigma_[i]);
      sigma_[i] = GetInverse(sigma_[i]);
    }
  }
}

//Gonna modify data a bit, so copy
void ExpectationMaximization::GetAproximation(vector< vector<Feature> > x) {
  std::random_shuffle(x.begin(), x.end());
  int bestK = 1;
  if (count_ == -1) { 
    vector< vector< vector<Feature> > > data(10);
    vector< vector< vector<Feature> > > test(10);
    for (size_t i = 0; i < x.size(); ++i) {
      for (int j = 0; j < 10; ++j) {
        if (10 * i / x.size() == j) {
          test[j].push_back(x[i]);
        } else {
          data[j].push_back(x[i]);
        }
      }
    }

    long double minSum = kInf;
    bestK = 1;
    for (int k = 1; k < 16; ++k) {
      long double sum = 0;
      for (int j = 0; j < 10; ++j) {
        kMeans(data[j], k);
        for (size_t i = 0; i < test[j].size(); ++i) {
          long double dst = kInf;
          for (size_t clst = 0; clst < mu_.size(); ++clst){
            dst = std::min(dst, (long double)sqrt(GetSquaredDistance(test[j][i], mu_[clst])));
          }
          sum += dst;
        }
      }
      if (sum * 1.01 < minSum) {
        std::cerr << sum << std::endl;
        minSum = sum;
        bestK = k;
      } else {
        break;
      }
    }
  } else {
    bestK = count_;
  }
  std::cerr << "There're : " << bestK << " clusters" << std::endl;
  kMeans(x, bestK, true);
}

long double ExpectationMaximization::Maximize(const vector< vector<Feature> > &x, long double delta) {
  vector< vector<long double> > g;
  g.resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    g[i].resize(w_.size());
  }
  long double maxg = 0.0;
  for (int iter = 0; iter < kMaxSteps; ++iter) {
    maxg = 0.0;
    //E step.
    for (size_t i = 0; i < x.size(); ++i) {
      long double sum = 0;
      vector<long double> tmp(w_.size());
      for (size_t j = 0; j < w_.size(); ++j) {
        tmp[j] = w_[j] * GetProbabilityOnComponent(x[i], j);
        sum += tmp[j];
      }
      for (size_t j = 0; j < w_.size(); ++j) {
        tmp[j] /= sum;
        maxg = std::max(maxg, (long double)fabs(g[i][j] - tmp[j]));
        g[i][j] = tmp[j];
      }
    }

    //M step.
    w_.assign(w_.size(), 0.0);
    for (size_t j = 0; j < w_.size(); ++j) {
      for (size_t i = 0; i < x.size(); ++i) {
        w_[j] += g[i][j];
      }
      w_[j] /= x.size();
    }
    for (size_t j = 0; j < w_.size(); ++j) {
      if (w_[j] < kEps) {
        continue;
      }
      mu_[j].assign(x[0].size(), 0.0);
      for (size_t i = 0; i < x.size(); ++i) {
        for (size_t k = 0; k < x[i].size(); ++k) {
          mu_[j][k] += g[i][j] * x[i][k];
        }
      }
      for (size_t k = 0; k < mu_[j].size(); ++k) {
        mu_[j][k] /= x.size() * w_[j];
      }

      sigma_[j].assign(x[0].size(), vector<long double>(x[0].size()));
      for (size_t i = 0; i < x.size(); ++i) {
        for (size_t k = 0; k < x[i].size(); ++k) {
          for (size_t l = 0; l < x[i].size(); ++l) {
            sigma_[j][k][l] += g[i][j] * (x[i][k] - mu_[j][k]) * (x[i][l] - mu_[j][l]);
          }
        }
      }
      for (size_t k = 0; k < sigma_[j].size(); ++k) {
        for (size_t l = 0; l < sigma_[j].size(); ++l) {
          sigma_[j][k][l] /= x.size() * w_[j];
        }
      }
    }
    for (size_t j = 0; j < w_.size(); ++j) {
      det_[j] = GetDet(sigma_[j]);
      sigma_[j] = GetInverse(sigma_[j]);
    }
    if (maxg < delta) {
    //  break;
    }
  }
  return maxg;
}

long double ExpectationMaximization::GetProbability(const vector<Feature> &x) {
  long double p = 0;
  for (size_t i = 0; i < w_.size(); ++i) {
    p += w_[i] * GetProbabilityOnComponent(x, i);
  }
  return p;
}

long double ExpectationMaximization::GetProbabilityOnComponent(const vector<Feature> &data, size_t id) {
  if (sigma_.empty()) {
    return 0.0;
  }
  long double result = 1.0;
  long double sum = 0.0;
  for (size_t i = 0; i < data.size(); ++i) {
    for (size_t j = 0; j < data.size(); ++j) {
      sum += (data[i] -  mu_[id][i]) * (data[j] - mu_[id][j])  * sigma_[id][i][j];
    }
  }

  result *= exp(-sum / 2.0) / pow(2.0 * kPi, data.size() / 2.0);
  if (det_[id] > 0.0) {
    return result / sqrt(det_[id]);
  } else {
    return 0.0;
  }
}