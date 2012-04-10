#include <cmath>
#include "util.h"
#include <iostream>

using std::vector;

vector< vector<long double> > Cholesky(const vector< vector<long double> > &matrix) {
  vector< vector<long double> > result(matrix.size(), vector<long double>(matrix.size()));

  for (size_t i = 0; i < matrix.size(); ++i) {
    long double s = 0;
    for (size_t k = 0; k < i; ++k) {
      s += result[k][i] * result[k][i];
    }
    long double d = matrix[i][i] - s;
    if (fabs(d) < kEps) {
      result[i][i] = 0.0;
    } else if (d < 0) {
      throw BadMatrix();
    } else {
      result[i][i] = sqrt(d);
    }
    for (size_t j = i + 1; j < matrix.size(); ++j) {
      s = 0;
      for (size_t k = 0; k < i; ++k) {
        s += result[k][i] * result[k][j];
      }
      result[i][j] = (matrix[i][j] - s) / result[i][i];
    }
  }

  return result;
}

vector< vector<long double> > GetCholeskyInverse(const vector< vector<long double> > &matrix) {
  vector< vector<long double> > result (matrix.size(), vector<long double>(matrix.size(), 0.0)),
    cur = Cholesky(matrix);

  for (int j = (int)cur.size() - 1; j > -1; --j) {
    long double v = cur[j][j];
    long double s = 0;
    for (size_t k = j + 1; k < cur.size(); ++k) {
      s += cur[j][k] * result[j][k];
    }
    result[j][j] = 1.0 / v / v - s / v;
    for (int i = j - 1; i > -1; --i) {
      s = 0;
      for (size_t k = i + 1; k < cur.size(); ++k) {
        s += cur[i][k] * result[k][j];
      }
      result[j][i] = result[i][j] = -s / cur[i][i];
    }
  }
  return result;
}

vector< vector<long double> > GetInverse(vector< vector<long double> > matrix) {
  //Regularization
  for (size_t i = 0; i < matrix.size(); ++i) {
    matrix[i][i] += 0.000065;
  }
  return GetCholeskyInverse(matrix);
}

long double GetDet(vector< vector<long double> > matrix) {
  try {
    vector< vector<long double> > matrix2 = Cholesky(matrix);
    long double ret = 1.0;
    for (size_t i = 0; i < matrix2.size(); ++i) {
      ret *= matrix2[i][i];
    }
    return ret * ret;
  } catch (...) {
  }
  double ret = 1.0;
  for (size_t i = 0; i < matrix.size(); ++i) {
    size_t ps = i;
    double mx = -1.0;
    for (size_t j = i; j < matrix.size(); ++j) {
      if (fabs(matrix[j][i]) > mx) {
        mx = fabs(matrix[j][i]);
        ps = j;
      }
    }
    if (mx > kEps) {
      std::swap(matrix[i], matrix[ps]);
      if (ps != i) {
        ret *= -1.0;
      }
    } else {
      return 0.0;
    }
    ret *= matrix[i][i];
    for (size_t j = i + 1; j < matrix.size(); ++j) {
      matrix[i][j] /= matrix[i][i];
    }
    for (size_t j = i + 1; j < matrix.size(); ++j) {
      for (size_t k = i + 1; k < matrix.size(); ++k) {
        matrix[j][k] -= matrix[i][k] * matrix[j][i];
      }
    }
  }
  return ret;
}