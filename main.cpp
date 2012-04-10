#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include "util.h"
#include "malevich_classifier.h"

//#define CROSSVALIDATION
//#define GETPOINTS


#ifdef CLUSTER
const char* kTrainname1 = "/home/malevich/classifier/train1.csv";
const char* kTestname1 = "/home/malevich/classifier/test1.csv";
const char* kTrainname2 = "/home/malevich/classifier/train2.csv";
const char* kTestname2 = "/home/malevich/classifier/test2.csv";
const char* kResultname = "/home/malevich/classifier/prediction.txt";
#else
const char* kTrainname1 = "train1.csv";
const char* kTestname1 = "test1.csv";
const char* kTrainname2 = "train2.csv";
const char* kTestname2 = "test2.csv";
const char* kResultname = "prediction.txt";
#endif

void LoadDataset(const std::string& filename, Dataset* dataset) {
  bool test = false;
  if (filename.find("test") != -1) {
    test = true;
  }
  std::string line;
  std::ifstream in(filename.c_str());
  int position = 0;
  while (std::getline(in, line)) {
    Object object;
    size_t features = std::count(line.begin(), line.end(), ',');
    if (test) {
      ++features;
    }
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream parser(line);
    for (size_t i = 0; i < features; ++i) {
      double value;
      parser >> value;
      object.features.push_back(value);
    }
    if (!test) {
      int label;
      parser >> label;
      object.class_label = label;
    }
    dataset->push_back(object);
    ++position;
  }
}

void PrintDataset(const std::string& filename, const Dataset &data1, const Dataset &data2) {
  std::ofstream out(filename.c_str());
  for (size_t i = 0; i < data1.size(); ++i) {
#ifdef GETPOINTS
    for (size_t j = 0; j < data1[i].features.size(); ++j) {
      out << data1[i].features[j] << ",";
    }
#endif
    out << data1[i].class_label << "\n";
  }
  for (size_t i = 0; i < data2.size(); ++i) {
#ifdef GETPOINTS
    for (size_t j = 0; j < data2[i].features.size(); ++j) {
      out << data2[i].features[j] << ",";
    }
#endif
    out << data2[i].class_label << "\n";
  }
}

void CrossValidation(Dataset data, int clusters = -1) {
  std::random_shuffle(data.begin(), data.end());
  double error = 0, total = 0;
  #pragma omp parallel for reduction(+ : total) reduction(+ : error)
  for (int i = 0; i < 10; ++i) {
    Dataset train, test, answer;
    for (size_t j = 0; j < data.size(); ++j) {
      if (j * 10 / data.size() == i) {
        test.push_back(data[j]);
        answer.push_back(data[j]);
      } else {
        train.push_back(data[j]);
      }
    }
    MalevichClassifier classfier1(clusters);
    classfier1.Learn(train);
    classfier1.Classify(&test);
    for (size_t j = 0; j < test.size(); ++j) {
      if (test[j].class_label != answer[j].class_label) {
        ++error;
      }
    }
    total += test.size();
  }
  printf("Errors percentage: %f", error / total);
}

int main() {
  Dataset train1, test1, train2, test2;
  LoadDataset(kTrainname1, &train1);
  LoadDataset(kTestname1, &test1);
  LoadDataset(kTrainname2, &train2);
  LoadDataset(kTestname2, &test2);

#ifdef CROSSVALIDATION
  CrossValidation(train1, 2);
  CrossValidation(train2, 10);
#else
  MalevichClassifier classfier1(2);
  classfier1.Learn(train1);
  classfier1.Classify(&test1);
  MalevichClassifier classfier2(10);
  classfier2.Learn(train2);
  classfier2.Classify(&test2);
  PrintDataset(kResultname, test1, test2);
#endif
  return 0;
}
