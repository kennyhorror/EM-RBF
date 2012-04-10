#ifndef CLASSIFIER_INTERFACE_H__
#define CLASSIFIER_INTERFACE_H__

#include <vector>
#include <string>

// значение признака
typedef double Feature;

// объект
struct Object {
    std::vector<Feature> features;
	int class_label;
};

// выборка объектов. Используется как для данных обучения, так и для данных теста.
typedef std::vector<Object> Dataset;

// функция загрузки данных из файла. Эту функцию вам следует реализовать в каком-нибудь .cpp
void LoadDataset(const std::string& filename, Dataset* dataset);

// интерфейс классификатора. Класс с вашим классификатором должен унаследоваться от ClassifierInterface.
class ClassifierInterface {
public:
    virtual void Learn(const Dataset& dataset) = 0;
    virtual void Classify(Dataset* dataset) = 0;
};

#endif
