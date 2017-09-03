//

#ifndef PQKMEANS_NATIVECLUSTERINGSAMPLE_H
#define PQKMEANS_NATIVECLUSTERINGSAMPLE_H

#include <cassert>
#include <vector>

class NativeClusteringSample {
private:
    std::vector<float> min_vec;
    std::vector<float> max_vec;
public:
    void fit_one(std::vector<float> pyvector);
    std::vector<float> transform_one(std::vector<float> pyvector);
};


#endif //PQKMEANS_NATIVECLUSTERINGSAMPLE_H
