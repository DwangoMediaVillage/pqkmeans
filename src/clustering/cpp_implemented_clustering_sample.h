//

#ifndef PQKMEANS_NATIVECLUSTERINGSAMPLE_H
#define PQKMEANS_NATIVECLUSTERINGSAMPLE_H

#include <cassert>
#include <vector>

namespace pqkmeans {
    class CppImplementedClusteringSample {
    private:
        std::vector<float> min_vec;
        std::vector<float> max_vec;
    public:
        void fit_one(const std::vector<float> &pyvector);

        int predict_one(const std::vector<float> &pyvector);
    };
}

#endif //PQKMEANS_NATIVECLUSTERINGSAMPLE_H
