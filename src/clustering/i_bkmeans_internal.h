//

#ifndef PQKMEANS_IBKMEANS_INTERNAL_H
#define PQKMEANS_IBKMEANS_INTERNAL_H

namespace pqkmeans {
    class IBKmeansInternal {
    public:
        virtual void fit(const std::vector<std::vector<unsigned int >> &data) = 0;
    };
}

#endif //PQKMEANS_IBKMEANS_INTERNAL_H
