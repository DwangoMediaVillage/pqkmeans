//

#ifndef PQKMEANS_IBKMEANSIMPL_H
#define PQKMEANS_IBKMEANSIMPL_H

namespace pqkmeans {
    class IBKmeansInternal {
    public:
        virtual void fit(const std::vector<std::vector<unsigned int >> &data) = 0;
    };
}

#endif //PQKMEANS_IBKMEANSIMPL_H
