#ifndef PQKMEANS_IBKMEANS_INTERNAL_H
#define PQKMEANS_IBKMEANS_INTERNAL_H

namespace pqkmeans {
class IBKmeansInternal {
public:
    virtual void fit(const std::vector<std::vector<unsigned int >> &data) = 0;
    virtual int FindNearestCentroid(const std::vector<unsigned int> &query) = 0;
    virtual const std::vector<int> GetAssignments() = 0;
    virtual const std::shared_ptr<std::vector<std::vector<unsigned int>>> GetClusterCenters() = 0;
    virtual ~IBKmeansInternal() {};
};
}

#endif //PQKMEANS_IBKMEANS_INTERNAL_H
