#include <boost/python.hpp>
#include <boost/python/numeric.hpp>

#include <iostream>
void hello() {
    std::cout << "hello" << std::endl;
}

BOOST_PYTHON_MODULE (_pqkmeans) {
    using namespace boost::python;
    def("hello", hello);
}