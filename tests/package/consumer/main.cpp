#include <mnet/result.hpp>

int main() {
    const mnet::Status status{};
    return status.ok() ? 0 : 1;
}
