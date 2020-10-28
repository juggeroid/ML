#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include <array>
#include <random>

// Let a, b be a points with coordinates (0, \theta_1), (1, \theta_2) respectively. It's known that \theta_1 ~ U[-1, 1], \theta_2 ~ U[-1, 1]. 
// There's also a fixed point x = (0.32, 0). Find the probability P(d(b, x) < d(a, x)), where d(\dot, \dot) is a euclidean metric.
// The analytic solution is available here: https://stats.stackexchange.com/questions/490357/probability-of-one-point-being-closer-to-fixed-x-in-mathbfr2-than-another

namespace {
	static constexpr auto ITERATIONS = 20'000'000;
}

struct point_t {
	constexpr point_t(double x, double y): x {x}, y {y} {}
	double x = 0, y = 0;
};

auto point_distance(point_t const& a, point_t const& b) {
	return std::hypot(a.x - b.x, a.y - b.y);
}

int main() {
	static std::uniform_real_distribution<double> distribution(-1, 1);
	static std::mt19937_64 generator {std::random_device {}()};
	static constexpr auto x = point_t {0.32, 0};
	auto probability = 0.0;
	for (size_t iteration = 0; iteration < ITERATIONS; ++iteration) {
		const auto [a, b] = std::pair {point_t {0, distribution(generator)}, point_t {1, distribution(generator)} };
		probability += point_distance(b, x) < point_distance(a, x);
	}
	std::cout << (probability / ITERATIONS) << "\n";
}
