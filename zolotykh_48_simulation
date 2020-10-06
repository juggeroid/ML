#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include <array>
#include <random>

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
