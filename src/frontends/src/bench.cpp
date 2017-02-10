#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/thread.hpp>

#include <clipper/config.hpp>
#include <clipper/datatypes.hpp>
#include <clipper/query_processor.hpp>

using namespace clipper;
using boost::future;
using std::vector;

constexpr int SLO_MICROS = 20000;

long get_duration_micros(
    std::chrono::time_point<std::chrono::high_resolution_clock> end,
    std::chrono::time_point<std::chrono::high_resolution_clock> start) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

double compute_mean(const std::vector<long>& measurements) {
  double sum = 0.0;
  for (auto m : measurements) {
    sum += m;
  }
  return sum / (double)measurements.size();
}

double compute_percentile(std::vector<long> measurements, double percentile) {
  // sort in ascending order
  std::sort(measurements.begin(), measurements.end());
  double sample_size = measurements.size();
  assert(percentile >= 0.0 && percentile <= 1.0);
  double x;
  if (percentile <= (1.0 / (sample_size + 1.0))) {
    x = 1.0;
  } else if (percentile > (1.0 / (sample_size + 1.0)) &&
             percentile < (sample_size / (sample_size + 1.0))) {
    x = percentile * (sample_size + 1.0);
  } else {
    x = sample_size;
  }
  int index = std::floor(x) - 1;
  double value = measurements[index];
  double remainder = std::fmod(x, 1.0);
  if (remainder != 0.0) {
    value += remainder * (measurements[index + 1] - measurements[index]);
  }
  return value;
}

std::shared_ptr<DoubleVector> generate_rand_doublevec(
    int input_len, boost::random::mt19937& gen) {
  vector<double> input;
  boost::random::uniform_real_distribution<> dist(0.0, 1.0);
  for (int i = 0; i < input_len; ++i) {
    input.push_back(dist(gen));
  }
  return std::make_shared<DoubleVector>(input);
}

Query generate_query(int input_len, boost::random::mt19937& gen) {
  std::shared_ptr<Input> input = generate_rand_doublevec(input_len, gen);
  vector<VersionedModelId> models{std::make_pair("m", 1),
                                  std::make_pair("j", 1)};
  return Query{"bench", 3, input, SLO_MICROS, "simple_policy", models};
}

void run_benchmark(QueryProcessor& qp, int num_requests) {
  boost::random::mt19937 gen(std::time(0));
  vector<future<Response>> preds;
  auto start = std::chrono::high_resolution_clock::now();

  for (int req_num = 0; req_num < num_requests; ++req_num) {
    preds.push_back(qp.predict(generate_query(1000, gen)));
  }

  vector<long> durations;
  double completed_tasks_sum = 0.0;

  for (auto p = preds.begin(); p != preds.end(); ++p) {
    Response r{p->get()};
    durations.push_back(r.duration_micros_);
    completed_tasks_sum += r.output_.y_hat_;
  }
  auto end = std::chrono::high_resolution_clock::now();
  double benchmark_time_secs =
      get_duration_micros(end, start) / 1000.0 / 1000.0;

  double thruput = num_requests / benchmark_time_secs;

  double p99 = compute_percentile(durations, 0.99);
  double mean_lat = compute_mean(durations);
  std::cout << "Sent " << num_requests << " in " << benchmark_time_secs
            << " seconds" << std::endl;
  std::cout << "Throughput: " << thruput << std::endl;
  std::cout << "p99 latency (us): " << p99 << ", mean latency (us) " << mean_lat
            << std::endl;
  std::cout << "Mean tasks completed: "
            << completed_tasks_sum / (double)num_requests << std::endl;
}

void drive_benchmark() {
  QueryProcessor qp;
  std::this_thread::sleep_for(std::chrono::seconds(3));
  std::string line;
  std::cout << "Please enter number of requests to make:" << std::endl;
  while (std::getline(std::cin, line)) {
    try {
      int num_reqs = std::stoi(line);
      std::cout << "Running benchmark..." << std::endl;
      run_benchmark(qp, num_reqs);
      std::cout << std::endl;
    } catch (std::invalid_argument e) {
    }
    std::cout << "Please enter number of requests to make:" << std::endl;
  }
}

int main() {
  clipper::Config& conf = clipper::get_config();
  // conf.set_redis_port(clipper_test::REDIS_TEST_PORT);
  conf.ready();
  drive_benchmark();
  return 0;
}
