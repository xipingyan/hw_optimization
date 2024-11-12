// Reference:
// https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/executing-multiple-kernels-on-the-device-at-the.html
// In-order / Out-of-order

#include <random>
#include <thread>

#include "my_common.hpp"

#define iter 10
#define array_size (1024)
using IntArray = std::array<int, array_size>;

// loop test 3 kernel.
int multi_queue(sycl::queue &q, const IntArray &a, const IntArray &b)
{
  IntArray s1, s2, s3;

  sycl::buffer a_buf(a);
  sycl::buffer b_buf(b);
  sycl::buffer sum_buf1(s1);
  sycl::buffer sum_buf2(s2);
  sycl::buffer sum_buf3(s3);

  size_t num_groups = 1; // 1 work-group
  size_t wg_size = 256;  // 256 work-items
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iter; i++)
  {
    q.submit([&](sycl::handler &h)
             {
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf1, h, sycl::write_only, sycl::no_init);

      // These kernels are created specifically with 1 group to ensure that they do not use the entire machine
      h.parallel_for<class my_kernel_1>(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                      [=](sycl::nd_item<1> index) {
                        size_t loc_id = index.get_local_id();
                        sum_acc[loc_id] = 0;
                        for (int j = 0; j < 1000; j++)
                          for (size_t i = loc_id; i < array_size; i += wg_size) {
                            sum_acc[loc_id] += a_acc[i] + b_acc[i];
                          }
                      }); });
    q.submit([&](sycl::handler &h)
             {
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf2, h, sycl::write_only, sycl::no_init);

      h.parallel_for<class my_kernel_2>(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                      [=](sycl::nd_item<1> index) {
                        size_t loc_id = index.get_local_id();
                        sum_acc[loc_id] = 0;
                        for (int j = 0; j < 1000; j++)
                          for (size_t i = loc_id; i < array_size; i += wg_size) {
                            sum_acc[loc_id] += a_acc[i] + b_acc[i];
                          }
                      }); });
    q.submit([&](sycl::handler &h)
             {
      sycl::accessor a_acc(a_buf, h, sycl::read_only);
      sycl::accessor b_acc(b_buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf3, h, sycl::write_only, sycl::no_init);

      h.parallel_for<class my_kernel_3>(sycl::nd_range<1>(num_groups * wg_size, wg_size),
        [=](sycl::nd_item<1> index) {
          size_t loc_id = index.get_local_id();
          sum_acc[loc_id] = 0;
          for (int j = 0; j < 1000; j++)
            for (size_t i = loc_id; i < array_size; i += wg_size) {
              sum_acc[loc_id] += a_acc[i] + b_acc[i];
            }
        }); });
  }
  q.wait();
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "  == multi_queue completed on device - took "
            << tm_diff_ms(t1, t2) << " ms\n";
  // check results
  return tm_diff_ms(t1, t2);
}

void test_in_order()
{
  std::cout << "== Start testing in-order queue...." << std::endl;
  IntArray a, b;
  sycl::property_list q_prop{sycl::property::queue::in_order()};
  sycl::queue q_in_order(sycl::gpu_selector_v, q_prop);
  print_device_beckend(q_in_order);
  auto order_properity = q_in_order.get_property<sycl::property::queue::in_order>();
  std::cout << "  order_properity = " << order_properity.getKind() << std::endl;

  std::cout << "  In order queue: Jitting+Execution time\n";
  multi_queue(q_in_order, a, b);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  std::cout << "  In order queue: Execution time\n";
  multi_queue(q_in_order, a, b);
}

void test_out_of_order()
{
  std::cout << "== Start testing out-of-order queue...." << std::endl;
  IntArray a, b;

  std::cout << "  Out of order queue: Jitting+Execution time\n";
  sycl::queue q_default(sycl::gpu_selector_v);
  print_device_beckend(q_default);

  multi_queue(q_default, a, b);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  std::cout << "  Out of order queue: Execution time\n";
  multi_queue(q_default, a, b);
}

int main(int argc, char *argv[])
{
  test_in_order();
  test_out_of_order();
  std::cout << "======================================" << std::endl;
  std::cout << "Conclustion: Out-of-order still execute in parallel based on onetrace check. I don't know why?" << std::endl;
  return 0;
}