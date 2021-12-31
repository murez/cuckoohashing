#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "xxhash.hcu"
#include "utils.hcu"
#include "cuckoo_serial.hcu"
#include "cuckoo_cuda_native.hcu"
#include <string>
void test_cuckoo_serial();
void test_cuckoo_cuda();
void test_cuckoo_cuda_native();
void bench_task1_cuckoo_serial();
void bench_task2_cuckoo_serial();
void bench_task3_cuckoo_serial();
void bench_task4_cuckoo_serial();
int main()
{
    // ha? naïve
    srand(19260817);

    CuckooCudaHashTable q(1000, 12, 3);
    uint32_t keys[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    q.insert(keys, 12);

    test_xxhash();
    test_cuckoo_serial();
    bench_task4_cuckoo_serial();
}

void bench_task1_cuckoo_serial()
{
    uint32_t hash_table_size = 1 << 25;
    for (int num_funcs = 2; num_funcs <= 4; ++num_funcs)
    {
        for (int scale = 10; scale <= 24; ++scale)
        {
            int n = 1 << scale;
            uint32_t *input_keys = new uint32_t[n];
            generate_unique_random(input_keys, n);
            for (int rep = 0; rep < 1; rep++)
            {
                CuckooSerialHashTable ht(hash_table_size, 4 * clog2(n), num_funcs);
                int level;
                double duation = time_func([&]
                                           { level = ht.insert(input_keys, n); });

                printf("time: %lf %d\n", duation, level);
            }
            delete[] input_keys;
        }
    }
}

void bench_task2_cuckoo_serial()
{
    uint32_t hash_table_size = 1 << 25;
    uint32_t n = 1 << 24;
    for (int num_funcs = 3; num_funcs <= 4; ++num_funcs)
    {
        uint32_t *insert_values = new uint32_t[n];
        uint32_t *lookup_values = new uint32_t[n];
        bool *results = new bool[n];
        for (int percent = 0; percent <= 10; ++percent)
        {
            int bound = ceil((1 - 0.1 * percent) * n);
            for (int rep = 0; rep < 5; ++rep)
            {
                generate_unique_random(insert_values, n);
                for (int i = 0; i < bound; ++i)
                    lookup_values[i] = insert_values[rand() % n];
                for (int i = bound; i < n; ++i)
                    lookup_values[i] = rand() % (int)(1.5 * n) + 1;
                CuckooSerialHashTable ht(hash_table_size, 4 * clog2(n), num_funcs);
                ht.insert(insert_values, n);
                double duation = time_func([&]
                                           { ht.lookup(lookup_values, results, n); });
                printf("time: %lf %d\n", duation, percent);
            }
        }
        delete[] insert_values;
        delete[] lookup_values;
        delete[] results;
    }
}

void bench_task3_cuckoo_serial()
{
    uint32_t n = 1 << 24;
    uint32_t *insert_values = new uint32_t[n];
    generate_unique_random(insert_values, n);
    for (int num_funcs = 3; num_funcs <= 4; ++num_funcs)
    {
        float ratios[] = {1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1.02, 1.01};
        for (int ri = 0; ri < 12; ++ri)
        {
            int size = ceil(ratios[ri] * n);
            for (int rep = 0; rep < 5; ++rep)
            {
                CuckooSerialHashTable ht(size, 4 * clog2(n), num_funcs);
                int level;
                double duation = time_func([&]
                                           { level = ht.insert(insert_values, n); });
                printf("%lf, %d\n", duation, level);
            }
        }
    }
}

void bench_task4_cuckoo_serial()
{
    int n = 0x1 << 24, size = ceil(1.4 * n);
    uint32_t *insert_values = new uint32_t[n];
    generate_unique_random(insert_values, n);
    for (int num_funcs = 3; num_funcs <= 4; ++num_funcs)
    {
        for (int bound_mul = 1; bound_mul <= 10; ++bound_mul)
        {
            for (int rep = 0; rep < 5; rep++)
            {
                CuckooSerialHashTable ht(size, bound_mul * clog2(n), num_funcs);
                int level;
                double duation = time_func([&]
                                           { level = ht.insert(insert_values, n); });
                printf("%lf, %d\n", duation, level);
            }
        }
    }
}
void test_cuckoo_serial()
{
    CuckooSerialHashTable x(1000, 12, 3);
    test_hashtable("CuckooSerialHashTable", x);
}

// void test_cuckoo_cuda()
// {
//     CuckooCudaHashTable x(1000, 12, 3);
//     test_hashtable("CuckooSerialHashTable", x);
// }