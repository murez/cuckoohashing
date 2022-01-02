#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "xxhash.hcu"
#include "utils.hcu"
#include "cuckoo_serial.hcu"
#include "cuckoo_cuda_native.hcu"
#include <string>
#include "CSVWriter.h"
using namespace csv;
void test_cuckoo_serial();
void test_cuckoo_cuda();
void bench_task1_cuckoo();
void bench_task2_cuckoo();
void bench_task3_cuckoo();
void bench_task4_cuckoo();

int main()
{
    // ha? naïve
    srand(19260817);

    // test_xxhash();
    // test_cuckoo_serial();
    // test_cuckoo_cuda();
    bench_task1_cuckoo();
    bench_task2_cuckoo();
    bench_task3_cuckoo();
    bench_task4_cuckoo();
    return 0;
}

void bench_task1_cuckoo()
{
    // create csv writer
    CsvWriter serial_writer("task1_serial.csv");
    CsvWriter cuda_writer("task1_cuda.csv");

    Record header;
    header.put("func_num");
    header.put("scale");
    header.put("mean");
    header.put("stddev");

    serial_writer.setHeader(header);
    cuda_writer.setHeader(header);

    printf("task 1\n");
    uint32_t hash_table_size = 1 << 25;

    for (int num_funcs = 2; num_funcs <= 4; ++num_funcs)
    {
        for (int scale = 10; scale <= 24; ++scale)
        {
            int n = 1 << scale;
            uint32_t *input_keys = new uint32_t[n];
            generate_unique_random(input_keys, n);

            double duration_serial[5];
            double duration_cuda[5];

            for (int rep = 0; rep < 5; rep++)
            {
                CuckooSerialHashTable ht_serial(hash_table_size, 4 * clog2(n), num_funcs);
                CuckooCudaHashTable ht_cuda(hash_table_size, 4 * clog2(n), num_funcs);
                int level_serial;
                int level_cuda;
                duration_serial[rep] = time_func([&]
                                                 { level_serial = ht_serial.insert(input_keys, n); });
                duration_cuda[rep] = time_func([&]
                                               { level_cuda = ht_cuda.insert(input_keys, n); });

                printf("hash-func-num %-10d scale %-10d repeat-times %-10d serial time: %-10lf rehash %-10d cuda time: %-10lf rehash %-10d\n", num_funcs, scale, rep, duration_serial[rep], level_serial, duration_cuda[rep], level_cuda);
            }

            double mean_serial = average(duration_serial, 5);
            double stddev_serial = standardDev(duration_serial, 5);

            double mean_cuda = average(duration_cuda, 5);
            double stddev_cuda = standardDev(duration_cuda, 5);

            Record record;
            record.put(num_funcs);
            record.put(scale);
            record.put(mean_serial);
            record.put(stddev_serial);
            serial_writer.insertRecord(record);

            Record record_cuda;
            record_cuda.put(num_funcs);
            record_cuda.put(scale);
            record_cuda.put(mean_cuda);
            record_cuda.put(stddev_cuda);
            cuda_writer.insertRecord(record_cuda);

            delete[] input_keys;
        }
    }

    serial_writer.write();
    cuda_writer.write();
}

void bench_task2_cuckoo()
{
    CsvWriter serial_writer("task2_serial.csv");
    CsvWriter cuda_writer("task2_cuda.csv");

    Record header;
    header.put("func_num");
    header.put("percent");
    header.put("mean");
    header.put("stddev");

    serial_writer.setHeader(header);
    cuda_writer.setHeader(header);

    printf("task 2\n");
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
            double duration_serial[5];
            double duration_cuda[5];
            for (int rep = 0; rep < 5; ++rep)
            {
                generate_unique_random(insert_values, n);
                for (int i = 0; i < bound; ++i)
                    lookup_values[i] = insert_values[rand() % n];
                for (int i = bound; i < n; ++i)
                    lookup_values[i] = rand() % (int)(1.5 * n) + 1;
                CuckooSerialHashTable ht_serial(hash_table_size, 4 * clog2(n), num_funcs);
                CuckooCudaHashTable ht_cuda(hash_table_size, 4 * clog2(n), num_funcs);
                ht_serial.insert(insert_values, n);
                ht_cuda.insert(insert_values, n);

                duration_serial[rep] = time_func([&]
                                                 { ht_serial.lookup(lookup_values, results, n); });
                duration_cuda[rep] = time_func([&]
                                               { ht_cuda.lookup(lookup_values, results, n); });
                printf("hash-func-num %-10d percent %-5d repeat-times %-10d serial time: %-10lf cuda time: %-10lf\n", num_funcs, percent * 10, rep, duration_serial[rep], duration_cuda[rep]);
            }

            double mean_serial = average(duration_serial, 5);
            double stddev_serial = standardDev(duration_serial, 5);

            double mean_cuda = average(duration_cuda, 5);
            double stddev_cuda = standardDev(duration_cuda, 5);

            Record record;
            record.put(num_funcs);
            record.put(percent);
            record.put(mean_serial);
            record.put(stddev_serial);
            serial_writer.insertRecord(record);

            Record record_cuda;
            record_cuda.put(num_funcs);
            record_cuda.put(percent);
            record_cuda.put(mean_cuda);
            record_cuda.put(stddev_cuda);
            cuda_writer.insertRecord(record_cuda);
        }
        delete[] insert_values;
        delete[] lookup_values;
        delete[] results;
    }

    serial_writer.write();
    cuda_writer.write();
}

void bench_task3_cuckoo()
{
    CsvWriter serial_writer("task3_serial.csv");
    CsvWriter cuda_writer("task3_cuda.csv");

    Record header;
    header.put("func_num");
    header.put("ratios");
    header.put("mean");
    header.put("stddev");

    serial_writer.setHeader(header);
    cuda_writer.setHeader(header);

    printf("task 3\n");
    uint32_t n = 1 << 24;
    uint32_t *insert_values = new uint32_t[n];
    generate_unique_random(insert_values, n);
    for (int num_funcs = 3; num_funcs <= 4; ++num_funcs)
    {
        float ratios[] = {1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1.02, 1.01};
        for (int ri = 0; ri < 12; ++ri)
        {
            double duration_serial[5];
            double duration_cuda[5];

            int size = ceil(ratios[ri] * n);
            for (int rep = 0; rep < 5; ++rep)
            {
                CuckooSerialHashTable ht_serial(size, 4 * clog2(n), num_funcs);
                CuckooCudaHashTable ht_cuda(size, 4 * clog2(n), num_funcs);
                int level_serial;
                int level_cuda;
                duration_serial[rep] = time_func([&]
                                                 { level_serial = ht_serial.insert(insert_values, n); });
                duration_cuda[rep] = time_func([&]
                                               { level_cuda = ht_cuda.insert(insert_values, n); });
                printf("hash-func-num %-10d ratios %-10lf repeat-times %-10d serial time: %-10lf rehash %-10d cuda time: %-10lf rehash %-10d\n", num_funcs, ratios[ri], rep, duration_serial[rep], level_serial, duration_cuda[rep], level_cuda);
            }
            double mean_serial = average(duration_serial, 5);
            double stddev_serial = standardDev(duration_serial, 5);

            double mean_cuda = average(duration_cuda, 5);
            double stddev_cuda = standardDev(duration_cuda, 5);

            Record record;
            record.put(num_funcs);
            record.put(ratios[ri]);
            record.put(mean_serial);
            record.put(stddev_serial);
            serial_writer.insertRecord(record);

            Record record_cuda;
            record_cuda.put(num_funcs);
            record_cuda.put(ratios[ri]);
            record_cuda.put(mean_cuda);
            record_cuda.put(stddev_cuda);
            cuda_writer.insertRecord(record_cuda);
        }
    }
    serial_writer.write();
    cuda_writer.write();
    delete[] insert_values;
}

void bench_task4_cuckoo()
{
    CsvWriter serial_writer("task4_serial.csv");
    CsvWriter cuda_writer("task4_cuda.csv");

    Record header;
    header.put("func_num");
    header.put("bound_mul");
    header.put("mean");
    header.put("stddev");

    serial_writer.setHeader(header);
    cuda_writer.setHeader(header);

    printf("task 4\n");
    int n = 0x1 << 24, size = ceil(1.4 * n);
    uint32_t *insert_values = new uint32_t[n];
    generate_unique_random(insert_values, n);
    for (int num_funcs = 3; num_funcs <= 4; ++num_funcs)
    {
        for (int bound_mul = 1; bound_mul <= 10; ++bound_mul)
        {

            double duration_serial[5];
            double duration_cuda[5];

            for (int rep = 0; rep < 5; rep++)
            {
                CuckooSerialHashTable ht_serial(size, bound_mul * clog2(n), num_funcs);
                CuckooCudaHashTable ht_cuda(size, bound_mul * clog2(n), num_funcs);
                int level_serial;
                int level_cuda;
                duration_serial[rep] = time_func([&]
                                                 { level_serial = ht_serial.insert(insert_values, n); });
                duration_cuda[rep] = time_func([&]
                                               { level_cuda = ht_cuda.insert(insert_values, n); });
                printf("hash-func-num %-10d bound-mul %-10d repeat-times %-10d serial time: %-10lf rehash %-10d cuda time: %-10lf rehash %-10d\n", num_funcs, bound_mul, rep, duration_serial[rep], level_serial, duration_cuda[rep], level_cuda);
            }
            double mean_serial = average(duration_serial, 5);
            double stddev_serial = standardDev(duration_serial, 5);

            double mean_cuda = average(duration_cuda, 5);
            double stddev_cuda = standardDev(duration_cuda, 5);

            Record record;
            record.put(num_funcs);
            record.put(bound_mul);
            record.put(mean_serial);
            record.put(stddev_serial);
            serial_writer.insertRecord(record);

            Record record_cuda;
            record_cuda.put(num_funcs);
            record_cuda.put(bound_mul);
            record_cuda.put(mean_cuda);
            record_cuda.put(stddev_cuda);
            cuda_writer.insertRecord(record_cuda);
        }
    }
    serial_writer.write();
    cuda_writer.write();
    delete[] insert_values;
}
void test_cuckoo_serial()
{
    CuckooSerialHashTable x(1000, 12, 5);
    test_hashtable("CuckooSerialHashTable", x);
}

void test_cuckoo_cuda()
{
    CuckooCudaHashTable x(1000, 12, 5);
    test_hashtable("CuckooCudaHashTable", x);
}