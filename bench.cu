#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "xxhash.hcu"
#include "utils.hcu"
#include "cuckoo_serial.hcu"

#include <string>
void test_cuckoo_serial();
void test_cuckoo_cuda_native();
void bench_task1_cuckoo_serial();

int main()
{
    test_xxhash();
    test_cuckoo_serial();
}

void bench_task1_cuckoo_serial(){
    
}

void test_cuckoo_serial()
{
    CuckooSerialHashTable x(1000, 12, 3);
    test_hashtable("CuckooSerialHashTable", x);
}