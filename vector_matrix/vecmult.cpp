#include <iostream>
#include <immintrin.h>
#include <fstream>
#include <chrono>

int main() {

    std::fstream inf("vecmult.in");
    std::fstream ouf("vecmult.out");
    int n;
    inf >> n;
    int * v1 = (int *)aligned_alloc(32, n * sizeof(int));
    int * v2 = (int *)aligned_alloc(32, n * sizeof(int));
    int * res = (int *)aligned_alloc(32, n * sizeof(int));
    
    for(int i = 0; i < n; i++)
    {
        inf >> v1[i];
    }
    for(int i = 0; i < n; i++)
    {
        inf >> v2[i];
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    __m256i * v11 = (__m256i*)v1;
    __m256i * v12 = (__m256i*)v2;
    __m256i * r = (__m256i*)res;

    for(int i = 0; i < n / 8; i++)
    {
        *r = _mm256_mullo_epi32(*v11, *v12);
        r++; v11++; v12++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    for (int i = 0; i < n; ++i) {
        ouf << res[i] << " ";
    }

    free(v1);
    free(v2);
    free(res);

    std::cout << "After AVX acceleration, execution time taken: " << duration.count() << " milliseconds" << std::endl;
    return 0;
}