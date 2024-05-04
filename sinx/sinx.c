#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "immintrin.h"

void sinx(int N, int terms, float * x, float * result)
{
    for(int i = 0; i < N; i++)
    {
        float value = x[i];
        float number = x[i] * x[i] * x[i];
        int denom = 6;
        int sign = -1;

        for(int j = 1; j <= terms; j++)
        {
            value += sign * number / denom;
            number *= x[i] * x[i];
            denom *= (2 * j + 2) * (2 * j + 3);
            sign *= -1;
        }
        result[i] = value;
    }
}

void sinx_avx_version1(int N, int terms, float * x, float * result)
{
    for(int i = 0; i < N / 8 * 8; i += 8)
    {
        __m256 value = _mm256_loadu_ps(x + i);          //load 8 x
        __m256 number = _mm256_mul_ps(value, value);    //number = value ^ 2
        __m256 number2 = _mm256_mul_ps(number, value);  //number2 = value ^ 3
        __m256 denom = _mm256_set1_ps(6.0f);            //denom = 6
        int sign = -1;

        for(int j = 1; j <= terms; j++)
        {
            value = _mm256_add_ps(value, _mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(sign), number2), denom));
            number2 = _mm256_mul_ps(number2, number);
            denom = _mm256_mul_ps(denom, _mm256_set1_ps((2.0f * j + 2.0f) * (2.0f * j + 3.0f)));
            sign *= -1;
        }

        _mm256_storeu_ps(result + i, value);
    }

    for(int i = N / 8 * 8; i < N; i++)
    {
        float value = x[i];
        float number = x[i] * x[i] * x[i];
        int denom = 6;
        int sign = -1;

        for(int j = 1; j <= terms; j++)
        {
            value += sign * number / denom;
            number *= x[i] * x[i];
            denom *= (2 * j + 2) * (2 * j + 3);
            sign *= -1;
        }
        result[i] = value;
    }
}

void sinx_avx_version2(int N, int terms, float * x, float * result)
{
    for(int i = 0; i < N / 8 * 8; i += 8)
    {
        __m256 value = _mm256_loadu_ps(x + i);          //load 8 x
        __m256 number = _mm256_mul_ps(value, value);    //number = value ^ 2
        __m256 number2 = _mm256_mul_ps(number, value);  //number2 = value ^ 3
        __m256 denom = _mm256_set1_ps(6.0f);            //denom = 6
        int sign = -1;

        for(int j = 1; j <= terms; j++)
        {
            value = _mm256_add_ps(value, _mm256_div_ps(number2, denom));
            if(sign == -1)
            {
                __m256 zero = _mm256_setzero_ps();
                value = _mm256_sub_ps(zero, value);
            }
            number2 = _mm256_mul_ps(number2, number);
            denom = _mm256_mul_ps(denom, _mm256_set1_ps((2.0f * j + 2.0f) * (2.0f * j + 3.0f)));
            sign *= -1;
        }

        _mm256_storeu_ps(result + i, value);
    }

    for(int i = N / 8 * 8; i < N; i++)
    {
        float value = x[i];
        float number = x[i] * x[i] * x[i];
        int denom = 6;
        int sign = -1;

        for(int j = 1; j <= terms; j++)
        {
            value += sign * number / denom;
            number *= x[i] * x[i];
            denom *= (2 * j + 2) * (2 * j + 3);
            sign *= -1;
        }
        result[i] = value;
    }
}

int main()
{
    const int N = 100000;
    const int terms = 5;
    float *x = (float *)malloc(N * sizeof(float));
    float *result_sinx = (float *)malloc(N * sizeof(float));
    float *result_sinx_avx = (float *)malloc(N * sizeof(float));

    // Generate random values for x
    for (int i = 0; i < N; ++i) {
        x[i] = (float)rand() / RAND_MAX;
    }

    clock_t start, end;
    double cpu_time_used;

    // Test sinx
    start = clock();
    sinx(N, terms, x, result_sinx);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time for sinx: %f seconds\n", cpu_time_used);

    // Test sinx_avx
    start = clock();
    sinx_avx_version1(N, terms, x, result_sinx_avx);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time for sinx_avx: %f seconds\n", cpu_time_used);

    // Compare results
    int mismatch_count = 0;
    for (int i = 0; i < N; ++i) {
        if (result_sinx[i] != result_sinx_avx[i]) {
            printf("Mismatch at index %d: sinx=%f, sinx_avx=%f\n", i, result_sinx[i], result_sinx_avx[i]);
            mismatch_count++;
        }
    }

    if (mismatch_count == 0) {
        printf("Results match.\n");
    } else {
        printf("Results do not match. Number of mismatches: %d\n", mismatch_count);
    }

    free(x);
    free(result_sinx);
    free(result_sinx_avx);

    return 0;
}
