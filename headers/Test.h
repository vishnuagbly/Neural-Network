#ifndef TEST_H
#define TEST_H
#include <iostream>
using namespace std;

namespace testName {
void print(int k);
}  // namespace testName

#ifdef __has_include
#if __has_include(<stdio.h>)
#include <stdio.h>
namespace testName {
void print();
}  // namespace testName
#endif
#endif

#endif