#pragma once

#define		M			1046527
#define		MAX			500000
#define		SENTINEL	2000000000

void maximumProfit(int* A, int n, int& maxPrifit);

//sort
void insertionSort(int A[], int n);
void bubbleSort(int A[], int n);
void selectionSort(int A[], int n);
void insertionSortInterval(int A[], int n, int g);
void shellSort(int A[], int n);

//advanced sorting algorithms
//merge sort
void mergeSort(int A[], int left, int right);
void merge(int A[], int left, int mid, int right);
//quick sort
int partition(int A[], int left, int right);
void quickSort(int A[], int left, int right);

//stack
void reversePolishNotation(char* A, int n, int& res);

//queue
void cyclicScheduling();
//list

//ALDS1_3_D: areas on the cross-section diagram
void areasCounting();

//search
int linearSearch(int A[], int n, int key);
int binarySearch(int A[], int n, int key);
//search :Hash
int h1(int key);
int h2(int key);
int hashSearch(int A[], int n, int key);
int hashInsert(int A[], int n, int key);
//search :STL:lower_bound
int lower_boundSearchSTL(int A[], int n, int key);

//ALDS1_4_D:allocation
typedef long long	llong;
int check(llong T[], int n, int k, llong P);
llong allocationMinCarrierCapcity(llong T[], int n, int k);

//recursive & divide and conquer (递归和分治法)
int findMaximum(int A[], int leftLocation, int rightLocation);

//ALDS1_5_A:exhaustive search
bool solve(int A[], int n, int i, int m);

