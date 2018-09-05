#include "stdafx.h"
#include "methodsAlgorithmFun.h"
#include <algorithm>
#include <stack>
#include <unordered_set>
#include <queue>
#include <xlocale>
#include <iostream>
#include <string>

/**
 * \brief to find the maxmum profit in the array A[n]:that is to find the minmum value in the left part of the jth number
 * \param A :the input integer array
 * \param n :the size of the input array
 * \param maxPrifit :the to be found maximum profit in the array
 */
void maximumProfit(int* A, int n, int& maxPrifit)
{
	int maxValue = -2000000000;
	int minValue = A[0];

	for(int i = 1; i < n; i++)
	{
		maxValue = std::max(maxValue, A[i] - minValue);
		minValue = std::min(minValue, A[i]);
	}
	maxPrifit = maxValue;
}


//sort
/**
 * \brief using insertion sort algorithm to sort the input array A[n]:the value from small to big: < < <
 *         insertion sort:  1)consider the begining element of the array is sorted
 *							2)take the following steps until all the elemnts in the array are sorted:
 *								a)take the begining element of the unsorted subarray as variable v
 *								b)let the elements in the sorted subarray, which is bigger than than the variable v, move one step backword 
 *								c)put the variable v in the vacancy
 *			insertion sort algorithm is quick when process relatively sorted array, Stable Sort
 * \tparam T :the type name of the input array A
 * \param A :the input array (pointer)
 * \param n :the size of the input array A
 */
template <typename T>
void insertionSort(T A[], int n)
{
	for(int i = 0; i < n; i++)
	{
		int v = A[i];
		int j = i - 1;
		while(j >= 0 && A[j] > v)
		{
			A[j + 1] = A[j];
			j--;
		}
		A[j + 1] = v;
	}
}

void insertionSort(int A[], int n)
{}

/**
 * \brief	using bubble sort algorithm to sort the input array A[n]:the value from small to big: < < <
 *			bubble sort:take the following steps until all the adjacent elemnts in the array are sorted,
 *						that is to say for all the possible i,we have A[i] < A[i+1]:
 *							1)start from the end of the array,compare evert adjacent two elements,
 *							  if A[i] > A[i+1], exchange the two element.
 *			bubble sort algorithm also is a Stable Sort algorithm.
 * \tparam T :the type name of the input array A
 * \param A :the input array (pointer)
 * \param n :the size of the input array A
 */
template <typename T>
void bubbleSort(T A[], int n)
{
	bool flag = true;
	for(int i = 0; flag; i++)
	{
		flag = false;
		for(int j = n - 1; j >= i + 1; j--)
		{
			if(A[j] < A[j - 1])
			{
				std::swap(A[j], A[j - 1]);
				flag = true;
			}
		}
	}
}

void bubbleSort(int A[], int n)
{}

/**
 * \brief	using selection sort algorithm to sort the input array A[n]:the value from small to big: < < <
 *			selection sort:take the following steps for n-1 times:
 *						   1)find the position of the minimum value, minj, in the unsorted subarray
 *						   2)exchange the value in position minj and the beginning value of the unsorted array
 *			selection sort algorithm is a unstable sort algorithm
 *			
 *			//to judge if a sort algorithm is an stable sort algorithm,
 *			//we can compare the result of the target sort algorithm to the result of a known stable sort algorithm,
 *			//and check if the two results are the same
 * \tparam T :the type name of the input array A
 * \param A :the input array (pointer)
 * \param n :the size of the input array A
 */
template <typename T>
void selectionSort(T A[], int n)
{
	for(int i = 0; i < n - 1; i++)
	{
		int minj = i;
		for(int j = i; j < n; j++)
		{
			if(A[j] < A[minj])
			{
				minj = j;
			}
		}

		T exV = A[i];
		A[i] = A[minj];
		A[minj] = exV;
	}
}

void selectionSort(int A[], int n)
{}

//shell sort
/**
 * \brief	insertion sort on array A[n] with moving step set as g 
 *			in this function the to be sorted subarray is 
 *				A[0],A[g];
 *				..., A[g+1];
 *				...,A[g+2];
 *				......
 *				...,A[g+k];
 *				while g+k <= n
 *														
 *			when g=1, this is the normal insertion sort process
 * \tparam T :the type name of the input array A
 * \param A :the input array (pointer)
 * \param n :the size of the input array A
 * \param g :the interval value
 */
template<typename T>
void insertionSortInterval(T A[], int n, int g)
{
	for(int i = g; i < n; i++)
	{
		int v = A[i];
		int j = i - g;
		while(j >= 0 && A[i] > v)
		{
			A[j + g] = A[j];
			j -= g;
		}
		A[j + g] = v;
	}
}

void insertionSortInterval(int A[], int n, int g)
{}

/**
 * \brief	shell sort is the loop of insertionSortInterval(T A[], int n, int g),and the value of g is reduced after every loop
 *				the sequence value of g :G[0], G[1],...
 *				the selection of the sequence of g varies.
 *			insertion sort algorithm can sort arraies which are relatively sorted, 
 *			and shell sort algorithm takes advantage of this feature of the insertion sort algorithm.
 *			
 *			In this function,G[i+1] = 3*G[i] + 1, and the result time complexity of the shell sort is O(N^1.25).
 *			Notice that, when the input array is well sorted, shell sort algorithm will be quite slow.
 * \tparam T :the type name of the input array A
 * \param A :the input array (pointer)
 * \param n :the size of the input array A
 */
template <typename T>
void shellSort(T A[], int n)
{
	std::vector<int> G;
	for(int h = 1; ;)
	{
		if (h > n)
			break;
		G.push_back(h);
		h = 3 * h + 1;
	}

	for(int i = G.size() - 1; i >= 0; i--)
	{
		insertionSortInterval(A, n, G[i]);
	}
}

void shellSort(int A[], int n)
{}

//advanced sorting algorithms
/**
 * \brief	merge sorting
 *			take the whole array as the object and use merge sorting to sort the array
 *			merge sorting:	1)divide the given array into two parts,each contain n/2 elements:(the left part and the right part)
 *							2)using mergeSort() to sort the two subarrays <using the function mergeSort(),recursion>
 *							3)using merge() to mix the two subarrays and get the final sorted array <conquer>
 *			when we have only one element,we execute merge().
 *			merge sorting is a stable sorting algorithm.
 * \param A :the array to be sorted
 * \param left :the left index of the sorting range
 * \param right :the right index of the sorting range < A[right] is not included >
 */
void mergeSort(int A[], int left, int right)
{
	if(left + 1 < right)
	{
		int mid = (left + right) / 2;
		mergeSort(A, left, mid);
		mergeSort(A, mid, right);
		merge(A, left, mid, right);
	}
}

void merge(int A[], int left, int mid, int right)
{
	int arrayLeft[MAX / 2 + 2];
	int arrayRight[MAX / 2 + 2];

	int nleft = mid - left;
	int nright = right - mid;

	for(int i = 0; i < nleft; i++)
	{
		arrayLeft[i] = A[left + i];
	}

	for(int i = 0; i < nright; i++)
	{
		arrayRight[i] = A[mid + i];
	}

	arrayLeft[nleft] = arrayRight[nright] = SENTINEL;

	int i = 0, j = 0;
	for(int k = left; k < right; k++)
	{
		if(arrayLeft[i] <= arrayRight[j])
		{
			A[k] = arrayLeft[i];
			i++;
		}
		else
		{
			A[k] = arrayRight[j];
			j++;
		}
	}
}


/**
 * \brief	divide the array A[left]...A[right] into two part:A[left]...A[q-1] , A[q+1]...A[right], and return the index q
 *			besides, each element in the part A[left]...A[q-1] is smaller than or equal to A[q]
 *					 each element in the part A[q+1]...A[right] is larger than A[q]
 *			notice: we use A[right] as the base of comparison to divide the array
 * \param A :the array to divide
 * \param left :the left index of the range of array to be divided
 * \param right :the right index of the range of array to be divided, A[right] is included in the subarray
 * \return :the index q
 */
int partition(int A[], int left, int right)
{
	int x = A[right];
	int i = left - 1;

	for(int j = left; j < right; j++)
	{
		if(A[j] <= x)
		{
			i++;
			int t = A[i];
			A[i] = A[j];
			A[j] = t;
		}
	}
	int t = A[i + 1];
	A[i + 1] = A[right];
	A[right] = t;
	return i + 1;
}

/**
 * \brief	quick sorting
 *			take the whole array as the object and use merge sorting to sort the array
 *			quick sorting:	1)using partition() to divide the object array into two parts:the left part and the right part
 *						2)using quickSort() to sort the left part of the subarray
 *						3)using quichSort() to sort the right part of the subarray
 *			quick sorting is an unstable sorting algorithm.
 *			different from merge sorting, quick sorting realizes the sorting in the partition period
 *			(merge sorting realizes sorting in the merge period)
 * \param A :the array to divide
 * \param left :the left index of the range of array to be sorted
 * \param right :the right index of the range of array to be sorted, A[right] is included
 */
void quickSort(int A[], int left, int right)
{
	if(left < right)
	{
		int q = partition(A, left, right);
		quickSort(A, left, q - 1);
		quickSort(A, q + 1, right);
	}
}


/**
 * \brief	reverse Polish notation
 *			using stack to realize the calculation
 * \param A :the input array which store the equation in reverse Polish notation to be caluculated
 * \param n :the size of the input array
 * \param res :the calculation result of the equation
 */
void reversePolishNotation(char* A, int n, int& res)
{
	std::stack<int> stkCal;
	std::string number = "";
	int a, b;
	for(int i = 0; i < n; i++)
	{
		if(A[i] == '+')
		{
			if(!number.empty())
			{
				stkCal.push(atoi(number.c_str()));
				number.clear();
			}

			a = stkCal.top();
			stkCal.pop();
			b = stkCal.top();
			stkCal.pop();
			stkCal.push(b + a);
		}
		else if(A[i] == '-')
		{
			if (!number.empty())
			{
				stkCal.push(atoi(number.c_str()));
				number.clear();
			}

			a = stkCal.top();
			stkCal.pop();
			b = stkCal.top();
			stkCal.pop();
			stkCal.push(b - a);
		}
		else if(A[i] == '*')
		{
			if (!number.empty())
			{
				stkCal.push(atoi(number.c_str()));
				number.clear();
			}

			a = stkCal.top();
			stkCal.pop();
			b = stkCal.top();
			stkCal.pop();
			stkCal.push(b * a);
		}
		else if(A[i] == '/')
		{
			if (!number.empty())
			{
				stkCal.push(atoi(number.c_str()));
				number.clear();
			}

			a = stkCal.top();
			stkCal.pop();
			b = stkCal.top();
			stkCal.pop();
			stkCal.push(b / a);
		}
		else if(A[i] == ' ')
		{
			if (!number.empty())
			{
				stkCal.push(atoi(number.c_str()));
				number.clear();
			}
		}
		else
		{
			number += A[i];
		}
	}

	res = stkCal.top();
	stkCal.pop();
}

/**
 * \brief	cyclic scheduling, deal with the tasks one by one with each task takes qms at one time,
 *			if the task has not finished yet,move the taks to the end of the queue,and start dealing with the next task
 */
void cyclicScheduling()
{
	int minTime = 100;
	int sumTime = 0;
	int number = 0;

	std::queue<std::pair<std::string, int>> QTask;

	std::cout << "Please input the size of the input array（number） and the minimum time in each processing(minTime):\n";
	std::cin >> number >> minTime;

	std::string name;
	int time;
	for(int i = 0; i < number; i++)
	{
		std::cin >> name;
		std::cin >> time;
		QTask.push(std::make_pair(name, time));
	}
	std::pair<std::string, int> p;

	while(!QTask.empty())
	{
		p = QTask.front();
		QTask.pop();

		int c = std::min(minTime, p.second);
		p.second -= c;
		sumTime += c;
		if(p.second > 0)
		{
			QTask.push(p);
		}
		else
		{
			std::cout << p.first << "\t" << sumTime << std::endl;
		}
	}
}

void areasCounting()
{
	std::stack<int>S1;
	std::stack<std::pair<int, int>> S2;

	char ch;
	int sum = 0;
	for(int i = 0; std::cin>>ch; i++)
	{
		if(ch == '\\')
		{
			S1.push(i);
		}
		else if(ch == '/' && S1.size() > 0)
		{
			int j = S1.top();
			S1.pop();
			sum += i - j;
			int a = i - j;
			while(S2.size() > 0 && S2.top().first > j)
			{
				a += S2.top().second;
				S2.pop();
			}
			S2.push(std::make_pair(j, a));
		}
	}
}

/**
 * \brief 含有标记的线性搜索：引入的标记可以将算法效率提高常数倍，算法复杂度为O(n)
 * \param A 
 * \param n 
 * \param key 
 * \return 
 */
int linearSearch(int A[], int n, int key)
{
	int i = 0;
	A[n] = key;
	while(A[i] != key)
	{
		i++;
	}
	if(i == n)
	{
		return -1;
	}

	return i;
}

/**
 * \brief	using binary search to find the key in the sorted array A[n]
 *			if the array A[n] is not sorted,we can sort the array first and then use binary search method to do the searching
 * \param A :the array to be searched,which is supposed to be a sorted array
 * \param n 
 * \param key 
 * \return 
 */
int binarySearch(int A[], int n, int key)
{
	int left = 0; 
	int right = n;
	int mid;
	while(left < right)
	{
		mid = (left + right) / 2;
		if (key == A[mid])
		{
			return mid;		
		}
		if(key > A[mid])
		{
			left = mid + 1;		//to search the right part of the array
		}
		else if(key < A[mid])
		{
			right = mid;		//to search the left part of the array
		}
	}
	return -1;
}

/**
 * \brief	散列表由容纳n个元素的数组A，及根据数据关键字决定数组下标的函数（散列函数）共同组成。
 *			使用散列法搜索时，要将数据的关键字输入该函数，由该函数决定数据在数组中的位置。
 *			使用散列法时，可能存在的一个问题是：“冲突”，即不同的关键字输入散列函数得到的散列值相同。
 *			开放地址法是解决这类冲突的常用手段之一。
 *			在双散列结构中使用的开放地址法：
 *			在双散列结构中，一旦出现冲突，程序就会调用第二个散列函数来求解散列值：H(k) = h(k,i) = (h1(k) + i * h2(k)) mod m
 *			散列函数h(k,i)具有关键字k和整数i两个参数，这里i是发生冲突后计算下一个散列值的次数，即只要散列函数H(k)起了冲突，
 *			就会调用h(k,0)、h(k,1)、h(k,2)…，直到不发生冲突为止，然后返回这个h(k,i)的值作为散列值。
 *			注：因为下标每次移动h2(k)个位置，所以必须保证数组A[n]的长度n与h2(k)互质，否则会出现无法生成下标的情况。
 * \param A :assume that when A[i] = 0, this means this position in the array is empty
 * \param n 
 * \param key 
 * \return 
 */
int h1(int key)
{
	return key % M;
}

int h2(int key)
{
	return 1 + (key % (M - 1));
}

int hashSearch(int A[], int n, int key)
{
	int h = 0;
	int i = 0;
	while(true)
	{
		h = (h1(key) + i * h2(key)) % M;
		if (A[h] == key)
		{
			return h;
		}
		if (A[h] == 0 || h >= n)
		{
			return -1;
		}
		i++;
	}
}

int hashInsert(int A[], int n, int key)
{
	int i = 0;
	int h = 0;
	while(true)
	{
		h = (h1(key) + i * h2(key)) % M;
		if (A[h] == 0)
		{
			A[h] = key;
			return h;
		}
		i++;
	}
}

/**
 * \brief	using lower_bound() function in STL to find the key in the sorted array A[n]
 *			if the array A[n] is not sorted,we can sort the array first and then use binary search method to do the searching
 *			lower_bound() return a iterator which point to the first element which is >=value
 *			PS:value is the third parameter of function lower_bound()
 * \param A 
 * \param n 
 * \param key 
 * \return 
 */
int lower_boundSearchSTL(int A[], int n, int key)
{
	int* pos = std::lower_bound(A, A + n, key);	//the first two parameters are used to specify searching range,the third parameter is the key value to search for
	int arrayIndex = -1;
	if(*pos == key)
	{
		arrayIndex = std::distance(A, pos);		//return the distance between the two pointer A ,pos
	}
	return arrayIndex;
}

/**
 * \brief checking how much goods k trucks can take, the maximum carrier capacity of which is P
 * \param T :record the weight of each goods
 * \param n :means there are n goods to be carried
 * \param k :means we have k trucks
 * \param P :the maximum carrier capacity
 * \return :the maximum number of goods k trucks can take, with the maximum carrier capacity of which is P
 */
int check(llong T[], int n, int k, llong P)
{
	int i = 0;
	for(int j = 0; j < k; j++)
	{
		llong s = 0;
		while(s + T[i] <= P)
		{
			s += T[i];
			i++;
			if(i == n)
			{
				return n;
			}
		}
	}
	return i;
}

/**
 * \brief using binary search to find the minmum maximum carrier capcity of a truck(this parameter of each truck is the same)
 * \param T :record the weight of each goods
 * \param n :means there are n goods to be carried
 * \param k :means we have k trucks
 * \return :the minmum maximum carrier capcity of a truck
 */
llong allocationMinCarrierCapcity(llong T[], int n, int k)
{
	llong left = 0;
	llong right = 100000 * 10000;  //货物数 * 1件货物的最大重量
	llong mid;

	while(right - left > 1)
	{
		mid = (left + right) / 2;
		int v = check(T, n, k, mid);  //检查P == mid时能装多少货物
		if ( v >= n)
		{
			right = mid;
		}
		else
		{
			left = mid;
		}
	}

	return right;
}

/**
 * \brief find the maximum value in the range from A[left] to A[right-1] in the array A[]
 * \param A :the origin array to search
 * \param leftLocation :the left index of the search range
 * \param rightLocation :the right index of the search range
 * \return 
 */
int findMaximum(int A[], int leftLocation, int rightLocation)
{
	int maxValue = 0;
	int mid = (leftLocation + rightLocation) / 2;
	if(leftLocation == rightLocation - 1)
	{
		maxValue = A[leftLocation];
	}
	else
	{
		int u = findMaximum(A, leftLocation, mid);		//递归求解前半部分的局部问题
		int v = findMaximum(A, mid, rightLocation);		//递归求解后半部分的局部问题
		maxValue = std::max(u, v);		//	conquer
	}

	return maxValue;
}

/**
 * \brief	using exhaustive search to find if we can find the combination of elements in array A[n] 
 *			to make the sum of these elements equals to m.
 *			this function is to indicate if we can do these with the selection range of elements 
 *			is from A[i+1] to A[n-1] in the array A[n].
 *			
 *			when we are solving such problem,we just need to call:solve(A, n, 0, m);
 * \param A :the array to search:A[n]
 * \param n 
 * \param i :the starting index(i+1) of searching elements in the array A[n]
 * \param m :the integer to sum for
 * \return :indicate if we can find the combination
 */
bool solve(int A[], int n, int i, int m)
{
	if(m == 0)
	{
		return true;
	}

	if(i >= n)
	{
		return false;
	}

	bool res = solve(A, n, i + 1, m) || solve(A, n, i + 1, m - A[i]);
	return res;
}
