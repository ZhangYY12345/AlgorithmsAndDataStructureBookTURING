// AlgorithmAndDataStructBookTURING.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "methodsAlgorithmFun.h"
#include <iostream>

int main()
{
	int n, k;
	llong T[100000];
	std::cin >> n >> k;
	for(int i = 0; i < n; i++)
	{
		std::cin >> T[i];
	}
	llong ans = allocationMinCarrierCapcity(T, n, k);
	std::cout << ans << std::endl;
	return 0;
}

