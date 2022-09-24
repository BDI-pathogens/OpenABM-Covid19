
  // Use a range of values for each function in turn, and return the results as csv to STDOUT

	// write CSV common header
  printf("library,test,input1,input2,input3,result\n");

	// NOTE tests without an input1 are the TIME TAKEN for that test to complete in nanoseconds

	// TODO output time taken per test too, and the library used
	clock_t start, end;
	double cpu_time_nsec;

	// rng uniform, default seed, 99 iterations
	generator* gen = rng_alloc();
	start = clock();
	double result;
  for (size_t i = 0;i <= 99; ++i) {
		result = rng_uniform(gen);
    printf("%s,rng_uniform_defaultseed,%ld,,,%12f\n",library,i,result);
	}
	end = clock();
	cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
	printf("%s,rng_uniform_defaultseed,,,,%12f\n",library,cpu_time_nsec);
	rng_free(gen);

	// rng uniform, custom seed, 99 iterations
	gen = rng_alloc();
	rng_set(gen, 74656);
	start = clock();
  for (size_t i = 0;i <= 99; ++i) {
		result = rng_uniform(gen);
    printf("%s,rng_uniform_74656seed,%ld,,,%12f\n",library,i,result);
	}
	end = clock();
	cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
	printf("%s,rng_uniform_74656seed,,,,%12f\n",library,cpu_time_nsec);
	rng_free(gen);

	// rng uniform int, default seed, 99 iterations, start at 1
	gen = rng_alloc();
	int resultint;
	start = clock();
  for (size_t i = 1;i <= 99; ++i) {
		resultint = rng_uniform_int(gen,i);
    printf("%s,rng_uniform_int,%ld,,,%d\n",library,i,resultint);
	}
	end = clock();
	cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
	printf("%s,rng_uniform_int,,,,%12f\n",library,cpu_time_nsec);
	rng_free(gen);

	// rng bernoulli, default seed, 99 iterations, p 0.01-0.99
	gen = rng_alloc();
	unsigned int resultuint;
	double indbl = 0.0;
	start = clock();
  for (size_t i = 1;i <= 99; ++i) {
		indbl += 0.01;
		resultuint = ran_bernoulli(gen,indbl);
    printf("%s,ran_bernoulli,%f,,,%d\n",library,indbl,resultuint);
	}
	end = clock();
	cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
	printf("%s,ran_bernoulli,,,,%12f\n",library,cpu_time_nsec);
	rng_free(gen);

	// random gamma, shape_par(i1) 1-10, scale_par(i2) 1-10
	gen = rng_alloc();
	start = clock();
  for (int i1 = 1;i1 <= 10; ++i1) {
  	for (int i2 = 1;i2 <= 10; ++i2) {
			result = ran_gamma(gen,i1,i2);
			printf("%s,ran_gamma,%d,%d,,%12f\n",library,i1,i2,result);
		}
	}
	end = clock();
	cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
	printf("%s,ran_gamma,,,,%12f\n",library,cpu_time_nsec);
	rng_free(gen);

	// ran exponential, default seed, 99 iterations, 0.01-0.99
	gen = rng_alloc();
	indbl = 0.0;
	start = clock();
  for (size_t i = 1;i <= 99; ++i) {
		indbl += 0.01;
		result = ran_exponential(gen,indbl);
    printf("%s,ran_exponential,%f,,,%12f\n",library,indbl,result);
	}
	end = clock();
	cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
	printf("%s,ran_exponential,,,,%12f\n",library,cpu_time_nsec);
	rng_free(gen);
	

	// ran negative binomial, default seed, p=0.01-0.99, n=1-10001 step 1000
	gen = rng_alloc();
	double indbl2 = 0.0;
	start = clock();
	for (size_t n = 1;n <= 10001; n += 1000) {
		indbl2 += 1.0; // increase 'n' which is taken as a double
		indbl = 0.0; // reset each iteration
		for (size_t i = 1;i <= 99; ++i) {
			indbl += 0.01;
			resultuint = ran_negative_binomial(gen, indbl, indbl2);
			printf("%s,ran_negative_binomial,%f,%f,,%d\n",library,indbl,indbl2,resultuint);
		}
	}
	end = clock();
	cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
	printf("%s,ran_negative_binomial,,,,%12f\n",library,cpu_time_nsec);
	rng_free(gen);
	
	// ran discrete default seed, i=1-100, K=array of values not adding to p=1.0
	// generate discrete probabilities multiplied by a factor (to simulate not adding up to p=1)
	gen = rng_alloc();
	const size_t K = 10000;
	double P[K];
	double multiplier = 80000.0;
	for (size_t pi = 0;pi < 10000; ++pi) {
		P[pi] = multiplier * rng_uniform(gen);
	}
	rng_free(gen);
	// now use the P array
	gen = rng_alloc();
	start = clock();
	for (size_t i = 1;i <= 100; ++i) {
		resultuint = ran_discrete(gen, K, P);
		printf("%s,ran_discrete,%ld,,,%d\n",library,i,resultuint);
	}
	end = clock();
	cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
	printf("%s,ran_discrete,,,,%12f\n",library,cpu_time_nsec);
	rng_free(gen);

	// Now do random shuffle, ten operations
	gen = rng_alloc();
	start = clock();
	for (size_t i = 1;i <= 10; ++i) {
		ran_shuffle(gen, P, K, sizeof(double));
		printf("%s,ran_shuffle,%ld,,,%f\n",library,i,P[0]);
	}
	end = clock();
	cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
	printf("%s,ran_shuffle,,,,%12f\n",library,cpu_time_nsec);
	rng_free(gen);


	// cdf gamma P, a=1-100 step 5,b=1-101 step 5, p=0.01-0.99 step 0.01
	indbl2 = 0.0;
	double p = 0.00;
	start = clock();
	for (size_t a = 1;a <= 3; ++a) {
		indbl = a; // increase 'n' which is taken as a double
		for (size_t b = 1;b <= 3; ++b) {
			indbl2 = b; // reset each iteration
			p = 0.00;
			for (size_t i = 1;i <= 99; ++i) {
				p += 0.01;
				result = cdf_gamma_P(i, indbl, indbl2);
				printf("%s,cdf_gamma_P,%ld,%f,%f,%12f\n",library,i,indbl,indbl2,result);
			}
		}
	}
	end = clock();
	cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
	printf("%s,cdf_gamma_P,,,,%12f\n",library,cpu_time_nsec);

	// cdf gamma Pinv, a=1-100 step 5,b=1-101 step 5, p=0.01-0.99 step 0.01
	indbl2 = 0.0;
	p = 0.00;
	start = clock();
	for (size_t a = 1;a <= 3; ++a) {
		indbl = a; // increase 'n' which is taken as a double
		for (size_t b = 1;b <= 5; ++b) {
			indbl2 = b; // reset each iteration
			p = 0.00;
			for (size_t i = 1;i <= 99; ++i) {
				p += 0.01;
				result = cdf_gamma_Pinv(p, indbl, indbl2);
				printf("%s,cdf_gamma_Pinv_plt0,%f,%f,%f,%12f\n",library,p,indbl,indbl2,result);
			}
		}
	}
	end = clock();
	cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
	printf("%s,cdf_gamma_Pinv_plt0,,,,%12f\n",library,cpu_time_nsec);

	/*
	// NOTE GSL only support 1 < p < 0
	// cdf gamma Pinv, using i not p, a=1-3,b=1-5, i=1-99
	indbl2 = 0.0;
	for (size_t a = 1;a <= 3; ++a) {
		indbl = a; // increase 'n' which is taken as a double
		for (size_t b = 1;b <= 5; ++b) {
			indbl2 = b; // reset each iteration
			for (size_t i = 1;i <= 99; ++i) {
				result = cdf_gamma_Pinv(i, indbl, indbl2);
				printf("%s,cdf_gamma_Pinv_pgt0,%ld,%f,%f,%12f\n",library,i,indbl,indbl2,result);
			}
		}
	}
	*/

	// cdf_exponential_Pinv, a=1-100 step 5,b=1-101 step 5, p=0.01-0.99 step 0.01
	p = 0.00;
	start = clock();
	for (size_t a = 1;a <= 101; a += 5) {
		indbl = a; // increase 'n' which is taken as a double
		p = 0.00;
		for (size_t i = 1;i <= 99; ++i) {
			p += 0.01;
			result = cdf_exponential_Pinv(p, indbl);
			printf("%s,cdf_exponential_Pinv,%f,%f,,%12f\n",library,p,indbl,result);
		}
	}
	end = clock();
	cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
	printf("%s,cdf_exponential_Pinv,,,,%12f\n",library,cpu_time_nsec);


	// inv_incomplete_gamma_p,n=1 to 1001 step 100, p=0.01-0.99 step 0.01
	start = clock();
	for (size_t n = 1;n <= 1001; n += 100) {
		p = 0.00;
		for (size_t i = 1;i <= 99; ++i) {
			p += 0.01;
			result = inv_incomplete_gamma_p(p, n);
			printf("%s,inv_incomplete_gamma_p,%f,%ld,,%12f\n",library,p,n,result);
		}
	}
	end = clock();
	cpu_time_nsec = (((double)(end - start)) * 1000000) / CLOCKS_PER_SEC;
	printf("%s,inv_incomplete_gamma_p,,,,%12f\n",library,cpu_time_nsec);
