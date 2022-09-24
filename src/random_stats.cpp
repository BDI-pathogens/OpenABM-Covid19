#include <math.h>

#include "random.h"

// Includes the stats and gcem modules
#include "stats.hpp"

void rng_initialise() {
  // Not required for the stats module
}

struct generator * rng_alloc() {
  generator* newgen = new generator; // C++ allocation
#ifdef GSL_COMPAT
  newgen->rng.seed(4357); // to seed the same as GSL's default
#endif
  return newgen;
}

void rng_free( generator * gen ) {
  if (gen) {
    delete (gen); // C++ deallocation
  }
}

// Works the same as GSL
void rng_set( generator * gen, long seed ) {
  gen->rng.seed(seed);
}

// Works the same as GSL
double rng_uniform( generator * gen) {
#ifdef GSL_COMPAT
  return stats::runif( 0.0, 1.0, gen->rng );
#else
  return stats::runif( 0.0, 1.0, gen->rng );
#endif
}

// Works the same as GSL
int rng_uniform_int( generator * gen, long p ) {
#ifdef GSL_COMPAT
  return stats::runif( 0.0, (float)p, gen->rng );
#else
  return stats::runif( 0l, p, gen->rng );
#endif
}

// Works the same as GSL
unsigned int ran_bernoulli( generator * gen, double p ) {
#ifdef GSL_COMPAT
  return static_cast<unsigned int>(stats::rbern( (float)p, gen->rng ));
#else
  return static_cast<unsigned int>(stats::rbern( p, gen->rng ));
#endif
}

// Note: rgamma in stats is more accurate than GSL. It uses rnorm not a ziggurat approach,
//       and uses log rather than an approximation in code. So slightly different results
//       to GSL should be expected.
double ran_gamma( generator * gen, double a, double b ) {
#ifdef GSL_COMPAT
  return stats::rgamma((float)a, (float)b, gen->rng);
#else
  return stats::rgamma(a, b, gen->rng);
#endif
}

// Works the same as GSL
double ran_exponential( generator * gen, double mu ) {
  double p = rng_uniform(gen);
  return -mu * log1p (-p);
  // return stats::rexp((float)mu, gen->rng); // Works very differently
}

// Note: rpois uses an unsigned long long int internally, and thus uses more bits from
//       the RNG than GSL does by default, even when specifying float as the input type.
//       This is a methodological difference which won't result in worse random generation
//       but may cost in performance terms.
unsigned int ran_negative_binomial( generator * gen, double p, double n ) {
  double x = ran_gamma( gen, n, 1.0 );
#ifdef GSL_COMPAT
  unsigned int k = stats::rpois( (float)(x*(1-p)/p), gen->rng );
#else
  unsigned int k = stats::rpois( (x*(1-p)/p), gen->rng );
#endif
  return k;
}

// Note: GSL uses Walker's Alias Method from Knuth's work:
//       https://en.wikipedia.org/wiki/Alias_method
//       Whereas below we directly calculate the exact probability each time
//       GSL pre-processes the table, but in OpenABM-Covid19's use we only
//       ever use this processed table once, making it less performant.
//       So here we choose accuracy and speed over GSL's preprocessed approach
//       This generates indexes almost the same as GSL, but more accurately
//       in terms of cumulative probability for each index position. This
//       algorithm is therefore less biased.
size_t ran_discrete( generator * gen, size_t K, const double *P ) {
  // Add up total of probability space for all P's
  double total = 0.0;
  std::size_t i = 0;
  for (;i < K; ++i) {
    total += P[i];
  }
  double scale = 1.0 / total; // turn future division into multiplication (faster generally)
  // Generate random number from generator
  double rnd = rng_uniform(gen);
  double cumulative = 0.0;
  // Note we could (Optional) store totals up to n equal spots within memory space to save lookup time (currently O(K), same as GSL)
  // Now loop through items to calculate cumulative probability until it equals or exceeds the random value
  i = 0;
  do {
    cumulative += P[i] * scale;
    if (cumulative >= rnd) {
      break;
    }
    ++i;
  } while (i < K);
  return i;
}

// Works the same as GSL
double cdf_gamma_P( double P, double a, double b ) {
  return stats::pgamma(P, a, b);
}

// Works the same as GSL
double cdf_gamma_Pinv( double P, double a, double b ) {
  // Inverse of cdf_gamma_P, not the inverse gamma distribution
  return stats::qgamma(P, a, b);
}

// Works the same as GSL
double cdf_exponential_Pinv( double P, double mu ) {
  return -mu * log1p (-P);
}

// Works the same as GSL
double inv_incomplete_gamma_p( double percentile, long n )
{
	if( n < 1 )
		return -1;

  return gcem::incomplete_gamma_inv(n, percentile);
}