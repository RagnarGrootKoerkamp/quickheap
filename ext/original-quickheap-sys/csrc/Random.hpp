/*
 *  Random.hpp
 *  sqh
 *
 *  Created by Rodrigo Paredes on 24-02-09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 *  Fast uniform pseudo-random generator. Ranges : [0, 2^31) or [0.0, 1.0)
 *  period at least 2^31, random bits: 32
 * 
 */

#ifndef SQH_RANDOM_HPP
#define SQH_RANDOM_HPP

class Random {
private:
  unsigned int ran32State;
public:
  static const int randMax = 0x7fffffff; // 2^31 - 1
  Random();
  Random(unsigned int seed);
  ~Random();
  void changeSeed(unsigned int seed); // changing ran32State
  int nextInt();
  float nextFloat();
};

/*
 * implementation
 */
inline Random::Random() : ran32State(0x0accede0){ }

inline Random::Random(unsigned int seed) : ran32State(seed){ }

inline Random::~Random() {}

inline void Random::changeSeed(unsigned int seed) { ran32State = seed; }

inline int Random::nextInt() {
  ran32State = 1664525 * ran32State + 1013904223;
  return ran32State >> 1; // changing the range [0,2^32) to [0,2^31)
}

inline float Random::nextFloat() {
  ran32State = 1664525 * ran32State + 1013904223;
  return 1.0/0xffffffff*ran32State; // changing the range [0,2^32) to [0.0,1.0)
}

#endif // SQH_RANDOM_HPP
