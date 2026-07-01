/*
 *  Random.cpp
 *  sqh
 *
 *  Created by Rodrigo Paredes on 24-02-09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 *  Fast uniform pseudo-random generator. Ranges : [0, 2^31) or [0.0, 1.0)
 *  period at least 2^31, random bits: 32
 * 
 */

#ifndef SQH_RANDOM_CPP
#define SQH_RANDOM_CPP

#include "Random.hpp"

// testing the class
//#define DEBUG_SQH_RANDOM
#ifdef DEBUG_SQH_RANDOM

#include <iostream>
using namespace std;

int main(int argc, char** argv){
  cout << "testing random..." << endl;
  
  Random rg;
  cout << rg.randMax << " " << flush << endl;
  cout << rg.nextInt() << " " << rg.nextInt() << " " << rg.nextInt() << " " << flush;
  cout << rg.nextInt() << " " << rg.nextInt() << " " << rg.nextInt() << " " << flush;
  cout << endl;
  
  cout << rg.nextFloat() << " " << rg.nextFloat() << " " << rg.nextFloat() << " " << flush;
  cout << rg.nextFloat() << " " << rg.nextFloat() << " " << rg.nextFloat() << " " << flush;
  cout << endl;
  
  return 0;
}

#endif // DEBUG_SQH_RANDOM

#endif // SQH_RANDOM_CPP
