/*
 *  Stack.cpp
 *  sqh
 *
 *  Created by Rodrigo Paredes on 18-02-09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 *  Simple stack of integers for pivot positions
 *
 */

#ifndef SQH_STACK_CPP
#define SQH_STACK_CPP

#include <iostream>
using namespace std;

#include "Stack.hpp"

// testing the class
#define DEBUG_SQH_STACK
#ifdef DEBUG_SQH_STACK

int main(int argc, char** argv){
  cout << "testing stacks..." << endl;
  
  Stack s(20); cout << s << endl;
  s.push(10); s.push(15); s.push(20); s.push(9); cout << s << endl;
  s.push(12); s.push(15); s.push(21); s.push(9); cout << s << endl;
  cout << "s.size() " << s.size() << endl;
  s.setTop(6); cout << "changing size!!, s.size() " << s.size() << " " << s << endl;
  cout << s.top() << " " << s.pop() << " " << s.pop() << endl << s << endl;
  cout << s.stack[0] << " " << s.stack[1] << endl;
  
  return 0;
}

#endif // DEBUG_SQH_STACK

#endif // SQH_STACK_CPP
