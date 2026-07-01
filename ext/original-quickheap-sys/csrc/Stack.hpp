/*
 *  Stack.hpp
 *  sqh
 *
 *  Created by Rodrigo Paredes on 18-02-09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 *  Simple stack of integers for pivot positions
 *
 */

#ifndef SQH_STACK_HPP
#define SQH_STACK_HPP

#include <sstream>

class Stack {
private:
  int tope;   // first free position
  int max;    // maximum size in the current stack
  
public:
  int *stack; // the array containing the elements
  
  Stack(long size);
  ~Stack();
  string toString() const;
  friend ostream& operator<< (ostream &os, const Stack &rhs);
  
  void push(int x);
  int top();
  int pop();
  void setTop(int size);
  int size();
  bool extractItem(int pos);
};

/*
 * implementation
 */

inline Stack::Stack(long size) {
  max = size;
  tope = 0;
  stack = new int[size];
}

inline Stack::~Stack(){
  delete []stack;
}

string Stack::toString() const { 
  ostringstream oss;
  oss << "max = " << max << ", tope = " << tope << ": ";
  for (int i=0; i<tope; i++) oss << stack[i] << " ";
  return oss.str();
}

ostream& operator<< (ostream &os, const Stack &rhs) {
  os << rhs.toString();
  return os;
}

inline void Stack::push(int x) {
  if (tope == max) return;
  stack[tope++] = x;
}

inline int Stack::top() {
  if (tope == 0) return -1;
  return stack[tope - 1];
}

inline int Stack:: pop(){
  if (tope > 0) {
    tope--;
    return stack[tope];
  } else return -1;
}

inline void Stack::setTop(int size) {
  if (size < tope) tope = size;
}

inline int Stack::size() {
  return tope;
}

bool Stack::extractItem(int pos) {
  if (pos < tope) {
    tope--;
    for (int i = pos; i < tope; i++)
      stack[i] = stack[i+1];
    return true; // item successfully extracted
  }
  else return false; // out of boundary
}

#endif // SQH_STACK_HPP
