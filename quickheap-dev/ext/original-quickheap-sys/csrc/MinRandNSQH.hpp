/*
 *  MinRandNSQH.hpp
 *  sqh
 *
 *  Created by Rodrigo Paredes on 20-02-09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 *  Min-order Randomized Quickheaps, basic operations
 *  Non-sliding version
 *
 */

#ifndef SQH_MIN_RAND_NSQH_HPP
#define SQH_MIN_RAND_NSQH_HPP

#include <sstream>
#include <math.h>
#include <iostream>
using namespace std;

#include "Stack.hpp"

static unsigned int NSRQHran32State = 0x0accede0; // state variable for pseudo-randon number generator

template<class T> class MinRandNSQH {
protected:
  T *heap;       // array to store the items
  Stack stack;   // stack of pivot positions
  int capacity;  // maximum size of the heap
  
  void swap(int pi, int pj);
  int partition(int pj); // partitioning the segment [0,pj-1]
  
public:
  MinRandNSQH(int size, bool randomSeed = true);
  ~MinRandNSQH();
  string toString() const;
  template<T> friend ostream& operator<< (ostream &os, const MinRandNSQH<T> &rhs);
  
  int size() const;
  bool isEmpty() const;
  void clear();
  
  bool insert(const T &item);
  T findMin();
  T extractMin();
  bool deleteByPos(int pos);
  T getByPos(int pos) const;
};

/*
 * implementation
 */

template<class T> MinRandNSQH<T>::MinRandNSQH(int size, bool randomSeed) :
stack((int)ceil(10.0*log2(size))), capacity(size) {
  heap = new T[size]; // allocating memory for the array
  stack.push(0); // the fictitious pivot points to 0
  if (randomSeed) {
    time_t t;
    NSRQHran32State = (unsigned int)time(&t);
  }
}

template<class T> inline MinRandNSQH<T>::~MinRandNSQH() {
  delete []heap;
}

template<class T> string MinRandNSQH<T>::toString() const { 
  ostringstream oss;
  int size = stack.stack[0];
  oss << "capacity = " << capacity << ", size = " << size << endl;
  oss << "stack: " << stack << endl;
  oss << "heap: ";
  for (int i=0; i<size; i++) oss << heap[i] << " ";
  return oss.str();
}

template<class T> ostream& operator<< (ostream &os, const MinRandNSQH<T> &rhs) {
  os << rhs.toString();
  return os;
}

template<class T> inline void MinRandNSQH<T>::swap(int pi, int pj) {
  T aux = heap[pi];
  heap[pi] = heap[pj];
  heap[pj] = aux;
}

template<class T> inline int MinRandNSQH<T>::size() const {
  return stack.stack[0];
}

template<class T> inline bool MinRandNSQH<T>::isEmpty() const {
  return (stack.stack[0] == 0);
}

template<class T> inline void MinRandNSQH<T>::clear() {
  stack.setTop(0); // emptying the stack
  stack.push(0); // pushing the position of the fictitious pivot
}

template<class T> bool MinRandNSQH<T>::insert(const T &item) {
  int insPos = stack.stack[0];
  if (insPos < capacity) {
    stack.stack[0] = insPos + 1; // moving the fictitious pivot
    int stackSize = stack.size(), i, rootPos;
    
    float randomFloat = 1.0/0xffffffff*NSRQHran32State;
    float fraction;
    T root;
    
    for (i = 1; i < stackSize; i++) {
      fraction = insPos/(insPos+1.0);
      if (randomFloat < fraction ) {
        // keeping the current root
        rootPos = stack.stack[i]; root = heap[rootPos];
        if (item < root) {
          // descending one level
          heap[insPos] = heap[rootPos+1]; // moving the first element
          heap[rootPos+1] = root; // moving the root
          stack.stack[i] = rootPos+1; // updating root position
          insPos = rootPos; // updating the insertion position
          randomFloat /= fraction; // reutilizing the random number
        }
        else if (item > root) {
          // stop, we reach the proper bucket
          NSRQHran32State = 1664525 * NSRQHran32State + 1013904223; // pseudo-random generator...
          break;
        }
        else {
          randomFloat /= fraction; // reutilizing the random number
          // item == root, choosing left or right
          fraction = (rootPos+1.0)/(insPos+1.0);
          if (randomFloat < fraction) {
            // left, descending one level
            heap[insPos] = heap[rootPos+1]; // moving the first element
            heap[rootPos+1] = root; // moving the root
            stack.stack[i] = rootPos+1; // updating root position
            insPos = rootPos; // updating the insertion position
            randomFloat /= fraction; // reutilizing the random number
          }
          else {
            // right, we reach the proper bucket
            NSRQHran32State = 1664525 * NSRQHran32State + 1013904223; // pseudo-random generator...
            break;
          }
        }
      }
      else {
        // flattening the LST
        stack.setTop(i);
        NSRQHran32State = 1664525 * NSRQHran32State + 1013904223; // pseudo-random generator...
        break;
      }
    }
    heap[insPos] = item; // inserting the key in his respective bucket    
    return true; // inserted
  }
  else return false; // run out of space
}

template<class T> int MinRandNSQH<T>::partition(int pj) {
  //we assume that pj is the position of the pivot on top of stack
  //if (pj == 0) return -1; // error, the minimum is on top of stack
  if (pj == 1) return 0; // the leftmost bucket only has 1 element, done
  
  int pivotPos = (int)(1.0/0xffffffff*NSRQHran32State * pj); // computing a random pivot position
  NSRQHran32State = 1664525 * NSRQHran32State + 1013904223; // updating the pseudo-random generator...
  
  T pivot = heap[pivotPos]; // obtaining the pivot
  heap[pivotPos] = heap[0]; // moving the first element to partition a continuous chunk
  int pi = 1; pj--; // fixing the chunk boundaries
  
  // first sweeping
  while ( (pi < pj) && (heap[pi] < pivot) )
    pi++;
  while ( (pi < pj) && (heap[pj] > pivot) )
    pj--;
  if (pi < pj) {
    swap(pi, pj); pi++; pj--;
  }
  
  // continue the sweeping. Now we don't need the (pi < pj) in the inner while's
  while (pi < pj) {
    while (heap[pi] < pivot)
      pi++;
    while (heap[pj] > pivot)
      pj--;
    if (pi < pj) {
      swap(pi, pj); pi++; pj--;
    }
  }
  
  if (heap[pi] >= pivot) pi--;
  heap[0] = heap[pi];
  heap[pi] = pivot;
  
  return pi;
}

template<class T> T MinRandNSQH<T>::findMin() {
  int pivotPos = stack.top();
  while (pivotPos > 0) {
    pivotPos = partition(pivotPos);
    stack.push(pivotPos);
  }
  return heap[0];
}

template<class T> T MinRandNSQH<T>::extractMin() {
  T item = findMin();
  deleteByPos(0);
  return item;
}

template<class T> bool MinRandNSQH<T>::deleteByPos(int pos) {
  if ( (pos < 0) || (pos >= stack.stack[0]) ) return false; // error!, out of boundaries
  
  // looking for the pivot/root position immediately smaller than pos
  int i;
  if (pos == 0) { // special case pos == 0, used in extractMin
    i = stack.size() - 1; // we will move the last element of this chunk, so we need
                          // the position of the next (to the right) pivot/root
  } else { // general case, traversing the stack
    int stackSize = stack.size();
    for (i = 1; i < stackSize; i++)
      if (pos > stack.stack[i]) break;
    i--; // we will move the last element of this chunk, so we need
         // the position of the next (to the right) pivot/root
  }
  
  if (pos == stack.stack[i]) { // if pos is a pivot, we need it's next pivot (next-next)
    stack.setTop(i); // deleting a pivot => flattening
    i--; // going to the next-next pivot
  }
  
  // moving elements and pivots
  int nextRootPos = stack.stack[i];
  heap[pos] = heap[nextRootPos-1]; // moving the last element into pos
  while (i > 0) {
    heap[nextRootPos-1] = heap[nextRootPos]; // moving the pivot
    stack.stack[i] = nextRootPos-1; // updating its position
    pos = nextRootPos; // updating the deletion position
    i--; // ascending one level
    nextRootPos = stack.stack[i]; 
    heap[pos] = heap[nextRootPos-1]; // moving the last element
  }
  stack.stack[0] = nextRootPos-1; // moving the fictitious pivot
  
  return true;
}

template<class T> inline T MinRandNSQH<T>::getByPos(int pos) const {
  //if ( (pos < 0) || (pos >= stack.stack[0]) ) return false; // error!, out of boundaries
  return heap[pos];
}

#endif // SQH_MIN_RAND_NSQH_HPP
