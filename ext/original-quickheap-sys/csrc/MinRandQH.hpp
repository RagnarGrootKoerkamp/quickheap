/*
 *  MinRandQH.hpp
 *  sqh
 *
 *  Created by Rodrigo Paredes on 20-02-09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 *  Min-order Randomized Quickheaps, basic operations
 *  circular array version
 *
 */

#ifndef SQH_MIN_RAND_QH_HPP
#define SQH_MIN_RAND_QH_HPP

#include <sstream>
#include <math.h>
#include <iostream>
using namespace std;

#include "Stack.hpp"

static unsigned int RQHran32State = 0x0accede0; // state variable for pseudo-randon number generator

template<class T> class MinRandQH {
protected:
  T *heap;       // array to store the items
  Stack stack;   // stack of pivot positions
  int capacity;  // maximum size of the heap, form 2^k
  int modMask;   // bit mask to fast compute modulo, form 2^k - 1
  int idx;       // index to indicate the first cell of the heap
                 // we must ensure that idx < capacity. See extractMin for details
  
  void swap(int pi, int pj);
  int partition(int pj); // partitioning the segment [idx,pj-1]
  
public:
  MinRandQH(int size, bool randomSeed = true);
  ~MinRandQH();
  string toString() const;
  template<T> friend ostream& operator<< (ostream &os, const MinRandQH<T> &rhs);
  
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

template<class T> MinRandQH<T>::MinRandQH(int size, bool randomSeed) :
stack((int)ceil(10.0*log2(2*size))), capacity(8) {
  while (capacity < size) capacity <<= 1; // calculating a proper heap capacity
  modMask = capacity - 1; // and the proper bit mask
  heap = new T[capacity]; // allocating memory for the array
  stack.push(0); // the fictitious pivot points to 0
  if (randomSeed) {
    time_t t;
    RQHran32State = (unsigned int)time(&t);
  }
  idx = 0;
}

template<class T> inline MinRandQH<T>::~MinRandQH() {
  delete []heap;
}

template<class T> string MinRandQH<T>::toString() const { 
  ostringstream oss;
  int lastCell = stack.stack[0] - 1;
  oss << "capacity = " << capacity << ", modMask = " << modMask
    << ", firstCell " << idx << ", lastCell = " << lastCell
    << ", size = " << stack.stack[0] - idx << endl;
  oss << "stack: " << stack << endl;
  oss << "heap: ";
  for (int i=idx; i <= lastCell; i++) oss << heap[i&modMask] << " ";
  return oss.str();
}

template<class T> ostream& operator<< (ostream &os, const MinRandQH<T> &rhs) {
  os << rhs.toString();
  return os;
}

template<class T> inline void MinRandQH<T>::swap(int pi, int pj) {
  // assume that 0 <= pi, pj < capacity!!
  T aux = heap[pi];
  heap[pi] = heap[pj];
  heap[pj] = aux;
}

template<class T> inline int MinRandQH<T>::size() const {
  return stack.stack[0] - idx;
}

template<class T> inline bool MinRandQH<T>::isEmpty() const {
  return (stack.stack[0] == idx);
}

template<class T> inline void MinRandQH<T>::clear() {
  stack.setTop(0); // emptying the stack
  stack.push(0); // pushing the position of the fictitious pivot
  idx = 0; // starting in cell 0, of course
}

template<class T> bool MinRandQH<T>::insert(const T &item) {
  int insPos = stack.stack[0];
  if (insPos == capacity + idx) return false; // run out of space
  
  // we have space, let's insert the item
  stack.stack[0] = insPos + 1; // moving the fictitious pivot
  int stackSize = stack.size(), i, rootPos;
  
  float randomFloat = 1.0/0xffffffff*RQHran32State;
  float fraction;
  T root;
    
  for (i = 1; i < stackSize; i++) {
    fraction = (insPos-idx)/(insPos-idx+1.0);
    if (randomFloat < fraction ) {
      // keeping the current root
      rootPos = stack.stack[i]; root = heap[rootPos&modMask];
      if (item < root) {
        // descending one level
        heap[insPos&modMask] = heap[(rootPos+1)&modMask]; // moving the first element
        heap[(rootPos+1)&modMask] = root; // moving the root
        stack.stack[i] = rootPos+1; // updating root position
        insPos = rootPos; // updating the insertion position
        randomFloat /= fraction; // reutilizing the random number
      }
      else if (item > root) {
        // stop, we reach the proper bucket
        RQHran32State = 1664525 * RQHran32State + 1013904223; // updating the pseudo-random generator
        break;
      }
      else {
        randomFloat /= fraction; // reutilizing the random number
        // item == root, choosing left or right
        fraction = (rootPos-idx+1.0)/(insPos-idx+1.0);
        if (randomFloat < fraction) {
          // left, descending one level
          heap[insPos&modMask] = heap[(rootPos+1)&modMask]; // moving the first element
          heap[(rootPos+1)&modMask] = root; // moving the root
          stack.stack[i] = rootPos+1; // updating root position
          insPos = rootPos; // updating the insertion position
          randomFloat /= fraction; // reutilizing the random number
        }
        else {
          // right, we reach the proper bucket
          RQHran32State = 1664525 * RQHran32State + 1013904223; // updating the pseudo-random generator
          break;
        }
      }
    }
    else {
      // flattening the LST
      stack.setTop(i);
      RQHran32State = 1664525 * RQHran32State + 1013904223; // updating the pseudo-random generator
      break;
    }
  }
  heap[insPos&modMask] = item; // inserting the key in his respective bucket    
  return true; // inserted
}

template<class T> int MinRandQH<T>::partition(int pj) {
  // partitioning the segment [idx,pj-1]. So, we assume that pj is the position of the
  // pivot on top of stack
  //if (pj == idx) return -1; // error, the minimum is on top of stack
  if (pj == idx + 1) return idx; // the leftmost bucket only has 1 element, done
  
  int pivotPos = (int)(1.0/0xffffffff*RQHran32State * (pj-idx)) + idx; // computing a random pivot position
  RQHran32State = 1664525 * RQHran32State + 1013904223; // updating the pseudo-random generator...
  T pivot = heap[pivotPos&modMask]; // obtaining the pivot
  heap[pivotPos&modMask] = heap[idx]; // moving the first element to partition a ``continuous'' chunk
  int pi = idx + 1; pj--; // fixing the chunk boundaries
  
  // first sweeping
  while ( (pi < pj) && (heap[pi&modMask] < pivot) )
    pi++;
  while ( (pi < pj) && (heap[pj&modMask] > pivot) )
    pj--;
  if (pi < pj) {
    swap(pi&modMask, pj&modMask); pi++; pj--;
  }
  
  // continue the sweeping. Now we don't need the (pi < pj) in the inner while's
  bool reduced = false;
  while ( (pj >= modMask) && (pi < pj) ) { // if pj >= modMask we need the &modMask
    while (heap[pi&modMask] < pivot)
      pi++;
    while (heap[pj&modMask] > pivot)
      pj--;
    if (pi < pj) {
      swap(pi&modMask, pj&modMask); pi++; pj--;
      if (pi > capacity) { // pi passes capacity, we can reduce both indices
        pi -= capacity; pj -= capacity; reduced = true;
      }
    }
  }
  while (pi < pj) { // fast case, both pi and pj are less than modMask
    while (heap[pi] < pivot)
      pi++;
    while (heap[pj] > pivot)
      pj--;
    if (pi < pj) {
      swap(pi, pj); pi++; pj--;
    }
  }
  if (reduced) {
    pi += capacity;
    // pj += capacity; // we don't need to do this as we don't use pj any more
  }
  
  if (heap[pi&modMask] >= pivot) pi--;
  heap[idx] = heap[pi&modMask];
  heap[pi&modMask] = pivot;
  
  return pi;
}

template<class T> T MinRandQH<T>::findMin() {
  int pivotPos = stack.top();
  while (pivotPos > idx) {
    pivotPos = partition(pivotPos);
    stack.push(pivotPos);
  }
  return heap[idx];
}

template<class T> T MinRandQH<T>::extractMin() {
  T item = findMin();
  idx++; // go to the cell next to the minimum
  stack.pop(); // pop the pivot position of the minimum
  
  if (idx == capacity) {
    // idx go out of the heap, reducing all the pivot positions in the stack
    // to start from 0
    int stackSize = stack.size(), i;
    for (i = 0; i < stackSize; i++)
      stack.stack[i] -= capacity;
    idx = 0; // we start from 0
  }
  return item;
}

template<class T> bool MinRandQH<T>::deleteByPos(int pos) {
  if ( (pos < idx) || (pos >= stack.stack[0]) ) return false; // error!, out of boundaries
  
  // looking for the pivot/root position immediately smaller than pos
  int i;
  if (pos == idx) { // special case pos == idx, alternative implementation for extractMin
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
  heap[pos&modMask] = heap[(nextRootPos-1)&modMask]; // moving the last element into pos
  while (i > 0) {
    heap[(nextRootPos-1)&modMask] = heap[nextRootPos&modMask]; // moving the pivot
    stack.stack[i] = nextRootPos-1; // updating its position
    pos = nextRootPos; // updating the deletion position
    i--; // ascending one level
    nextRootPos = stack.stack[i]; 
    heap[pos&modMask] = heap[(nextRootPos-1)&modMask]; // moving the last element
  }
  stack.stack[0] = nextRootPos-1; // moving the fictitious pivot
  
  return true;
}

template<class T> inline T MinRandQH<T>::getByPos(int pos) const {
  //if ( (pos < idx) || (pos >= stack.stack[0]) ) return false; // error!, out of boundaries
  return heap[pos&modMask];
}

#endif // SQH_MIN_RAND_QH_HPP
