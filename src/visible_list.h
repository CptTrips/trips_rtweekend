#ifndef VISIBLE_LIST_H
#define VISIBLE_LIST_H

#include "visible.h"

class VisibleList: public Visible {
  public:
    VisibleList();
    VisibleList(Visible **v, int n);
    ~VisibleList();
    virtual bool intersect(const Ray& r, float tmin, float tmax, Intersection& ixn) const;

    Visible **scenery;
    int list_size;
};


#endif
