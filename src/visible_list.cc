#include "visible_list.h"

VisibleList::VisibleList() {}

VisibleList::VisibleList(Visible **v, int n) {
  scenery = v;
  list_size = n;
}

VisibleList::~VisibleList() {

  for (int i=0; i<list_size; i++) { // put this in visible list destructor
    delete scenery[i];
  }

  delete scenery;
}

bool VisibleList::intersect(const Ray& r, float tmin, float tmax, Intersection& ixn) const {

  Intersection temp_ixn;
  bool any_intersect = false;
  double current_closest = tmax;

  for (int i = 0; i<list_size; i++) {

    if (scenery[i]->intersect(r, tmin, current_closest, temp_ixn)) {

      any_intersect = true;
      current_closest = temp_ixn.t;
      ixn = temp_ixn;
    }
  }

  return any_intersect;
}
