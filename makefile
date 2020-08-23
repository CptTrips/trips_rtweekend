CXX = g++
CFLAGS = -g -Wall #-fsanitize=address 
LDFLAGS = -lpalette -Llib


raytrace : bin/main.o bin/ray.o bin/sphere.o bin/visible_list.o bin/rand.o bin/metal.o bin/diffuse.o bin/dielectric.o bin/material.o
	$(CXX) $(CFLAGS) -o raytrace bin/main.o bin/ray.o bin/sphere.o bin/visible_list.o bin/rand.o bin/metal.o bin/diffuse.o bin/dielectric.o bin/material.o $(LDFLAGS)

bin/ray.o : src/ray.cc src/ray.h src/vec3.h
	$(CXX) $(CFLAGS) -o bin/ray.o -c -Isrc src/ray.cc

bin/sphere.o : src/sphere.cc src/sphere.h src/visible.h
	$(CXX) $(CFLAGS) -o bin/sphere.o -c -Isrc src/sphere.cc

bin/visible_list.o : src/visible_list.cc src/visible_list.h src/visible.h
	$(CXX) $(CFLAGS) -o bin/visible_list.o -c -Isrc src/visible_list.cc

bin/rand.o : src/rand.cc src/rand.h src/vec3.h
	$(CXX) $(CFLAGS) -o bin/rand.o -c -Isrc src/rand.cc

bin/material.o : src/material.h src/material.cc
	$(CXX) $(CFLAGS) -o bin/material.o -c -Isrc src/material.cc

bin/metal.o : src/metal.cc src/metal.h src/rand.h
	$(CXX) $(CFLAGS) -o bin/metal.o -c -Isrc src/metal.cc

bin/diffuse.o : src/diffuse.cc src/diffuse.h src/rand.h
	$(CXX) $(CFLAGS) -o bin/diffuse.o -c -Isrc src/diffuse.cc

bin/dielectric.o : src/dielectric.cc src/dielectric.h src/rand.h
	$(CXX) $(CFLAGS) -o bin/dielectric.o -c -Isrc src/dielectric.cc

bin/main.o : src/main.cc src/ray.h src/visible_list.h src/sphere.h src/rand.h src/metal.h src/diffuse.h src/dielectric.h include/palette.h
	$(CXX) $(CFLAGS) -c -o bin/main.o -Isrc -Iinclude src/main.cc

.PHONY: clean
clean :
	rm bin/*.o
