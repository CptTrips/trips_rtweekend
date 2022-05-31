#pragma once
#include <vector>
#include "visible.h"


class Scene
{
	std::vector<Visible*> scene_objects;

	Scene(const Scene& s) = delete;

	Scene& operator=(const Scene& s) = delete;


public:

	Scene(std::vector<Visible*> scene_objects) : scene_objects(scene_objects) {}

	Scene(Scene&& s);

	Scene&& operator=(Scene&& s);

	std::vector<Visible*>::iterator& begin();
	std::vector<Visible*>::const_iterator& begin() const;

	std::vector<Visible*>::iterator& end();
	std::vector<Visible*>::const_iterator& end() const;

	~Scene();
};