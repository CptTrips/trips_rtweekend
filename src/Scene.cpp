#include "Scene.h"


Scene::~Scene()
{
	std::cout << "Deleting scene" << std::endl;
	for (auto v : scene_objects)
		delete v;
}

std::vector<Visible*>::iterator& Scene::begin()
{
	return scene_objects.begin();
}

std::vector<Visible*>::const_iterator& Scene::begin() const
{
	return scene_objects.begin();
}

std::vector<Visible*>::iterator& Scene::end()
{
	return scene_objects.end();
}

std::vector<Visible*>::const_iterator& Scene::end() const
{
	return scene_objects.end();
}
