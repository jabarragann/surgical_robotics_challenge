#pragma once
#include <afFramework.h>
#include <afAttributes.h>
#include <yaml-cpp/yaml.h>

using namespace std;

class ShaderConfigObject
{
public:
    string body_namespace;
    string body_name;
    vector<int> rgb;

    ShaderConfigObject(string obj_namespace, string name, vector<int> rgb)
        : body_namespace(obj_namespace), body_name(name), rgb(rgb){};

    ShaderConfigObject(string name, YAML::Node node);

    string get_full_namespace();
};

class ShaderConfigObjectVector
{
public:
    ShaderConfigObjectVector() = default;
    ShaderConfigObjectVector(vector<ShaderConfigObject> objs);
    void add_item(ShaderConfigObject item);
    size_t size() { return shader_config_objects.size(); };

    void print_namespaces();
    int get_namespace_idx(string ns);

private:
    vector<ShaderConfigObject> shader_config_objects;
    vector<string> config_objects_namespaces;
};