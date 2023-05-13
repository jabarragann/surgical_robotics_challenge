#include "ShaderConfigObject.h"

ShaderConfigObject::ShaderConfigObject(string name, YAML::Node config_node)
{
    body_name = name;
    if (config_node["namespace"].IsDefined())
        body_namespace = config_node["namespace"].as<string>();
    else
        body_namespace = "/ambf/env/";

    if (config_node["rgb"].IsDefined())
        rgb = config_node["rgb"].as<vector<int>>();
    else
        rgb = {255, 255, 255};
}

string ShaderConfigObject::get_full_namespace()
{
    return body_namespace + "BODY " + body_name;
}

ShaderConfigObjectVector::ShaderConfigObjectVector(vector<ShaderConfigObject> objs)
{
    for (ShaderConfigObject obj : objs)
    {
        shader_config_objects.push_back(obj);
        config_objects_namespaces.push_back(obj.get_full_namespace());
    }
}

void ShaderConfigObjectVector::add_item(ShaderConfigObject item)
{
    shader_config_objects.push_back(item);
    config_objects_namespaces.push_back(item.get_full_namespace());
}

void ShaderConfigObjectVector::print_namespaces()
{
    for (string ns : config_objects_namespaces)
    {
        cout << ns << endl;
    }
}

int ShaderConfigObjectVector::get_namespace_idx(string ns)
{
    vector<string>::iterator it;
    it = find(config_objects_namespaces.begin(), config_objects_namespaces.end(), ns);

    if (it != config_objects_namespaces.end())
        return it - config_objects_namespaces.begin();
    else
        return -1;

    return 0;
}
