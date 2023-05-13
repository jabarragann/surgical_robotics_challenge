#include "shaders_config.h"

using namespace std;

afProcessingShaderConfig::afProcessingShaderConfig()
{
    // Constructor is needed for AF_REGISTER_OBJECT_PLUGIN
}

int afProcessingShaderConfig::init(const afBaseObjectPtr a_afObjectPtr, const afBaseObjectAttribsPtr a_objectAttribs)
{
    m_camera = (afCameraPtr)a_afObjectPtr;
    m_world_ptr = m_camera->m_afWorld;
    m_camera->setOverrideRendering(true);

    vector<ShaderConfigObject> config_objects = parse_camera_config();
    shader_config_objects = ShaderConfigObjectVector(config_objects);

    cout << "objects in config" << endl;
    shader_config_objects.print_namespaces();

    return 1;
}

void afProcessingShaderConfig::graphicsUpdate()
{
    if (first_graphics_update) // Obtain pointers to scene objects
    {
        rigid_bodies_map_ptr = m_world_ptr->getRigidBodyMap();
        first_graphics_update = false;

        fill_new_materials_map();
        cout << "first graphics update" << endl;
    }

    // Manually render camera
    afRenderOptions ro;
    ro.m_updateLabels = false;
    m_camera->render(ro);
}

void afProcessingShaderConfig::physicsUpdate(double dt)
{
}

void afProcessingShaderConfig::reset()
{
}

bool afProcessingShaderConfig::close()
{
    return true;
}

vector<ShaderConfigObject> afProcessingShaderConfig::parse_camera_config()
{
    YAML::Node camera_yaml_specs;
    afBaseObjectAttribsPtr camera_specs = m_camera->getAttributes();
    camera_yaml_specs = YAML::Load(camera_specs->getSpecificationData().m_rawData);

    vector<ShaderConfigObject> shader_config_objects;

    if (camera_yaml_specs["preprocessing shaders config"])
    {
        YAML::Node shaders_config = camera_yaml_specs["preprocessing shaders config"];
        YAML::Node::const_iterator it = shaders_config.begin();

        string body_name;
        string body_namespace;
        vector<int> rgb;

        for (it; it != shaders_config.end(); it++)
        {
            body_name = it->first.as<string>();
            ShaderConfigObject config_obj(body_name, shaders_config[body_name]);
            shader_config_objects.push_back(config_obj);
        }
    }
    else
    {
        cout << "INFO! PREPROCESSING SHADER PLUGIN "
             << "No 'preprocessing shaders config' field found in yaml" << endl;
    }

    return shader_config_objects;
}

void afProcessingShaderConfig::fill_new_materials_map()
{
    afBaseObjectMap::const_iterator it = rigid_bodies_map_ptr->begin();

    cout << "Number of objects in simulation:"
         << rigid_bodies_map_ptr->size() << endl;

    for (pair<string, afBaseObjectPtr> kv : *rigid_bodies_map_ptr)
    {
        int idx = shader_config_objects.get_namespace_idx(kv.first);
        if (idx != -1)
        {
            cout << "Reconfiguring" << kv.first << endl;
        }
    }
}
