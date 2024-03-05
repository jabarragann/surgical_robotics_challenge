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
        first_graphics_update = false;
        fill_rigid_bodies_map();
        fill_new_materials_map();
    }

    load_shader_materials();

    // Manually render camera
    afRenderOptions ro;
    ro.m_updateLabels = false;
    m_camera->render(ro);

    restore_original_materials();
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
    afRigidBodyMap::const_iterator it = rigid_bodies_map.begin();

    cout << "Number of objects in simulation:"
         << rigid_bodies_map.size() << endl;

    for (pair<string, afBaseObjectPtr> kv : rigid_bodies_map)
    {
        int idx = shader_config_objects.get_namespace_idx(kv.first);
        if (idx != -1)
        {

            vector<int> rgb = shader_config_objects.get_rgb_at(idx);
            cMaterial *newMat = new cMaterial();
            newMat->m_diffuse.set(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0);
            new_materials_map[kv.first] = newMat;

            cout << "Reconfiguring" << kv.first << endl;
        }
        else // Default material - black
        {
            cMaterial *newMat = new cMaterial();
            newMat->m_diffuse.set(0.0, 0.0, 0.0, 1.0);
            new_materials_map[kv.first] = newMat;
        }
    }
}

void afProcessingShaderConfig::fill_rigid_bodies_map()
{
    afBaseObjectMap *obj_map = m_world_ptr->getRigidBodyMap();
    afBaseObjectMap::const_iterator it = obj_map->begin();

    for (pair<string, afBaseObjectPtr> kv : *obj_map)
    {
        rigid_bodies_map[kv.first] = dynamic_cast<afRigidBody *>(obj_map->at(kv.first));
    }
}

void afProcessingShaderConfig::load_shader_materials()
{
    afRigidBodyMap::const_iterator it = rigid_bodies_map.begin();

    for (pair<string, afRigidBodyPtr> kv : rigid_bodies_map)
    {
        kv.second->m_visualMesh->backupMaterialColors(true);
        kv.second->m_visualMesh->setMaterial(*new_materials_map[kv.first]);
        kv.second->m_visualMesh->m_material->setModificationFlags(true);
    }
}

void afProcessingShaderConfig::restore_original_materials()
{
    afRigidBodyMap::const_iterator it = rigid_bodies_map.begin();

    for (pair<string, afRigidBodyPtr> kv : rigid_bodies_map)
    {
        kv.second->m_visualMesh->restoreMaterialColors(true);
    }
}
