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

    // parse_camera_config();

    return 1;
}

void afProcessingShaderConfig::graphicsUpdate()
{
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

bool afProcessingShaderConfig::parse_camera_config()
{
    YAML::Node camera_yaml_specs;
    afBaseObjectAttribsPtr camera_specs = m_camera->getAttributes();
    camera_yaml_specs = YAML::Load(camera_specs->getSpecificationData().m_rawData);

    return true;
}
