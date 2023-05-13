#define GL_SILENCE_DEPRECATION
#include <afFramework.h>
#include <afAttributes.h>
#include <yaml-cpp/yaml.h>
#include <map>
#include "ShaderConfigObject.h"

using namespace std;
using namespace ambf;

class afProcessingShaderConfig : public afObjectPlugin
{
public:
    afProcessingShaderConfig();
    virtual int init(const afBaseObjectPtr a_afObjectPtr, const afBaseObjectAttribsPtr a_objectAttribs) override;
    virtual void graphicsUpdate() override;
    virtual void physicsUpdate(double dt) override;
    virtual void reset() override;
    virtual bool close() override;

protected:
    bool first_graphics_update = true;
    afCameraPtr m_camera;
    afWorldPtr m_world_ptr;
    afVector3d needle_diffuse;
    afBaseObjectMap *rigid_bodies_map_ptr;
    map<string, cMaterial> new_materials_map;
    ShaderConfigObjectVector shader_config_objects;

protected:
    vector<ShaderConfigObject> parse_camera_config();
    void fill_new_materials_map();
};

AF_REGISTER_OBJECT_PLUGIN(afProcessingShaderConfig)