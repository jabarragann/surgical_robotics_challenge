#define GL_SILENCE_DEPRECATION
#include <afFramework.h>
#include <afAttributes.h>
#include <yaml-cpp/yaml.h>

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
    bool first_graphics_update = false;
    afCameraPtr m_camera;
    afWorldPtr m_world_ptr;
    afVector3d needle_diffuse;

protected:
    bool parse_camera_config();
};

AF_REGISTER_OBJECT_PLUGIN(afProcessingShaderConfig)